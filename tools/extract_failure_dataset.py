#!/usr/bin/env python3
"""手動修正ラベル × 追跡時特徴量 を結合して失敗検知の学習セットを吐く。

案②（失敗検知）のピース2。projects/*.mvproj.json を走査し、
`track_features`（追跡時のフレーム単位特徴量）を `keyframes[].src` の
手動/自動ラベルと結合し、1フレーム1行の CSV を出力する。

ラベル定義:
  正例(1) = 手動修正フレーム（src=="manual"）。人が直した = AI が失敗した。
            うち visible==False は「不在化（画面外へ退避）」＝ kind="absent"。
  負例(0) = 自動フレーム（src=="auto"）かつ 最寄り手動フレームから
            NEIGHBOR フレーム超離れている（再伝播ゾーンを避ける）。
            **かつ「レビュー済み区間」の内側**（--reviewed-only 既定ON。下記）。
  不明(-1) = レビュー済み区間の外の自動フレーム（kind="unreviewed"）。
            **行は残すが負例にはしない。** 実ゲートの th はトラック全フレームの
            lmk_conf 中央値から決まるので、行ごと落とすと下流が中央値を部分集合
            から計算してしまう。レビュー予算の分母にも必要。
  除外    = 手動の近傍(<=NEIGHBOR)の自動フレーム（伝播ゾーン, 曖昧）。
            特徴量の無いフレームは学習に使えないので行を出さない
            （ただし「特徴量欠損の手動」件数はサマリで報告する）。

⚠ **「auto」は「AIが成功した」を意味しない。「人が触らなかった」を意味する。**
人が見ていない区間の auto を負例にすると、**未ラベルの失敗を「成功例」として
教える**ことになる。2026-07-17 に実害を確認:
  hy_clip1 は 17,822 フレーム中 人が触ったのは f233-2501（13%）だけ。残り
  15,082 フレームを負例にしていたため 自動の lmk_conf 中央値が 0.0063→0.0120 に
  押し上げられ、**手動(0.0096) < 自動(0.0120) という弁別の逆転**が起きていた
  （＝「この動画では特徴量が効かない」という誤った結論を生んだ）。
  未レビュー区間には実際にマスクが頬に乗って口が露出したフレーム(f8390 等)が
  あり、それらが全て label=0 だった。レビュー済み区間だけに絞ると 1.52x で
  逆転は解消する。

そこで **負例は「レビュー済み区間」= [最初の手動フレーム, 最後の手動フレーム]
の内側からのみ採る**。人が触った形跡のある範囲だけを「見た」とみなす近似。
手動が1件も無いトラックは全体が未レビューなので負例を1件も出さない
（例: hy_clip2 は手動0件で 26,168 行すべてが未ラベルだった）。

出力は特徴量のあるフレームのみ。設計:
  .company/engineering/docs/manual-correction-as-failure-labels.md

使い方:
  python tools/extract_failure_dataset.py            # projects/ 全体 → dataset.csv
  python tools/extract_failure_dataset.py --neighbor 8 --out /tmp/ds.csv
  python tools/extract_failure_dataset.py --projects projects --summary-only
  python tools/extract_failure_dataset.py --no-reviewed-only   # 旧挙動（非推奨）
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
from pathlib import Path

FEATURE_COLS = ["lmk_conf", "conf", "dev", "gated", "gate_th"]
ROW_COLS = (["project", "person", "frame"] + FEATURE_COLS
            + ["src", "visible", "dist_to_manual", "label", "kind"])


def _nearest_dist(sorted_manual: list[int], f: int) -> float:
    """f から最寄りの手動フレームまでの距離（手動が無ければ inf）。"""
    if not sorted_manual:
        return float("inf")
    i = bisect.bisect_left(sorted_manual, f)
    best = float("inf")
    if i < len(sorted_manual):
        best = min(best, sorted_manual[i] - f)
    if i > 0:
        best = min(best, f - sorted_manual[i - 1])
    return float(best)


def iter_project_rows(proj: dict, neighbor: int, reviewed_only: bool = True):
    """1プロジェクト分の行と統計を生成する。"""
    pid = proj.get("id", "?")
    keyframes = proj.get("keyframes", {}) or {}
    track_features = proj.get("track_features", {}) or {}

    stats = {"pos_manual": 0, "pos_absent": 0, "neg": 0,
             "excl_zone": 0, "excl_unreviewed": 0,
             "manual_no_feat": 0, "feat_no_kf": 0}

    for person_id, feats in track_features.items():
        kfs = keyframes.get(person_id, {}) or {}
        manual = sorted(int(f) for f, k in kfs.items()
                        if (k or {}).get("src") == "manual")

        # レビュー済み区間 = 最初の手動〜最後の手動。人が触った形跡のある範囲だけを
        # 「見た」とみなす近似（docstring の ⚠ 参照）。手動が無ければ全体が未レビュー。
        if reviewed_only:
            rev_lo, rev_hi = ((manual[0], manual[-1]) if manual
                              else (None, None))
        else:
            rev_lo, rev_hi = (float("-inf"), float("inf"))

        # 特徴量欠損の手動（正例だが教師にできない）件数をカウント
        for mf in manual:
            if str(mf) not in feats:
                stats["manual_no_feat"] += 1

        for fstr, feat in feats.items():
            f = int(fstr)
            kf = kfs.get(fstr)
            if kf is None:
                # 特徴量はあるが keyframe が無い（追跡が途切れた等）→ ラベル不能
                stats["feat_no_kf"] += 1
                continue
            src = kf.get("src")
            visible = kf.get("visible", True)
            dist = _nearest_dist(manual, f)

            if src == "manual":
                label = 1
                if not visible:
                    kind = "absent"
                    stats["pos_absent"] += 1
                else:
                    kind = "manual"
                    stats["pos_manual"] += 1
            elif src == "auto" and dist > neighbor:
                # 未レビュー区間の auto は「成功」ではなく「見ていない」。
                # **負例にはしないが行は残す（label=-1）**。
                # ⚠ 捨ててはいけない: 実ゲートの th は「トラック全フレーム」の
                # lmk_conf 中央値から決まる（tracker._gate_mouth）。行を落とすと
                # 下流が中央値を部分集合から計算してしまい、オフライン予測が
                # 実ゲートとズレる（2026-07-17 に実際に踏んだ:
                # hy_clip1 の median が 0.0120→0.0063 に化けた）。
                # また「レビュー予算」の分母は人が見る全フレームなので、
                # 未レビュー行も分母には要る。
                if rev_lo is None or not (rev_lo <= f <= rev_hi):
                    label = -1
                    kind = "unreviewed"
                    stats["excl_unreviewed"] += 1
                else:
                    label = 0
                    kind = "auto_far"
                    stats["neg"] += 1
            else:  # auto かつ手動の近傍 = 伝播ゾーン → 除外
                stats["excl_zone"] += 1
                continue

            row = {
                "project": pid, "person": person_id, "frame": f,
                "src": src, "visible": int(bool(visible)),
                "dist_to_manual": ("" if dist == float("inf")
                                   else int(dist)),
                "label": label, "kind": kind,
            }
            for c in FEATURE_COLS:
                v = feat.get(c, "")
                row[c] = int(v) if isinstance(v, bool) else v
            yield row, None

    yield None, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects", default="projects",
                    help="プロジェクト .mvproj.json のディレクトリ")
    ap.add_argument("--out", default="dataset.csv", help="出力 CSV パス")
    ap.add_argument("--neighbor", type=int, default=5,
                    help="手動フレームからこの距離以内の auto は除外（伝播ゾーン）")
    ap.add_argument("--summary-only", action="store_true",
                    help="CSV を書かずサマリだけ表示")
    ap.add_argument("--no-reviewed-only", dest="reviewed_only",
                    action="store_false",
                    help="レビュー済み区間の外の auto も負例にする（旧挙動・非推奨。"
                         "未ラベルの失敗を「成功例」として教えることになる）")
    ap.set_defaults(reviewed_only=True)
    args = ap.parse_args(argv)

    proj_dir = Path(args.projects)
    files = sorted(proj_dir.glob("*.mvproj.json"))
    if not files:
        print(f"プロジェクトが見つかりません: {proj_dir}", file=sys.stderr)
        return 1

    total = {"pos_manual": 0, "pos_absent": 0, "neg": 0,
             "excl_zone": 0, "excl_unreviewed": 0,
             "manual_no_feat": 0, "feat_no_kf": 0}
    n_rows = 0
    writer = None
    out_f = None
    if not args.summary_only:
        out_f = open(args.out, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(out_f, fieldnames=ROW_COLS)
        writer.writeheader()

    per_project = []
    for fp in files:
        try:
            proj = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  skip {fp.name}: {e}", file=sys.stderr)
            continue
        p_rows = 0
        for row, stats in iter_project_rows(proj, args.neighbor,
                                            args.reviewed_only):
            if row is not None:
                if writer:
                    writer.writerow(row)
                n_rows += 1
                p_rows += 1
            if stats is not None:
                for k, v in stats.items():
                    total[k] += v
                has_tf = bool(proj.get("track_features"))
                per_project.append((proj.get("id", fp.stem), p_rows, has_tf))

    if out_f:
        out_f.close()

    pos = total["pos_manual"] + total["pos_absent"]
    print("=" * 60)
    print(f"プロジェクト数: {len(files)}")
    print(f"出力行数(特徴量あり): {n_rows}"
          + ("" if args.summary_only else f"  -> {args.out}"))
    print(f"  正例 label=1: {pos}"
          f"  (manual={total['pos_manual']}, absent={total['pos_absent']})")
    print(f"  負例 label=0: {total['neg']}  (auto_far"
          + (", レビュー済み区間のみ)" if args.reviewed_only else ")"))
    print(f"  除外(伝播ゾーン): {total['excl_zone']}")
    if args.reviewed_only:
        print(f"  除外(未レビュー区間の auto): {total['excl_unreviewed']}"
              "  ← 人が見ていない＝成功の証拠が無い")
    else:
        print("  ⚠ --no-reviewed-only: 未レビュー区間の auto も負例に含めた。"
              "未ラベルの失敗を「成功例」として教えている可能性がある")
    print("-" * 60)
    print("データ品質:")
    print(f"  特徴量欠損の手動(正例だが教師化不可): {total['manual_no_feat']}"
          "  ← ブートストラップ(全体reseed)で埋まる")
    print(f"  特徴量あるが keyframe 無し: {total['feat_no_kf']}")
    if args.reviewed_only and total["excl_unreviewed"]:
        denom = total["neg"] + total["excl_unreviewed"]
        print(f"  レビュー率(負例候補のうち実際に見た割合): "
              f"{100 * total['neg'] / denom:.0f}%"
              f"  ({total['neg']}/{denom})")
    if pos and total["neg"]:
        print(f"  正例:負例 = 1 : {total['neg'] / pos:.1f}")
    print("-" * 60)
    print("プロジェクト別(特徴量あり=track_features 保持):")
    for pid, rows, has_tf in per_project:
        flag = "" if has_tf else "  [track_features なし=未ブートストラップ]"
        print(f"  {pid:24s} rows={rows}{flag}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
