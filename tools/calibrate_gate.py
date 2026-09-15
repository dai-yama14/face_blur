#!/usr/bin/env python3
"""
calibrate_gate.py - 口元ゲートのしきい値を手動修正ラベルで較正する（案② ピース3）

`tools/extract_failure_dataset.py` が吐いたデータセットを入力に、
`tracker._gate_mouth` の判定をオフラインで再現し、しきい値
（MOUTH_GATE_K / MOUTH_GATE_DEVIATION / FLOOR / CEIL）を掃引して
**「失敗リコール @ 固定手動バジェット」** で評価する。

設計: .company/engineering/docs/manual-correction-as-failure-labels.md

## 何をするか / しないか
- する: 「人が実際に直したフレーム（label=1）を、レビュー対象を膨らませずに
  拾えるか」でしきい値を選ぶ。**別動画クロス検証（leave-one-video-out）が本体**で、
  「動画Aで較正した値が動画Bで手動量を爆発させないか」を直接測る（過去の A→B 悪化の再発防止）。
- しない: 配置（楕円の位置・サイズ）の変更。ゲートの出力は「実測を捨てて補間に倒す
  ／レビュー送り」だけなので、外しても手動が少し増えるだけでマスクは壊れない。

## 使い方
    python tools/extract_failure_dataset.py --out /tmp/ds.csv
    python tools/calibrate_gate.py --dataset /tmp/ds.csv --budget 0.10

## ⚠ 近似について（重要）
実ゲートの `dev` は「trusted フレームからの補間」との食い違いで、trusted は
th（=K×median）に依存する。つまり K を変えると dev も変わる。しかし保存済みの
`dev` は**出荷時の th で計算された値**であり、掃引しても再計算できない
（pre-gate の refined 楕円は保存されていない）。

したがって本ツールの dev 依存部分は近似:
  - K / FLOOR / CEIL の掃引 = lmk_conf 側のしきい値。これは正確に再現できる
  - DEVIATION の掃引 = 保存済み dev に対する判定。th が出荷値から大きく離れると誤差が乗る
候補が絞れたら、実際にその定数で再測定して確認すること（--verify-hint 参照）。
"""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict

# tracker.py の現行値（= 出荷時のベースライン）
CUR_K = 2.0
CUR_FLOOR = 0.012
CUR_CEIL = 0.060
CUR_DEVIATION = 0.5
CUR_MIN_SAMPLES = 20
CONF_FAILED = 9.9


def load(path: str) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                r["lmk_conf"] = float(r["lmk_conf"]) if r["lmk_conf"] else None
                r["dev"] = float(r["dev"]) if r["dev"] else None
                r["conf"] = float(r["conf"]) if r["conf"] else None
                r["label"] = int(r["label"])
            except (ValueError, KeyError):
                continue
            rows.append(r)
    return rows


def track_key(r: dict) -> tuple:
    """しきい値はトラック単位で自己校正される（_gate_mouth と同じ粒度）。"""
    return (r["project"], r["person"])


def track_threshold(rows: list[dict], k: float, floor: float,
                    ceil: float, gate_off_above: float | None = None
                    ) -> float | None:
    """_gate_mouth と同じ式で、そのトラックの th を求める。

    ⚠ 中央値は **そのトラックの全フレーム**（label=-1 の未レビュー行を含む）
    から取ること。実ゲート（tracker._gate_mouth）がそうしているため。
    負例だけの部分集合から計算すると th がズレて予測が実機と食い違う。

    実測できた lmk_conf（< CONF_FAILED）が MIN_SAMPLES 未満ならゲートは
    走らない（= None を返し、そのトラックは1フレームも棄却しない）。

    gate_off_above: **トラック単位の有効/無効判定**（2026-07-17 追加）。
    そのトラックの median lmk_conf がこの値以上なら、ゲート自体を無効化する。

    根拠: ゲートは「th = K×median で分離できる裾があり、その裾が失敗に対応する」
    ことを前提にする。だが median 自体が既に悪い水準（≒FLOOR）まで上がっている
    トラックでは分布にコントラストが無く、裾は失敗と対応しない。実測でも
    median 0.0113〜0.0120 の hy_clip4/hy_clip1 は弁別比 0.95x/0.79x（＝手動の方が
    lmk_conf が低い＝逆転）で、ゲートは無関係なフレームを立てて予算だけ食った
    （クロス検証で budget 11.6% / 22.6%）。**ラベル無しで観測できる median だけで
    この状態を検出できる**のが利点。
    """
    vals = [r["lmk_conf"] for r in rows
            if r["lmk_conf"] is not None and r["lmk_conf"] < CONF_FAILED]
    if len(vals) < CUR_MIN_SAMPLES:
        return None
    med = statistics.median(vals)
    if gate_off_above is not None and med >= gate_off_above:
        return None                    # このトラックではゲートを使わない
    return min(ceil, max(floor, k * med))


def predict(rows: list[dict], k: float, floor: float, ceil: float,
            deviation: float, gate_off_above: float | None = None
            ) -> list[bool]:
    """_gate_mouth の判定を再現: (lmk_conf > th) AND (dev > DEVIATION)。"""
    by_track = defaultdict(list)
    for r in rows:
        by_track[track_key(r)].append(r)
    th_of = {t: track_threshold(rs, k, floor, ceil, gate_off_above)
             for t, rs in by_track.items()}
    out = []
    for r in rows:
        th = th_of[track_key(r)]
        if th is None or r["lmk_conf"] is None or r["dev"] is None:
            out.append(False)          # ゲート不発 = 棄却しない（安全側）
            continue
        out.append(r["lmk_conf"] > th and r["dev"] > deviation)
    return out


def evaluate(rows: list[dict], flags: list[bool]) -> dict:
    """失敗リコールとレビュー予算を測る。

    ラベルは3値（extract_failure_dataset.py）:
      1  = 人が直した = AI が失敗した（正例）
      0  = 人が見たうえで直さなかった = AI が成功した（負例）
      -1 = 人が見ていない = **成功か失敗か不明**

    recall  : label=1 のうちゲートが拾えた割合
    budget  : **全フレーム**のうちゲートが立った割合（= 人が確認する量。
              label=-1 も人が見る対象なので分母に含める）
    fpr     : label=0 のうち立った割合（＝確実に無駄な指摘）
    unknown : label=-1 のうち立った割合。**無駄とは断定できない**
              （未ラベルの本物の失敗を含みうる）。ここを「無駄」に数えると、
              レビューが薄い動画ほどゲートが悪く見える誤りを生む
              （2026-07-17 に実際に踏んだ: 「無駄 2,006件削減」は
              未レビューフレームを無駄と数えていただけで、実際は9件だった）。
    """
    pos = [i for i, r in enumerate(rows) if r["label"] == 1]
    neg = [i for i, r in enumerate(rows) if r["label"] == 0]
    unk = [i for i, r in enumerate(rows) if r["label"] == -1]
    if not pos:
        return {}
    hit = sum(1 for i in pos if flags[i])
    fp = sum(1 for i in neg if flags[i])
    fu = sum(1 for i in unk if flags[i])
    n_flag = sum(flags)
    return {
        "recall": hit / len(pos),
        "budget": n_flag / len(rows),
        "fpr": fp / len(neg) if neg else 0.0,
        "unknown_rate": fu / len(unk) if unk else 0.0,
        "n_pos": len(pos), "n_neg": len(neg), "n_unk": len(unk),
        "hit": hit, "fp": fp, "flagged_unknown": fu, "n_flag": n_flag,
    }


def sweep(rows: list[dict], grid: dict, budget: float) -> list[tuple]:
    """予算以内で最大リコールになる組み合わせを探す。"""
    results = []
    for k in grid["k"]:
        for dv in grid["deviation"]:
            for fl in grid["floor"]:
                for ce in grid["ceil"]:
                    if fl >= ce:
                        continue
                    m = evaluate(rows, predict(rows, k, fl, ce, dv))
                    if not m:
                        continue
                    results.append(((k, dv, fl, ce), m))
    # 予算内で recall 最大 → 同点なら budget が小さい方
    ok = [r for r in results if r[1]["budget"] <= budget]
    ok.sort(key=lambda r: (-r[1]["recall"], r[1]["budget"]))
    return ok or sorted(results, key=lambda r: r[1]["budget"])


def cross_validate(rows: list[dict], grid: dict, budget: float) -> None:
    """leave-one-video-out。**これが本体**。

    動画Aたちで選んだしきい値が、未見の動画Bで
    「手動量を爆発させずに失敗を拾えるか」を測る。過去の A→B 悪化をそのまま測る指標。
    """
    projects = sorted({r["project"] for r in rows})
    print("\n" + "=" * 78)
    print("別動画クロス検証（leave-one-video-out）: 学習に使っていない動画での成績")
    print("=" * 78)
    print("%-18s %-24s %7s %7s %7s %7s" % (
        "held-out(未見)", "train で選んだ(K,DEV,FL,CE)", "recall", "budget",
        "確実無駄", "レビュー率"))
    agg = []
    for held in projects:
        tr = [r for r in rows if r["project"] != held]
        te = [r for r in rows if r["project"] == held]
        if not tr or not te or not any(r["label"] == 1 for r in te):
            print("%-18s %s" % (held.split("-")[0], "（正例が無いのでスキップ）"))
            continue
        best = sweep(tr, grid, budget)
        if not best:
            continue
        params = best[0][0]
        m = evaluate(te, predict(te, *[params[0], params[2], params[3],
                                       params[1]]))
        if not m:
            continue
        agg.append(m)
        # レビュー率 = その動画で「見た」フレームの割合。低いほど数字の信用が落ちる
        rev = m["n_neg"] / max(1, m["n_neg"] + m["n_unk"])
        print("%-18s %-24s %6.1f%% %6.1f%% %7d %6.0f%%" % (
            held.split("-")[0], "(%.1f, %.2f, %.3f, %.3f)" % params,
            100 * m["recall"], 100 * m["budget"], m["fp"], 100 * rev))
    if agg:
        print("-" * 78)
        print("%-18s %-24s %6.1f%% %6.1f%% %7d" % (
            "平均", "", 100 * statistics.mean(a["recall"] for a in agg),
            100 * statistics.mean(a["budget"] for a in agg),
            statistics.mean(a["fp"] for a in agg)))
        print("\n※ budget が予算(%.0f%%)を大きく超える動画があれば、その動画では"
              % (100 * budget))
        print("  「未見の素材で手動量が爆発する」= A→B 悪化の再発。定数を採用しないこと。")
        print("※ **レビュー率が低い動画の数字は信用しない**。budget の大半が")
        print("  「人が見ていないフレームへの指摘」で、それが無駄かどうかは不明。")
        print("  （2026-07-17: この取り違えで幻の「無駄42%削減」を報告した）")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True,
                    help="extract_failure_dataset.py の出力 CSV")
    ap.add_argument("--budget", type=float, default=0.10,
                    help="固定手動バジェット: 立ててよいフレームの割合 (既定 0.10)")
    ap.add_argument("--no-cv", action="store_true", help="クロス検証を省略")
    args = ap.parse_args(argv)

    rows = load(args.dataset)
    if not rows:
        print("データがありません", file=sys.stderr)
        return 1

    have_dev = sum(1 for r in rows if r["dev"] is not None)
    have_lmk = sum(1 for r in rows if r["lmk_conf"] is not None
                   and r["lmk_conf"] < CONF_FAILED)
    n_unk = sum(1 for r in rows if r["label"] == -1)
    print("=" * 78)
    print("データセット: %d 行 / 正例 %d / 負例 %d / **未レビュー %d** / 動画 %d本" % (
        len(rows), sum(1 for r in rows if r["label"] == 1),
        sum(1 for r in rows if r["label"] == 0), n_unk,
        len({r["project"] for r in rows})))
    if n_unk:
        print("  ※ 未レビュー(label=-1)は「人が見ていない＝正誤不明」。"
              "recall/誤検出の計算には使わないが、")
        print("     レビュー予算(budget)の分母と、トラック中央値の計算には含める")
    print("  lmk_conf 実測あり: %d (%.0f%%)   dev あり: %d (%.0f%%)" % (
        have_lmk, 100 * have_lmk / len(rows), have_dev,
        100 * have_dev / len(rows)))
    if have_lmk / len(rows) < 0.5:
        print("  ⚠ lmk_conf の実測率が低い。3DDFA が死んだ状態で測っていないか")
        print("     確認すること（GET /api/health / stats.lmk3d_loaded）")

    # 現行値のベースライン
    base = evaluate(rows, predict(rows, CUR_K, CUR_FLOOR, CUR_CEIL,
                                  CUR_DEVIATION))
    print("\n現行値 (K=%.1f, DEV=%.2f, FLOOR=%.3f, CEIL=%.3f):" % (
        CUR_K, CUR_DEVIATION, CUR_FLOOR, CUR_CEIL))
    if base:
        print("  失敗リコール %.1f%% (拾えた %d/%d)  レビュー予算 %.1f%%" % (
            100 * base["recall"], base["hit"], base["n_pos"],
            100 * base["budget"]))
        print("  確実な無駄 %d/%d (%.1f%%)   正誤不明を立てた %d/%d (%.1f%%)" % (
            base["fp"], base["n_neg"], 100 * base["fpr"],
            base["flagged_unknown"], base["n_unk"],
            100 * base["unknown_rate"]))

    # グリッドは現行値(K=2.0/DEV=0.5)の**両側**に十分広げる。初回の掃引で
    # K=1.0・DEV=0.15（当時の下限）に張り付いた＝最適が枠外にあったため。
    # 端に張り付いたら枠を広げること（下の warning が検出する）。
    # グリッドは現行値(K=2.0/DEV=0.5)の**両側**に十分広げる。初回の掃引で
    # K=1.0・DEV=0.15（当時の下限）に張り付いた＝最適が枠外にあったため。
    # 端に張り付いたら枠を広げること（下の warning が検出する）。
    #
    # ⚠ FLOOR は細かく刻む。この素材は lmk_conf の median が 0.005〜0.012 で、
    # th=max(FLOOR, K×median) の FLOOR 側にクランプされるトラックが多い
    # ＝ **K を動かしても th が動かない**（K=0.5 と 0.7 が同じ結果になる）。
    # 実質の効き目は FLOOR にある。
    grid = {
        "k": [0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0],
        "deviation": [0.02, 0.05, 0.08, 0.12, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0],
        "floor": [0.002, 0.004, 0.006, 0.008, 0.010, 0.012, 0.016, 0.020],
        "ceil": [0.030, 0.060, 0.120, 0.300],
    }
    print("\n全データで予算 %.0f%% 以内の上位候補:" % (100 * args.budget))
    print("%-30s %8s %8s %8s %8s" % ("(K, DEV, FLOOR, CEIL)", "recall",
                                     "budget", "確実な無駄", "不明立て"))
    ranked = sweep(rows, grid, args.budget)
    for params, m in ranked[:8]:
        print("%-30s %7.1f%% %7.1f%% %8d %8d" % (
            "(%.2f, %.2f, %.3f, %.3f)" % params,
            100 * m["recall"], 100 * m["budget"], m["fp"],
            m["flagged_unknown"]))

    # 最適がグリッドの端に来たら、最適は枠の外にある = 掃引範囲が足りない
    if ranked:
        best = ranked[0][0]
        rails = []
        for name, val, axis in (("K", best[0], grid["k"]),
                                ("DEVIATION", best[1], grid["deviation"]),
                                ("FLOOR", best[2], grid["floor"]),
                                ("CEIL", best[3], grid["ceil"])):
            if val == min(axis):
                rails.append("%s が下限 %g に張り付き" % (name, val))
            elif val == max(axis):
                rails.append("%s が上限 %g に張り付き" % (name, val))
        if rails:
            print("\n  ⚠ 最適がグリッド端: %s" % " / ".join(rails))
            print("    → 最適は掃引範囲の外にある。grid を広げて測り直すこと。")

    if not args.no_cv:
        cross_validate(rows, grid, args.budget)

    print("\n" + "=" * 78)
    print("⚠ 採用前に必ず: 候補の定数を tracker.py に入れて実際に再測定し、")
    print("  オフライン予測と実ゲートの棄却数が一致するか確認すること。")
    print("  本ツールの dev は出荷時 th で計算された保存値であり、K を大きく")
    print("  動かすと実ゲートの dev とズレる（docstring の「近似について」参照）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
