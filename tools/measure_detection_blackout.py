#!/usr/bin/env python3
"""顔位置の「根拠が無い」フレームを実測する。

問い（2026-07-23）:
  体のパーツ検出（骨格・ポーズ）を導入する価値があるか判断するため、
  1. 顔検出が全滅している（＝顔位置の根拠が無い）フレームは全体の何%か
  2. そこに手動修正がどれだけ集中しているか（＝人手時間を食っているか）
  を測る。

観測の根拠:
  tracker.convert_ident_to_mouth() は Pass1(局所)→Pass2(全画面)→Pass3(頭部近傍)
  →Pass4(拡大回転救済) の順に顔を探し、**成功したフレームだけ** refined に入れる。
  track_features は `for fi in refined:` で作られる（tracker.py:1325）。
  信頼度ゲートは feats を作った**後**に走り、棄却フレームは gated=True で残る
  （tracker.py:1426）。したがって:

    BLACK : visible キーフレームがあるのに track_features に行が無い
            → 4パス全滅。顔位置の根拠がゼロ
    GATED : 行はあるが gated=True
            → 顔は見つかったがフィットが破綻して捨てられた

  **どちらも最終的には頭部楕円基準の相対補間で置かれる**（＝位置の根拠が無い）。
  ただし体キーポイントで救えるのは BLACK だけ。GATED は顔が見つかっている
  ので、効くのはランドマークモデルの強化（large-pose-landmarks-3ddfa.md）。
  この2つは必ず分けて数えること。

「測っていない」を「成功」にも「失敗」にも数えない:
  UNMEAS: track_features that 人物が無い / 空。region="face" のプロジェクトは
          口元測定パスを通らないのでここに入る。集計から除外する。

集中度のベースラインは「レビュー済み区間」= [最初の手動, 最後の手動] の内側で取る。
人が見ていない区間を分母に入れると、そもそも手動が発生しえないフレームで
ベースラインが薄まり、集中度が過大に出る（extract_failure_dataset.py と同じ約束）。

使い方:
  python tools/measure_detection_blackout.py
  python tools/measure_detection_blackout.py --projects projects --event-gap 30
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys

HAS, BLACK, GATED = "has", "black", "gated"
CATS = (HAS, BLACK, GATED)


def _runs(sorted_frames: list[int], gap: int = 1) -> list[tuple[int, int]]:
    """連続（間隔 <= gap）なフレーム群を [start, end] の塊にまとめる。"""
    out: list[tuple[int, int]] = []
    for f in sorted_frames:
        if out and f - out[-1][1] <= gap:
            out[-1] = (out[-1][0], f)
        else:
            out.append((f, f))
    return out


def analyze_person(kfs: dict, tf: dict, event_gap: int) -> dict:
    manual = sorted(int(f) for f, k in kfs.items()
                    if (k or {}).get("src") == "manual")
    rev_lo, rev_hi = (manual[0], manual[-1]) if manual else (None, None)

    c = dict.fromkeys(CATS, 0)      # visible 全体
    rc = dict.fromkeys(CATS, 0)     # レビュー済み区間内
    mc = dict.fromkeys(CATS, 0)     # 手動修正フレーム
    invisible = 0
    black: list[int] = []
    nogrounds: list[int] = []       # BLACK + GATED

    for fstr, kf in kfs.items():
        if not (kf or {}).get("visible", True):
            invisible += 1          # 画面外退避。顔が無いのが正しいので除外
            continue
        f = int(fstr)
        row = tf.get(fstr)
        if row is None:
            cat = BLACK
            black.append(f)
        elif row.get("gated"):
            cat = GATED
        else:
            cat = HAS
        if cat != HAS:
            nogrounds.append(f)
        c[cat] += 1
        if rev_lo is not None and rev_lo <= f <= rev_hi:
            rc[cat] += 1
        if (kf or {}).get("src") == "manual":
            mc[cat] += 1

    black.sort()
    nogrounds.sort()
    ng_runs = _runs(nogrounds)
    ng_lens = [b - a + 1 for a, b in ng_runs]

    # 修正イベント（手動フレームの塊）と、根拠なしフレームを含むイベント
    ngset = set(nogrounds)
    bset = set(black)
    events = _runs(manual, event_gap)
    ev_ng = sum(1 for a, b in events
                if any(f in ngset for f in range(a, b + 1)))
    ev_black = sum(1 for a, b in events
                   if any(f in bset for f in range(a, b + 1)))

    return {
        "all": c, "reviewed": rc, "manual": mc,
        "visible": sum(c.values()), "invisible": invisible,
        "n_manual": len(manual),
        "ng_runs": len(ng_runs),
        "ng_len_max": max(ng_lens) if ng_lens else 0,
        "ng_len_med": statistics.median(ng_lens) if ng_lens else 0,
        "events": len(events), "events_ng": ev_ng, "events_black": ev_black,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects", default="projects")
    ap.add_argument("--event-gap", type=int, default=30,
                    help="この間隔以内の手動フレームを1修正イベントとみなす")
    args = ap.parse_args(argv)

    files = sorted(glob.glob(f"{args.projects}/*.mvproj.json"))
    if not files:
        print(f"プロジェクトが見つかりません: {args.projects}", file=sys.stderr)
        return 1

    tot = dict.fromkeys(CATS, 0)
    rtot = dict.fromkeys(CATS, 0)
    mtot = dict.fromkeys(CATS, 0)
    ev_tot = ev_ng = ev_black = 0
    unmeas = 0
    maxlens: list[int] = []
    rows = []
    skipped = []

    for fp in files:
        p = json.loads(open(fp, encoding="utf-8").read())
        pid = p.get("id", "?")
        region = (p.get("targets") or {}).get("region", "?")
        tf_all = p.get("track_features") or {}
        for person_id, kfs in (p.get("keyframes") or {}).items():
            tf = tf_all.get(person_id) or {}
            if not tf:
                # 未測定。成功にも失敗にも数えない
                n = sum(1 for k in kfs.values()
                        if (k or {}).get("visible", True))
                unmeas += n
                skipped.append((pid, region, n))
                continue
            r = analyze_person(kfs, tf, args.event_gap)
            for k in CATS:
                tot[k] += r["all"][k]
                rtot[k] += r["reviewed"][k]
                mtot[k] += r["manual"][k]
            ev_tot += r["events"]
            ev_ng += r["events_ng"]
            ev_black += r["events_black"]
            if r["ng_len_max"]:
                maxlens.append(r["ng_len_max"])
            rows.append((pid, region, r))

    def ng(d):
        return d[BLACK] + d[GATED]

    print("=" * 104)
    print("顔位置の根拠が無いフレームの実測  (visible キーフレームのみ)")
    print("  BLACK=4パス全滅（体キーポイントで救える候補） / "
          "GATED=顔は取れたがフィット破綻（ランドマーク強化の領分）")
    print("=" * 104)
    print(f"{'project':22s} {'測定済':>7s} {'全滅':>6s} {'ゲート':>6s} "
          f"{'根拠なし':>8s} {'率':>7s} {'塊':>4s} {'最長':>5s} | "
          f"{'手動':>5s} {'手動中':>6s} {'集中度':>7s}")
    print("-" * 104)
    for pid, region, r in rows:
        meas = sum(r["all"].values())
        rmeas = sum(r["reviewed"].values())
        base = ng(r["reviewed"]) / rmeas if rmeas else 0
        nm = sum(r["manual"].values())
        mr = ng(r["manual"]) / nm if nm else None
        lift = (f"{mr / base:5.2f}x" if base > 0 and mr is not None
                else "    -")
        print(f"{pid[:22]:22s} {meas:7d} {r['all'][BLACK]:6d} "
              f"{r['all'][GATED]:6d} {ng(r['all']):8d} "
              f"{100 * ng(r['all']) / meas:6.1f}% {r['ng_runs']:4d} "
              f"{r['ng_len_max']:5d} | {r['n_manual']:5d} "
              f"{ng(r['manual']):6d} {lift:>7s}")
    print("-" * 104)

    meas = sum(tot.values())
    rmeas = sum(rtot.values())
    nm = sum(mtot.values())
    print(f"測定済 {meas}  /  未測定 {unmeas}（集計から除外）")
    print()
    print(f"[Q1] 全滅(BLACK)     : {tot[BLACK]:6d} = {100*tot[BLACK]/meas:5.1f}%")
    print(f"     ゲート棄却(GATED): {tot[GATED]:6d} = {100*tot[GATED]/meas:5.1f}%")
    print(f"     → 顔位置の根拠なし: {ng(tot)} / {meas} = "
          f"{100*ng(tot)/meas:.1f}%")
    print(f"     レビュー済み区間内: {ng(rtot)} / {rmeas} = "
          f"{100*ng(rtot)/rmeas:.1f}%   ← 集中度のベースライン")
    if maxlens:
        print(f"     連続長: 動画ごとの最長の中央値 "
              f"{statistics.median(maxlens):.0f}f / 最大 {max(maxlens)}f")
    print()
    print(f"[Q2] 手動修正 {nm} フレーム")
    print(f"     うち根拠なし: {ng(mtot)} = {100*ng(mtot)/nm:.1f}%"
          f"   (全滅 {mtot[BLACK]} = {100*mtot[BLACK]/nm:.1f}%, "
          f"ゲート {mtot[GATED]} = {100*mtot[GATED]/nm:.1f}%)")
    if rmeas and nm:
        base = ng(rtot) / rmeas
        print(f"     集中度(リフト): {(ng(mtot)/nm)/base:.2f}x")
    print()
    print(f"[Q3] 修正イベント(間隔{args.event_gap}f以内をまとめた塊): {ev_tot}"
          "   ← 人手時間に近い数え方")
    print(f"     根拠なしを含む: {ev_ng} = {100*ev_ng/ev_tot:.1f}%"
          f"   / 全滅を含む: {ev_black} = {100*ev_black/ev_tot:.1f}%")
    if skipped:
        print()
        print("[未測定] track_features が無い/空（region=face は口元パスを通らない）:")
        for pid, region, n in skipped:
            print(f"  {pid:24s} region={region:6s} visible={n}")
    print("=" * 104)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
