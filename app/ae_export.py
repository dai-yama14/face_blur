"""
ae_export.py - After Effects 取り込み用 JSON 書き出し

~/projects/movieedit/mouth_mask_importer2.jsx が読む「Ver.0.17 (v017)」形式へ
プロジェクトを変換する。jsx 側は改変不要。

出力形式:
{
  "version": "1.0",
  "project":  { video_path, video_filename, frame_count, fps, width, height,
                created_at, modified_at },
  "persons":  [ { id, label, color:[r,g,b], mask_type } ],
  "keyframes":{ "<person_id>": { "<frame>": { shape:"ellipse",
                cx,cy,rx,ry,angle, visible, status } } }
}
座標は動画のピクセル座標（フル解像度）。AE のコンプ/レイヤーを同解像度にすれば 1:1。
"""
from __future__ import annotations

from pathlib import Path

# movieedit2 の keyframe.src → jsx が認識する status への対応
# （jsx の allowStatus は detected/sam2/interpolated/propagated/fallback/manual/keyframe）
_SRC_TO_STATUS = {
    "manual": "manual",
    "auto": "propagated",
}


def _hex_to_rgb(color: str | None) -> list[int]:
    """"#ff5555" → [255, 85, 85]。不正値は赤にフォールバック。"""
    if isinstance(color, str) and color.startswith("#") and len(color) == 7:
        try:
            return [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
        except ValueError:
            pass
    return [255, 80, 80]


def _ellipse_kf(kf: dict) -> dict | None:
    """保存済みキーフレーム → v017 の1フレーム分。楕円でなければ None。"""
    for key in ("cx", "cy", "rx", "ry"):
        if kf.get(key) is None:
            return None
    status = _SRC_TO_STATUS.get(kf.get("src"), "propagated")
    return {
        "shape": "ellipse",
        "cx": round(float(kf["cx"]), 3),
        "cy": round(float(kf["cy"]), 3),
        "rx": round(float(kf["rx"]), 3),
        "ry": round(float(kf["ry"]), 3),
        "angle": round(float(kf.get("angle", 0.0)), 4),
        "visible": bool(kf.get("visible", True)),
        "status": status,
    }


# AE 側は 1 フレーム = 1 マスクキーフレーム（setValueAtTime）で取り込むため、
# 数万フレームをそのまま渡すと maskPath への setValueAtTime が数万回走り、
# AE がメモリ膨張して 10 分級のハング→クラッシュに至る。書き出し側で
# 「直線補間で許容誤差内に収まる中間 KF」を間引いて渡数を減らす（RDP 風）。
# visible の切替フレームは境界として必ず残す。
# 0.5px では clip1(34922F) が 14265KF までしか減らず AE には依然多すぎたため、
# クラッシュ圏を明確に外す 2.0px を既定に。ブラーマスクはフェザー＋拡張が
# 乗るので 2px 級のずれは実質不可視。精度重視なら ?tol_px= で下げられる。
DEFAULT_TOL_PX = 2.0   # cx,cy,rx,ry の許容誤差（px）
DEFAULT_TOL_ANG = 1.0  # angle の許容誤差（度）


def _kf_dev(a: dict, b: dict, m: dict, t: float, tol_px: float, tol_ang: float) -> float:
    """フレーム a→b を直線補間した時刻 t の値と、実測 m とのずれを
    「許容量に対する比」で返す。1.0 超なら許容誤差オーバー。"""
    dev = 0.0
    for key in ("cx", "cy", "rx", "ry"):
        lin = a[key] + (b[key] - a[key]) * t
        dev = max(dev, abs(m[key] - lin) / tol_px)
    lin_ang = a["angle"] + (b["angle"] - a["angle"]) * t
    dev = max(dev, abs(m["angle"] - lin_ang) / tol_ang)
    return dev


def _thin_run(frames: list[int], kfs: dict, tol_px: float, tol_ang: float) -> list[int]:
    """同一 visible の連続ブロック（frames は昇順）を RDP で間引き、
    残すフレーム番号のリストを返す。両端は必ず残す。"""
    n = len(frames)
    if n <= 2:
        return list(frames)
    keep = [False] * n
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        a, b = kfs[frames[lo]], kfs[frames[hi]]
        span = frames[hi] - frames[lo]
        worst_dev, worst_i = 0.0, -1
        for i in range(lo + 1, hi):
            t = (frames[i] - frames[lo]) / span
            d = _kf_dev(a, b, kfs[frames[i]], t, tol_px, tol_ang)
            if d > worst_dev:
                worst_dev, worst_i = d, i
        if worst_dev > 1.0:
            keep[worst_i] = True
            stack.append((lo, worst_i))
            stack.append((worst_i, hi))
    return [frames[i] for i in range(n) if keep[i]]


def _thin_keyframes(conv: dict, tol_px: float, tol_ang: float) -> dict:
    """1 人物ぶんの v017 keyframe dict を間引く。visible の切替を境界に
    ブロック分割し、各ブロックを RDP 簡約する。"""
    if tol_px <= 0 and tol_ang <= 0:
        return conv
    frames = sorted(int(f) for f in conv.keys())
    if len(frames) <= 2:
        return conv
    # visible が変わるフレームでブロックを切る
    blocks: list[list[int]] = []
    cur: list[int] = []
    prev_vis = None
    for fn in frames:
        vis = conv[str(fn)].get("visible", True)
        if prev_vis is not None and vis != prev_vis:
            blocks.append(cur)
            cur = []
        cur.append(fn)
        prev_vis = vis
    if cur:
        blocks.append(cur)
    kept: dict = {}
    for blk in blocks:
        for fn in _thin_run(blk, {f: conv[str(f)] for f in blk}, tol_px, tol_ang):
            kept[str(fn)] = conv[str(fn)]
    return kept


def build_ae_json(
    proj: dict,
    tol_px: float = DEFAULT_TOL_PX,
    tol_ang: float = DEFAULT_TOL_ANG,
) -> dict:
    """movieedit2 プロジェクト dict を AE 取り込み用 v017 dict へ変換する。
    tol_px/tol_ang を 0 にすると間引きを無効化（全 KF 出力）。"""
    video = proj.get("video", {})
    path = video.get("path", "")
    out: dict = {
        "version": "1.0",
        "project": {
            "video_path": path,
            "video_filename": Path(path).name if path else "",
            "frame_count": int(video.get("frame_count", 0)),
            "fps": float(video.get("fps", 30.0)),
            "width": int(video.get("width", 1920)),
            "height": int(video.get("height", 1080)),
            "created_at": proj.get("created", ""),
            "modified_at": proj.get("updated", ""),
        },
        "persons": [],
        "keyframes": {},
    }

    keyframes_all = proj.get("keyframes", {})
    for p in proj.get("persons", []):
        pid = p.get("id")
        if not pid:
            continue
        out["persons"].append({
            "id": pid,
            "label": p.get("label", pid),
            "color": _hex_to_rgb(p.get("color")),
            "mask_type": p.get("region", "face"),
        })
        frames = keyframes_all.get(pid, {})
        conv: dict = {}
        for fk, kf in frames.items():
            if not isinstance(kf, dict):
                continue
            e = _ellipse_kf(kf)
            if e is not None:
                conv[str(fk)] = e
        out["keyframes"][pid] = _thin_keyframes(conv, tol_px, tol_ang)

    return out
