"""
render.py - キーフレーム補間とブラー合成

セーブファイルのキーフレームから任意フレームの楕円を補間し、
ガウシアンブラー / モザイクをフェザー付きで合成する。
プレビューと mp4 書き出しの両方がこのモジュールを使う（見た目が一致する）。
"""
from __future__ import annotations

import cv2
import numpy as np

# キーフレームが途切れた後、この フレーム数までは直前の位置を保持してブラーし続ける
# （漏れ＝致命的なので、消すよりブラーし続ける方向に倒す）
HOLD_FRAMES = 12

# 長い無検出区間で、前後キーフレームとも画面端付近（＝フレームアウト）なら、
# 画面内を補間せず画面外へ退避する（人物が画面外にいる想定）。
# 画面内で後ろ向き/横顔が続く長区間は端に来ないためこの条件では退避せず、
# 従来どおりブラーを継続する（漏れ＝致命的の安全側を維持）
PARK_MIN_GAP = 30       # これ以上の無検出フレーム数が画面外退避の対象
PARK_EDGE_FACTOR = 1.5  # 中心が最寄り画面端から rmax*この値 以内なら「端」扱い


def _lerp_angle(a: float, b: float, t: float) -> float:
    """角度を最短経路で補間する（-180〜180 に正規化）。"""
    diff = (b - a + 180.0) % 360.0 - 180.0
    return a + diff * t


def _near_edge(kf: dict, vw: int, vh: int) -> bool:
    """楕円中心が最寄りの画面端に近い（フレームアウトしかけ）か。"""
    rmax = max(kf["rx"], kf["ry"])
    d = min(kf["cx"], vw - kf["cx"], kf["cy"], vh - kf["cy"])
    return d < rmax * PARK_EDGE_FACTOR


def _parked_ellipse(ref: dict, vw: int, vh: int) -> dict:
    """
    検出できない区間用: 直近の楕円を「最寄りの画面端のすぐ外」へ待機させる。
    ブラーは画面に掛からないが、ズームアウトすれば掴んで戻せる位置に置く。
    """
    rmax = max(ref["rx"], ref["ry"])
    margin = rmax * 1.3 + 24  # フェザー込みで完全に画面外に出る距離
    dists = {"left": ref["cx"], "right": vw - ref["cx"],
             "top": ref["cy"], "bottom": vh - ref["cy"]}
    edge = min(dists, key=dists.get)
    cx, cy = ref["cx"], ref["cy"]
    if edge == "left":
        cx = -margin
    elif edge == "right":
        cx = vw + margin
    elif edge == "top":
        cy = -margin
    else:
        cy = vh + margin
    out = dict(ref)
    out.update(cx=cx, cy=cy, visible=True, parked=True)
    return out


def interp_ellipse(keyframes: dict, frame: int,
                   vw: int | None = None, vh: int | None = None) -> dict | None:
    """
    person の keyframes（{"フレーム番号(str)": kf}）から frame 時点の楕円を補間する。
    戻り値: {cx, cy, rx, ry, angle, visible} または None（非表示）。
    vw/vh（動画サイズ）を渡すと、検出できない区間では None の代わりに
    画面外パーキングの楕円（parked=True）を返す。
    """
    if not keyframes:
        return None
    exact = keyframes.get(str(frame))
    if exact is not None:
        # 不在マーク: 編集用（vw/vh あり）は画面外パーキング、
        # ぼかし用（vw/vh なし）は None＝ブラーを掛けない
        if exact.get("absent"):
            return _parked_ellipse(exact, vw, vh) if (vw and vh) else None
        return exact if exact.get("visible", True) else None

    frames = sorted(int(k) for k in keyframes.keys())
    prev_f = None
    next_f = None
    for f in frames:
        if f < frame:
            prev_f = f
        elif f > frame:
            next_f = f
            break

    prev_kf = keyframes.get(str(prev_f)) if prev_f is not None else None
    next_kf = keyframes.get(str(next_f)) if next_f is not None else None

    prev_ok = prev_kf is not None and prev_kf.get("visible", True)
    next_ok = next_kf is not None and next_kf.get("visible", True)

    if prev_ok and next_ok:
        gap = next_f - prev_f
        # 長い無検出区間 かつ 前後ともフレーム端 → 画面外へ退避（両端は HOLD ぶん保持）
        if (vw and vh and gap >= PARK_MIN_GAP
                and frame - prev_f > HOLD_FRAMES and next_f - frame > HOLD_FRAMES
                and _near_edge(prev_kf, vw, vh) and _near_edge(next_kf, vw, vh)):
            ref = prev_kf if (frame - prev_f) <= (next_f - frame) else next_kf
            return _parked_ellipse(ref, vw, vh)
        t = (frame - prev_f) / (next_f - prev_f)
        return {
            "cx": prev_kf["cx"] + (next_kf["cx"] - prev_kf["cx"]) * t,
            "cy": prev_kf["cy"] + (next_kf["cy"] - prev_kf["cy"]) * t,
            "rx": prev_kf["rx"] + (next_kf["rx"] - prev_kf["rx"]) * t,
            "ry": prev_kf["ry"] + (next_kf["ry"] - prev_kf["ry"]) * t,
            "angle": _lerp_angle(prev_kf.get("angle", 0.0),
                                 next_kf.get("angle", 0.0), t),
            "visible": True,
        }
    if prev_ok and frame - prev_f <= HOLD_FRAMES:
        return dict(prev_kf, visible=True)
    if next_ok and next_f - frame <= HOLD_FRAMES:
        return dict(next_kf, visible=True)
    # 検出できない区間: 画面外パーキング（編集用。ブラーは掛からない）
    if vw and vh:
        ref = prev_kf if prev_ok else (next_kf if next_ok else None)
        if ref is not None:
            return _parked_ellipse(ref, vw, vh)
    return None


def _interp_scalar(kfs: dict, frame: int) -> float | None:
    """スカラー値キーフレーム（{"フレーム": 値}）の線形補間。端はホールド。"""
    if not kfs:
        return None
    exact = kfs.get(str(frame))
    if exact is not None:
        return float(exact)
    frames = sorted(int(k) for k in kfs.keys())
    prev_f = next_f = None
    for f in frames:
        if f < frame:
            prev_f = f
        else:
            next_f = f
            break
    if prev_f is not None and next_f is not None:
        t = (frame - prev_f) / (next_f - prev_f)
        a, b = float(kfs[str(prev_f)]), float(kfs[str(next_f)])
        return a + (b - a) * t
    if prev_f is not None:
        return float(kfs[str(prev_f)])
    return float(kfs[str(next_f)])


def strength_at(person: dict, frame: int) -> float:
    """ブラー濃度: 濃さキーフレームがあれば補間値、なければ固定値。"""
    v = _interp_scalar(person.get("strength_kfs") or {}, frame)
    if v is not None:
        return v
    return float(person.get("blur", {}).get("strength", 30))


def _mosaic(roi: np.ndarray, strength: int) -> np.ndarray:
    h, w = roi.shape[:2]
    block = max(2, int(min(h, w) / max(2, 60 - strength * 0.5)))
    small = cv2.resize(roi, (max(1, w // block), max(1, h // block)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def apply_blurs(frame: np.ndarray, persons: list[dict],
                keyframes_all: dict, frame_index: int) -> np.ndarray:
    """フレームに全人物のブラーを合成して返す（元の frame は変更しない）。"""
    out = frame
    h, w = frame.shape[:2]

    for person in persons:
        if not person.get("enabled", True):
            continue
        ell = interp_ellipse(keyframes_all.get(person["id"], {}), frame_index)
        if ell is None or ell.get("parked"):
            continue

        cx, cy = ell["cx"], ell["cy"]
        rx, ry = max(2.0, ell["rx"]), max(2.0, ell["ry"])
        angle = float(ell.get("angle", 0.0))
        blur_cfg = person.get("blur", {})
        strength = int(round(strength_at(person, frame_index)))
        if strength <= 0:
            continue  # 濃さ0 = このフレームはブラーなし
        feather = float(blur_cfg.get("feather", 0.25))
        btype = blur_cfg.get("type", "gaussian")

        # フェザー分も含めた ROI を切り出して処理（全画面処理を回避）
        # 回転を考慮し半径は max(rx, ry) で安全側にとる
        feather_px = feather * min(rx, ry)
        margin = int(feather_px * 2 + 4)
        rmax = max(rx, ry)
        x1 = max(0, int(cx - rmax) - margin)
        y1 = max(0, int(cy - rmax) - margin)
        x2 = min(w, int(cx + rmax) + margin)
        y2 = min(h, int(cy + rmax) + margin)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue

        if out is frame:
            out = frame.copy()
        roi = out[y1:y2, x1:x2]

        if btype == "mosaic":
            blurred = _mosaic(roi, strength)
        else:
            # strength(1-100) をカーネルサイズへ。顔サイズにも比例させる
            k = int(max(3, (strength / 100.0) * min(rx, ry) * 1.6))
            k = k * 2 + 1
            blurred = cv2.GaussianBlur(roi, (k, k), 0)

        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, (int(cx) - x1, int(cy) - y1), (int(rx), int(ry)),
                    angle, 0, 360, 255, -1)
        if feather_px >= 1:
            fk = int(feather_px) * 2 + 1
            mask = cv2.GaussianBlur(mask, (fk, fk), 0)

        m = (mask.astype(np.float32) / 255.0)[:, :, None]
        roi[:] = (roi.astype(np.float32) * (1 - m)
                  + blurred.astype(np.float32) * m).astype(np.uint8)

    return out
