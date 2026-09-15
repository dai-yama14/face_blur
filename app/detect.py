"""
detect.py - 自動顔検出（バックグラウンドジョブ）

InsightFace(SCRFD) で全フレームを検出し、IoUトラッキングで人物トラックに束ね、
楕円キーフレームとしてプロジェクトに書き込む。

v2構想（ArcFaceアンカー + SAM 2 伝播 + Claude自動クリック）の足場となる一次実装。
検出結果はエディターで1クリック修正できる前提の「たたき台」を作る役割。
"""
from __future__ import annotations

import threading
import traceback

import cv2
import numpy as np

from . import landmarks3d
from . import project as prj
from .video import manager as video_manager

# 楕円パディング（顔bboxに対する余裕。髪・輪郭までカバー）
PAD_X = 1.35
PAD_Y = 1.55
# 口元楕円パラメータ。基準は頑健スケール s =（目間隔 + 目〜口距離）/ 2
# （tracker.py の Mocha式 _face_frame と同じ。横顔で口角間距離が射影上
#   崩壊し、鼻口距離が鼻の横突き出しで膨張する問題を避ける）。
# 正面実測: 口角間距離 mw ≈ 0.78s, 鼻口距離 d ≈ 0.51s（ht_clip1 で校正）
MOUTH_W_RATIO = 2.0      # 横幅 = 口角間距離 x この値（正面で口幅に追従）
MOUTH_RX_MIN_S = 0.70    # 横半径の下限 = s x この値（横顔での崩壊防止）
MOUTH_ASPECT = 1.1       # 縦横比（フォールバック用。小鼻〜顎カバーの縦長）
# 縦スパン（口=0, s単位）: 上端 = SHIFT - RY = -0.64s（鼻先の少し上）,
# 下端 = SHIFT + RY = +1.08s（顎下）。d単位では -1.27d / +2.13d に相当
# （2026-07-03 オーナー要望「小鼻〜顎」の見え方を維持）
MOUTH_DOWN_SHIFT_S = 0.22  # 中心を口から顎側へずらす量（s比）
MOUTH_RY_S = 0.86          # 縦半径 = s x この値
# 106点ランドマーク（buffalo_l 2d106）の領域インデックス。
# 唇+小鼻+下顎輪郭の実点に楕円を当てるため（横顔で5点式が崩れる問題の対策）
LMK_MOUTH = list(range(52, 72))       # 唇（外輪郭+内輪郭 20点）
LMK_NOSE_BOTTOM = list(range(76, 87))  # 小鼻・鼻下
LMK_CONTOUR = list(range(0, 33))      # 顔輪郭（下部が下顎〜顎先）
MOUTH_LMK_PAD = 1.10       # 実点群に対する楕円マージン
MOUTH_LMK_PCT = 3          # 外れ値除去のパーセンタイル（横顔の奥側推定点対策）
# 3DDFA_V2 の iBUG 68点レイアウトの領域インデックス（大ポーズ対応の本命）
LMK68_JAW = list(range(0, 17))        # 顎輪郭（0=右端, 8=顎先, 16=左端）
LMK68_CHIN = [8]                      # 顎先（点8のみ）。7/9以降は横顔でエラ側へ
                                      # 回り込み楕円が首まで肥大するため含めない
LMK68_NOSE_BOTTOM = list(range(31, 36))  # 小鼻・鼻下
LMK68_MOUTH = list(range(48, 68))     # 唇（外輪郭+内輪郭）
LMK68_REYE = list(range(36, 42))
LMK68_LEYE = list(range(42, 48))
LMK68_BROWS = list(range(17, 27))     # 眉（顔全体楕円の上端の目安）
# 口元(68点): 縦は「小鼻〜顎先」を確実に含める。3DMM由来で外れ値が出にくいため
# パーセンタイル除去はせず min/max で端点を取り、下方向にはクランプしない。
MOUTH68_PAD_V = 1.08       # 縦マージン（小鼻・顎先を楕円内に収める。包含で最終保証）
MOUTH68_PAD_H = 1.06       # 横マージン（頬まで広げた幅に対する余白）
MOUTH68_CHEEK_FRAC = 0.5   # 口幅→頬(顔輪郭)の何割まで横に含めるか（0.5=頬の半分）。
    # ただし **横顔ほど頬を含めない**（profile で 0 まで線形に減衰させる）。
    # 横顔では顔輪郭が奥（耳側）へ回り込むため、頬を含めると楕円の中心が頬の上へ
    # 引っ張られ、マスクが口元から後ろへずれていく（オーナー報告 hy_clip2 f4208〜）。
    # 実測（hy_clip1 の信頼できるフィット83件, 手動修正が正解）: IoU中央
    #   0.5固定(従来) 0.753 → 0.5*(1-0.5p) 0.760 → **0.5*(1-p) 0.770**
    #   → 0.25固定 0.732 / 0固定 0.542（正面では頬が要る。消してはいけない）
    # 検証: .company/engineering/docs/mouth-mask-accuracy-hy.md
MOUTH68_CONTAIN_MARGIN = 1.02  # 包含スケールの余白（縁ぎりぎり回避。小さめでタイトに）
# ⚠ 「上端を鼻先で頭打ちにするガード」は試したが **却下**（2026-07-11）。
#   包含スケールを上書きしてしまうため、目までの余裕は稼げても
#   **小鼻が楕円から漏れる**（hy_clip2 で 30% / hy_clip1 で 12% のフレーム）。
#   ぼかしツールで鼻が漏れるのは本末転倒。上端を下げたいときは下の
#   NOSE_BASE / NOSE_EXT を減らすこと（包含保証が効くので漏れない）。
MOUTH68_NOSE_EXT = 0.45    # 横顔で小鼻側へ足す余裕（縦スパン比。横顔度が最大の時の値）
MOUTH68_NOSE_BASE = 0.15   # 正面含め常に小鼻側(上)へ足す縦の余裕（縦スパン比）
    # 0.22/0.60 → 0.15/0.45 へ縮小（2026-07-11、オーナー報告「目までマスクされる」）。
    # 小鼻・顎先の包含は下の包含スケールが保証するので、この余裕を減らしても漏れない。
    # 実測（目までの余裕 = 1.0 が楕円の縁。大きいほど目から遠い）:
    #                     hy_clip2   hy_clip1   小鼻漏れ  顎漏れ  IoU(hy1)
    #   0.22/0.60(旧)       1.30       1.19        0%      0%     0.770
    #   0.18/0.50           1.36       1.27        0%      0%     0.756
    #   0.15/0.45(採用)     1.39       1.31        0%      0%     0.745
    #   0.12/0.35           1.40       1.34        0%      0%     0.724
    # IoU は手描き正解との一致度なので、目に被らない方を優先して 0.15/0.45 を採用。
    # ※ 実際にぼける範囲は render.py の feather（既定 0.25×半径）ぶん外側へ広がる。
    #   「目にかかる」体感にはそちらも効く。エディタの feather でも調整可能。
    # 検証: .company/engineering/docs/mouth-mask-accuracy-hy.md
MOUTH68_CHIN_EXT = 0.20    # 顎側(下)へ足す縦の余裕（縦スパン比）。
    # 従来は鼻側にだけ余裕を足しており、顎側は 3DDFA の顎先ランドマーク(点8)を
    # 内包するだけだった。しかし実際のマスクは顎先より少し外まで要る。
    # 実測（hy_clip1 手動修正103件を正解）: 顎先が楕円からはみ出す率
    #   0.00(従来) 83% → 0.10 45% → 0.20 17% → 0.30 11%
    # 一方で楕円が縦に伸びるぶん、手描き正解との IoU は下がる
    #   0.774 → 0.760 → 0.728 → 0.687
    # ぼかしツールとしては「覆い漏れ＝実害」「覆いすぎ＝見た目」で非対称なので、
    # IoU を少し犠牲にして 0.20 を採用（オーナー判断, 2026-07-11）。
    # 検証: .company/engineering/docs/mouth-mask-accuracy-hy.md
FACE_LMK_TOP_EXT = 1.05    # 顔全体: 眉→顎スパンに対し額・髪側へ上方拡張する比
FACE_LMK_W_PAD = 1.18      # 顔全体: 横方向マージン（輪郭の外側=髪・耳）
FACE_LMK_PCT = 2           # 顔全体の外れ値除去パーセンタイル
# これ未満のフレーム数しか続かないトラックはノイズとして捨てる
MIN_TRACK_LEN = 5
# トラック照合の IoU 閾値と、見失いを許容するフレーム数
IOU_THRESHOLD = 0.25
MAX_MISSES = 15

_apps: dict[bool, object] = {}
_app_lock = threading.Lock()


def get_face_app(with_rec: bool = False):
    """
    FaceAnalysis は初期化が重いのでプロセス内で使い回す。
    with_rec=True で ArcFace 認識（embedding）付きインスタンスを返す。
    """
    with _app_lock:
        if with_rec not in _apps:
            from insightface.app import FaceAnalysis
            # 検出用インスタンスには 106点ランドマークも載せる（口元楕円で
            # 横顔の顎・唇輪郭を直接取るため。認識用は identity 照合専用）
            modules = (["detection", "recognition"] if with_rec
                       else ["detection", "landmark_2d_106"])
            app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=modules,
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            _apps[with_rec] = app
        return _apps[with_rec]


def _get_face_app():
    return get_face_app(False)


# ── 検出リコールの救済 ───────────────────────────────────────────────
# 夜間・手持ちのブラー・あご上げ・大きな面内回転が重なると SCRFD が顔を
# 1つも返さない。hy_clip1 の手動修正フレームでは 32% がこの状態だった
# （マスクは補間で置かれるだけになり、位置の根拠が無くなる）。
#
# 位置が既知（頭部トラック）であることを利用し、局所を拡大・回転して検出を
# 試みる。実測で検出率 68% → 96%。
# ただし救済したフレームは 3DDFA が壊れやすい極端姿勢そのものなので、
# **必ず landmarks3d の信頼度ゲートとセットで使う**こと。単独で有効化すると
# 崩壊マスクを描く数が増えて逆効果になる（実測: 救済フレームの83%が崩壊）。
# 検証: .company/engineering/docs/mouth-mask-accuracy-hy.md
ROBUST_ANGLES = (0, 30, -30, 60, -60, 90, -90)
ROBUST_SCALE = 2.0      # 局所クロップの拡大率（ブラー・小顔に効く）
ROBUST_THRESH = 0.2     # 検出スコアのしきい値（既定 0.5 では取りこぼす）
ROBUST_GOOD_ENOUGH = 0.5  # このスコアで見つかれば残りの角度は試さない（早期終了）

_lowthr_app = None


def _get_face_app_lowthr():
    global _lowthr_app
    with _app_lock:
        if _lowthr_app is None:
            from insightface.app import FaceAnalysis
            app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection", "landmark_2d_106"],
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640),
                        det_thresh=ROBUST_THRESH)
            globals()["_lowthr_app"] = app
        return globals()["_lowthr_app"]


def detect_face_robust(frame, cx: float, cy: float, r: float) -> dict | None:
    """
    (cx, cy) 付近・半径 r 程度に顔があると分かっている前提で、局所を拡大＋回転
    して検出する。通常の検出が失敗したフレームの最後の救済手段。

    68点は**検出できた回転・拡大フレーム上でフィットしてから元座標へ戻す**。
    3DDFA も面内回転に弱いため、顔が起きた状態で推論するほうが精度が出る。

    返り値は _detect_face_near と同じ形（元フレーム座標）+ lmk_conf。
    """
    h, w = frame.shape[:2]
    pad = r * 2.5
    x1, y1 = max(0, int(cx - pad)), max(0, int(cy - pad))
    x2, y2 = min(w, int(cx + pad)), min(h, int(cy + pad))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    s = ROBUST_SCALE
    crop = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    ccx, ccy = (cx - x1) * s, (cy - y1) * s
    ch, cw = crop.shape[:2]
    app = _get_face_app_lowthr()

    best = None
    for ang in ROBUST_ANGLES:
        if ang == 0:
            img, M = crop, np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        else:
            M = cv2.getRotationMatrix2D((cw / 2, ch / 2), ang, 1.0)
            img = cv2.warpAffine(crop, M, (cw, ch), flags=cv2.INTER_LINEAR)
        # 期待位置を回転後座標へ
        px, py = M @ np.array([ccx, ccy, 1.0])
        f = _nearest_face(app.get(img), px, py, r * s * 4)
        if f is None:
            continue
        if best is None or float(f.det_score) > best[0]:
            best = (float(f.det_score), f, img, M)
        # 十分な確信度で見つかったら残りの角度は試さない（救済は重いので早期終了）
        if best[0] >= ROBUST_GOOD_ENOUGH:
            break
    if best is None:
        return None

    score, f, img, M = best
    Mi = cv2.invertAffineTransform(M)

    def to_orig(pts):
        """検出フレーム（回転・拡大後）の座標 → 元フレーム座標。"""
        p = np.asarray(pts, float).reshape(-1, 2)
        q = np.c_[p, np.ones(len(p))] @ Mi.T
        return np.c_[q[:, 0] / s + x1, q[:, 1] / s + y1]

    bb = f.bbox.astype(float)
    lmk68, lmk_conf = landmarks3d.get_landmarks_68_conf(img, bb.tolist())
    if lmk68 is not None:
        lmk68 = np.c_[to_orig(lmk68[:, :2]), lmk68[:, 2] / s]

    corners = to_orig([[bb[0], bb[1]], [bb[2], bb[1]],
                       [bb[2], bb[3]], [bb[0], bb[3]]])
    lm106 = getattr(f, "landmark_2d_106", None)
    return {
        "bbox": [float(corners[:, 0].min()), float(corners[:, 1].min()),
                 float(corners[:, 0].max()), float(corners[:, 1].max())],
        "kps": to_orig(f.kps.astype(float)).tolist(),
        "lmk": (to_orig(lm106.astype(float)[:, :2]).tolist()
                if lm106 is not None else None),
        "lmk68": lmk68,
        "lmk_conf": lmk_conf,
        "det_score": score,
    }


def _nearest_face(faces, px: float, py: float, limit: float):
    """(px, py) に最も近い顔。limit より遠ければ別人とみなし None。"""
    best, best_d = None, None
    for f in faces:
        if f.kps is None:
            continue
        b = f.bbox
        d = float(np.hypot((b[0] + b[2]) / 2 - px, (b[1] + b[3]) / 2 - py))
        if best_d is None or d < best_d:
            best, best_d = f, d
    if best is None or best_d > limit:
        return None
    return best


def landmarks_for_frame(frame) -> list[dict]:
    """指定フレームの全顔について、エディタが認識するランドマークを返す（表示・検証用）。

    エディタの口元/顔楕円は 68点(3DDFA_V2) → 106点 → 5点(SCRFD kps) の優先順位で
    ランドマークを実測して作られる（tracker.py _mouth_from_det と同じ序列）。ここでは
    取得できた全種類を返し、`source` に実際に採用される段を入れる。表示側で段ごとに
    色分けし、横顔でどの点が取れて/崩れているかを目視できるようにする。
    """
    faces = _get_face_app().get(frame)
    out: list[dict] = []
    for f in faces:
        bbox = [float(v) for v in f.bbox]
        kps = (np.asarray(f.kps, float).tolist()
               if getattr(f, "kps", None) is not None else None)
        lmk106 = getattr(f, "landmark_2d_106", None)
        lmk106 = (np.asarray(lmk106, float)[:, :2].tolist()
                  if lmk106 is not None else None)
        lmk68 = landmarks3d.get_landmarks_68(frame, bbox)
        lmk68 = (np.asarray(lmk68, float)[:, :2].tolist()
                 if lmk68 is not None else None)
        source = "lmk68" if lmk68 else ("lmk106" if lmk106 else "kps")
        out.append({"bbox": bbox, "kps": kps, "lmk106": lmk106,
                    "lmk68": lmk68, "source": source,
                    "det_score": float(getattr(f, "det_score", 0.0) or 0.0)})
    return out


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


class _Track:
    def __init__(self, tid: int, bbox, frame: int):
        self.id = tid
        self.bbox = bbox
        self.last_frame = frame
        self.misses = 0
        self.keyframes: dict[str, dict] = {}
        self.last_ell: dict | None = None   # 直近に採用した楕円（安定化の基準）


def _roll_angle(kps) -> float:
    """両目のランドマークから顔の傾き（roll, 度）を求める。"""
    if kps is None or len(kps) < 2:
        return 0.0
    ex, ey = kps[1][0] - kps[0][0], kps[1][1] - kps[0][1]
    return float(np.degrees(np.arctan2(ey, ex)))


def face_to_ellipse(bbox, kps=None) -> dict:
    """顔全体マスク: bbox ベース + 目の傾きで回転。"""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return {
        "cx": round((x1 + x2) / 2, 1),
        # 顔bboxは額が浅いので中心をやや上へ
        "cy": round((y1 + y2) / 2 - h * 0.05, 1),
        "rx": round(w / 2 * PAD_X, 1),
        "ry": round(h / 2 * PAD_Y, 1),
        "angle": round(_roll_angle(kps), 1),
        "visible": True,
        "src": "auto",
    }


def mouth_to_ellipse(bbox, kps=None) -> dict | None:
    """
    口元のみマスク: ランドマーク5点から頑健スケール s と顔の「下」方向を
    取り、口中点基準の楕円を作る。

    s =（目間隔 + 目〜口距離）/ 2 はポーズ（ヨー・ピッチ）でほぼ不変。
    口角間距離（横顔で0に崩壊）や鼻口距離（横顔で鼻の突き出し分膨張）に
    サイズを直接依存させない（2026-07-04 ht_clip1 スリバー問題の修正）。
    """
    if kps is None or len(kps) < 5:
        return None
    e1, e2, _nose, ml, mr = kps[0], kps[1], kps[2], kps[3], kps[4]
    mw = float(np.hypot(mr[0] - ml[0], mr[1] - ml[1]))
    mx, my = (ml[0] + mr[0]) / 2, (ml[1] + mr[1]) / 2
    ex, ey = (e1[0] + e2[0]) / 2, (e1[1] + e2[1]) / 2

    inter_eye = float(np.hypot(e2[0] - e1[0], e2[1] - e1[1]))
    eye_mouth = float(np.hypot(mx - ex, my - ey))
    s = (inter_eye + eye_mouth) / 2
    if s < 2:
        return None

    # 顔の「下」方向 = 目中点→口中点（横顔でも安定。鼻→口は使わない）
    if eye_mouth < 2:
        ux, uy = 0.0, 1.0
    else:
        ux, uy = (mx - ex) / eye_mouth, (my - ey) / eye_mouth

    # 正面では実際の口幅に追従、横顔では s ベースの下限で崩壊を防ぐ
    rx = max(mw / 2 * MOUTH_W_RATIO, s * MOUTH_RX_MIN_S)
    ry = s * MOUTH_RY_S
    # 縦横比の安全クランプ（スリバー・過扁平の両方を防ぐ）
    ry = min(max(ry, rx * 0.67), rx * 1.6)
    # 楕円の angle=0 は ry が画面下向き。「下」方向 u に合わせて回転
    angle = float(np.degrees(np.arctan2(uy, ux))) - 90.0
    return {
        "cx": round(mx + ux * s * MOUTH_DOWN_SHIFT_S, 1),
        "cy": round(my + uy * s * MOUTH_DOWN_SHIFT_S, 1),
        "rx": round(rx, 1),
        "ry": round(ry, 1),
        "angle": round(angle, 1),
        "visible": True,
        "src": "auto",
    }


def mouth_to_ellipse_lmk(lmk, kps=None) -> dict | None:
    """
    口元のみマスク（106点ランドマーク版）: 唇+小鼻+下顎輪郭の実点群に、
    顔の「下」方向へ整列した向き付きバウンディング楕円を当てる。

    横顔でも顎輪郭・唇の実点が取れるため、5点式（射影で口角が崩壊・
    鼻先が突き出す）より遥かに安定する（2026-07-04 横顔判定強化）。
    向きを滑らかに変化する顔の下方向に固定するので、フレーム間で
    長短軸が入れ替わらず時間的にも安定する。
    """
    if lmk is None or len(lmk) < 106 or kps is None or len(kps) < 5:
        return None
    lmk = np.asarray(lmk, dtype=float)
    mouth = lmk[LMK_MOUTH]
    mc = mouth.mean(axis=0)

    # 顔の「下」方向 = 目中点→口中点（ヨー・ピッチに頑健）
    e_mid = (np.asarray(kps[0], float) + np.asarray(kps[1], float)) / 2
    dv = mc - e_mid
    dn = float(np.hypot(dv[0], dv[1]))
    if dn < 2:
        return None
    u = dv / dn                       # 下方向
    w = np.array([-u[1], u[0]])       # 横方向（下方向の直交）

    # 口中点より顎側（下方向成分が正）にある輪郭点 = 下顎〜顎先
    contour = lmk[LMK_CONTOUR]
    below = contour[(contour - mc) @ u > 0]
    pts = [mouth, lmk[LMK_NOSE_BOTTOM]]
    if len(below):
        pts.append(below)
    pts = np.vstack(pts)

    # 顔座標系（u, w）へ射影し、パーセンタイルで外れ値（横顔の奥側推定点）を除去
    a = (pts - mc) @ u
    b = (pts - mc) @ w
    a_lo, a_hi = np.percentile(a, [MOUTH_LMK_PCT, 100 - MOUTH_LMK_PCT])
    b_lo, b_hi = np.percentile(b, [MOUTH_LMK_PCT, 100 - MOUTH_LMK_PCT])
    ry = (a_hi - a_lo) / 2 * MOUTH_LMK_PAD
    rx = (b_hi - b_lo) / 2 * MOUTH_LMK_PAD
    if rx < 2 or ry < 2:
        return None
    # 縦横比の安全クランプ（ランドマーク異常でも退化楕円を出さない）
    ry = min(max(ry, rx * 0.5), rx * 2.2)
    center = mc + (a_lo + a_hi) / 2 * u + (b_lo + b_hi) / 2 * w

    # angle=0 は ry が画面下向き。「下」方向 u に合わせて回転
    angle = float(np.degrees(np.arctan2(u[1], u[0]))) - 90.0
    return {
        "cx": round(float(center[0]), 1),
        "cy": round(float(center[1]), 1),
        "rx": round(float(rx), 1),
        "ry": round(float(ry), 1),
        "angle": round(angle, 1),
        "visible": True,
        "src": "auto",
    }


def _oriented_ellipse(pts: np.ndarray, u: np.ndarray, mc: np.ndarray,
                      pct: float, pad: float) -> tuple:
    """点群 pts を顔座標系(u=下方向, w=横方向)へ射影し、パーセンタイルで
    外れ値を除いた向き付きバウンディング楕円 (center, rx, ry) を返す。"""
    w = np.array([-u[1], u[0]])
    a = (pts - mc) @ u
    b = (pts - mc) @ w
    a_lo, a_hi = np.percentile(a, [pct, 100 - pct])
    b_lo, b_hi = np.percentile(b, [pct, 100 - pct])
    ry = (a_hi - a_lo) / 2 * pad
    rx = (b_hi - b_lo) / 2 * pad
    center = mc + (a_lo + a_hi) / 2 * u + (b_lo + b_hi) / 2 * w
    return center, rx, ry, (a_lo, a_hi, b_lo, b_hi)


def ellipse_from_lmk68(lmk68, region: str) -> dict | None:
    """
    3DDFA_V2 の iBUG 68点(3D)から領域楕円を作る（大ポーズ対応の本命）。

    口元 / 顔全体のどちらも、顔の「下」方向へ整列した向き付きバウンディング
    楕円で構築する。3DMM フィット由来なので真横でも顎・唇が正確。
    """
    if lmk68 is None or len(lmk68) < 68:
        return None
    lmk = np.asarray(lmk68, dtype=float)[:, :2]
    # 顔の「下」方向 = 目中点→口中点（ヨー・ピッチに頑健）
    e_mid = (lmk[LMK68_REYE].mean(0) + lmk[LMK68_LEYE].mean(0)) / 2
    m_mid = lmk[LMK68_MOUTH].mean(0)
    dv = m_mid - e_mid
    dn = float(np.hypot(dv[0], dv[1]))
    if dn < 2:
        return None
    u = dv / dn
    angle = float(np.degrees(np.arctan2(u[1], u[0]))) - 90.0

    if region == "mouth":
        mc = m_mid
        w = np.array([-u[1], u[0]])
        # 「小鼻〜顎先」を必ず内側に収める対象点: 小鼻(31-35)+口+顎先(点8)。
        # 顎先は横顔でも取りこぼさないよう u フィルタでなく明示インデックスで含める
        target = np.vstack([lmk[LMK68_NOSE_BOTTOM], lmk[LMK68_MOUTH],
                            lmk[LMK68_CHIN]])
        # 横顔度(ヨー): 小鼻中心の横オフセット。正面≈0、横顔で増大。これに応じて
        # 小鼻側へ余裕を足すので、正面の口元は太らせず横顔だけ小鼻を厚く含める
        _au = (target - mc) @ u
        span_u = float(_au.max() - _au.min())
        yaw_w = float((lmk[LMK68_NOSE_BOTTOM].mean(0) - mc) @ w)
        profile = min(1.0, abs(yaw_w) / (span_u + 1e-6))
        # 縦(小鼻側/上)は正面でも一定量含める(BASE)＋横顔ほど追加。
        # 横(前方)は小鼻が前に張り出す横顔のときだけ足す
        nose_ext_v = span_u * (MOUTH68_NOSE_BASE + MOUTH68_NOSE_EXT * profile)
        nose_ext_h = span_u * MOUTH68_NOSE_EXT * profile

        # 縦(u): 小鼻〜顎先を min/max で確実に含める。
        # 小鼻側(上)へ nose_ext_v、顎側(下)へ chin_ext ぶん余裕を足す。
        # 顎側を足さないと、横顔で顎先が楕円から出る（実測 83%）
        a = (target - mc) @ u
        a_lo, a_hi = a.min(), a.max()
        a_lo -= nose_ext_v
        a_hi += span_u * MOUTH68_CHIN_EXT
        ry = (a_hi - a_lo) / 2 * MOUTH68_PAD_V
        center_u = (a_lo + a_hi) / 2

        # 横(w): 口幅から、頬（顔輪郭）側へ CHEEK_FRAC ぶん広げる（左右の各側で）。
        # 横顔では顔輪郭が奥へ回り込むので、頬の取り込みを profile で減衰させる。
        # そうしないと楕円の中心が頬へ引っ張られ、マスクが口元から後ろへずれる
        cheek_frac = MOUTH68_CHEEK_FRAC * (1.0 - profile)
        bm = (lmk[LMK68_MOUTH] - mc) @ w
        bm_lo, bm_hi = bm.min(), bm.max()
        jaw = lmk[LMK68_JAW]
        ja_u = (jaw - mc) @ u
        ja_w = (jaw - mc) @ w
        band = np.abs(ja_u) < span_u             # 口〜顎の高さ帯の輪郭＝頬
        cheek_hi = ja_w[band].max() if band.any() else bm_hi
        cheek_lo = ja_w[band].min() if band.any() else bm_lo
        b_hi = bm_hi + cheek_frac * max(0.0, cheek_hi - bm_hi)
        b_lo = bm_lo + cheek_frac * min(0.0, cheek_lo - bm_lo)
        # 横顔では小鼻が口より前方へ突き出し、口幅の外へ出る。縦だけでなく横も
        # 「小鼻〜顎先」の実位置を必ず内包し、さらに小鼻が張り出す側へ余裕を足す
        tw = (target - mc) @ w
        b_lo = min(b_lo, float(tw.min()))
        b_hi = max(b_hi, float(tw.max()))
        if yaw_w >= 0:
            b_hi += nose_ext_h
        else:
            b_lo -= nose_ext_h
        rx = (b_hi - b_lo) / 2 * MOUTH68_PAD_H
        center_w = (b_lo + b_hi) / 2

        center = mc + center_u * u + center_w * w
        if rx < 2 or ry < 2:
            return None
        # 明示的な包含: 小鼻・顎先が確実に内側へ（縁ぎりぎりを避ける小さめ余白）。
        # 横を頬まで広げた分 off-axis の顎先が収まりやすく、縦の膨張も抑えられる
        au = (target - center) @ u
        aw = (target - center) @ w
        need = np.sqrt(np.max((aw / rx) ** 2 + (au / ry) ** 2))
        s = min(1.5, max(1.0, float(need))) * MOUTH68_CONTAIN_MARGIN
        rx *= s
        ry *= s
    else:
        # 顔全体: 眉〜顎の全点で楕円を作り、額・髪側（上方向 -u）へ拡張
        mc = lmk.mean(0)
        center, rx, ry, ext = _oriented_ellipse(
            lmk, u, mc, FACE_LMK_PCT, 1.0)
        rx *= FACE_LMK_W_PAD
        a_lo, a_hi, _b_lo, _b_hi = ext
        span = a_hi - a_lo                    # 眉→顎の縦スパン
        top_ext = span * FACE_LMK_TOP_EXT     # 額・髪ぶんを上へ足す
        ry = (span + top_ext) / 2
        # 中心を上（-u）へ寄せて上端を伸ばす
        center = center - u * (top_ext / 2)
        if rx < 2 or ry < 2:
            return None

    return {
        "cx": round(float(center[0]), 1),
        "cy": round(float(center[1]), 1),
        "rx": round(float(rx), 1),
        "ry": round(float(ry), 1),
        "angle": round(angle, 1),
        "visible": True,
        "src": "auto",
    }


def detection_to_ellipse(bbox, kps, region: str, lmk=None,
                         lmk68=None) -> dict | None:
    if lmk68 is not None:
        e = ellipse_from_lmk68(lmk68, region)
        if e is not None:
            return e
    if region == "mouth":
        m = mouth_to_ellipse_lmk(lmk, kps) if lmk is not None else None
        if m is None:
            m = mouth_to_ellipse(bbox, kps)
        return m or face_to_ellipse(bbox, kps)
    return face_to_ellipse(bbox, kps)


# 横顔などで 3DDFA がミスフィットし、rx/ry/cx/cy が単発で大きく跳ねて
# マスクがフリックする。前フレーム(採用値)からの変化量を「サイズ比」で
# 外れ値クランプし、軽い EMA をかける。通常の動き(サイズ比で小さい変化)は
# 素通りなので速い動きに遅れない。跳ねだけを抑える。
STAB_POS_CLAMP = 0.35   # 位置(cx,cy)の1フレーム許容変化（前楕円の平均半径比）
STAB_SIZE_CLAMP = 0.30  # サイズ(rx,ry)の1フレーム許容変化（前値比）
STAB_ANG_CLAMP = 20.0   # 角度の1フレーム許容変化（度）
STAB_EMA = 0.7          # 軽EMA（現在値の重み。高いほど遅れない）


def stabilize_ellipse(prev: dict | None, e: dict | None) -> dict | None:
    """前フレームの採用楕円 prev を基準に、新楕円 e の単発ポップを抑える。
    prev が無ければ（トラック先頭など）そのまま返す。"""
    if prev is None or e is None:
        return e

    def _clamp(v, p, lim):
        d = max(-lim, min(lim, v - p))
        return p + d

    R = max(1.0, (float(prev["rx"]) + float(prev["ry"])) / 2)
    cx = _clamp(float(e["cx"]), float(prev["cx"]), STAB_POS_CLAMP * R)
    cy = _clamp(float(e["cy"]), float(prev["cy"]), STAB_POS_CLAMP * R)
    rx = _clamp(float(e["rx"]), float(prev["rx"]), STAB_SIZE_CLAMP * float(prev["rx"]))
    ry = _clamp(float(e["ry"]), float(prev["ry"]), STAB_SIZE_CLAMP * float(prev["ry"]))
    # 角度は最短弧でクランプ
    da = ((float(e.get("angle", 0.0)) - float(prev.get("angle", 0.0)) + 180.0)
          % 360.0) - 180.0
    da = max(-STAB_ANG_CLAMP, min(STAB_ANG_CLAMP, da))

    a = STAB_EMA
    out = dict(e)
    out["cx"] = round(a * cx + (1 - a) * float(prev["cx"]), 1)
    out["cy"] = round(a * cy + (1 - a) * float(prev["cy"]), 1)
    out["rx"] = round(a * rx + (1 - a) * float(prev["rx"]), 1)
    out["ry"] = round(a * ry + (1 - a) * float(prev["ry"]), 1)
    out["angle"] = round(float(prev.get("angle", 0.0)) + a * da, 1)
    return out


class DetectJob:
    def __init__(self, pid: str, start: int = 0, end: int | None = None,
                 region: str = "face"):
        self.pid = pid
        self.start = start
        self.end = end
        self.region = region if region in ("face", "mouth") else "face"
        self.state = "running"   # running / done / error / cancelled
        self.progress = 0.0
        self.message = "初期化中..."
        self.cancel_requested = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start_job(self):
        self._thread.start()

    def status(self) -> dict:
        return {"state": self.state, "progress": round(self.progress, 3),
                "message": self.message}

    def _run(self):
        try:
            self._detect()
        except Exception as e:
            traceback.print_exc()
            self.state = "error"
            self.message = f"エラー: {e}"

    def _detect(self):
        proj = prj.load_project(self.pid)
        video = proj["video"]
        src = video_manager.get(video["path"])
        total = video["frame_count"]
        end = min(self.end if self.end is not None else total, total)

        self.message = "検出モデル読み込み中..."
        app = _get_face_app()

        tracks: list[_Track] = []
        finished: list[_Track] = []
        next_tid = 1

        self.message = "顔検出中..."
        for fi in range(self.start, end):
            if self.cancel_requested:
                self.state = "cancelled"
                self.message = "キャンセルされました"
                return

            frame = src.get_frame(fi)
            if frame is None:
                break

            faces = app.get(frame)
            bboxes = [f.bbox.astype(float).tolist() for f in faces]
            kps_list = [
                f.kps.astype(float).tolist() if f.kps is not None else None
                for f in faces
            ]
            lmk_list = [
                (f.landmark_2d_106.astype(float).tolist()
                 if getattr(f, "landmark_2d_106", None) is not None else None)
                for f in faces
            ]
            # 3DDFA_V2 の 68点3D（大ポーズ対応の本命。使えなければ None）
            lmk68_list = [landmarks3d.get_landmarks_68(frame, bb)
                          for bb in bboxes]

            # 貪欲 IoU マッチング
            unmatched = list(range(len(bboxes)))
            for tr in tracks:
                best_j, best_iou = -1, IOU_THRESHOLD
                for j in unmatched:
                    v = _iou(tr.bbox, bboxes[j])
                    if v > best_iou:
                        best_j, best_iou = j, v
                if best_j >= 0:
                    unmatched.remove(best_j)
                    tr.bbox = bboxes[best_j]
                    tr.last_frame = fi
                    tr.misses = 0
                    ell = detection_to_ellipse(tr.bbox, kps_list[best_j],
                                               self.region, lmk_list[best_j],
                                               lmk68_list[best_j])
                    if ell:
                        ell = stabilize_ellipse(tr.last_ell, ell)
                        tr.keyframes[str(fi)] = ell
                        tr.last_ell = ell
                else:
                    tr.misses += 1

            for j in unmatched:
                tr = _Track(next_tid, bboxes[j], fi)
                ell = detection_to_ellipse(bboxes[j], kps_list[j],
                                           self.region, lmk_list[j],
                                           lmk68_list[j])
                if ell:
                    tr.keyframes[str(fi)] = ell
                    tr.last_ell = ell
                next_tid += 1
                tracks.append(tr)

            still = []
            for tr in tracks:
                if tr.misses > MAX_MISSES:
                    finished.append(tr)
                else:
                    still.append(tr)
            tracks = still

            self.progress = (fi - self.start + 1) / max(1, end - self.start)

        finished.extend(tracks)
        finished = [t for t in finished if len(t.keyframes) >= MIN_TRACK_LEN]
        finished.sort(key=lambda t: len(t.keyframes), reverse=True)

        # 最新のプロジェクトに反映（検出中のユーザー編集を上書きしないよう再ロード）
        self.message = "結果を保存中..."
        proj = prj.load_project(self.pid)

        # 以前の自動検出人物（手動キーフレームなし）は置き換え対象として除去
        keep_persons = []
        for p in proj["persons"]:
            kfs = proj["keyframes"].get(p["id"], {})
            has_manual = any(kf.get("src") == "manual" for kf in kfs.values())
            if has_manual or not kfs:
                keep_persons.append(p)
            else:
                proj["keyframes"].pop(p["id"], None)
        proj["persons"] = keep_persons

        for tr in finished:
            person = prj.new_person(proj["persons"])
            person["region"] = self.region
            if self.region == "mouth":
                person["label"] = person["label"].replace("人物", "口元")
            proj["persons"].append(person)
            proj["keyframes"][person["id"]] = tr.keyframes

        prj.save_project(proj)
        self.progress = 1.0
        self.state = "done"
        self.message = f"完了: {len(finished)} 人物を検出"


# pid -> DetectJob
_jobs: dict[str, DetectJob] = {}
_jobs_lock = threading.Lock()


def start_detect(pid: str, start: int = 0, end: int | None = None,
                 region: str = "face") -> dict:
    with _jobs_lock:
        job = _jobs.get(pid)
        if job and job.state == "running":
            return {"error": "検出ジョブが実行中です"}
        job = DetectJob(pid, start, end, region)
        _jobs[pid] = job
        job.start_job()
        return job.status()


def detect_status(pid: str) -> dict | None:
    job = _jobs.get(pid)
    return job.status() if job else None


def cancel_detect(pid: str) -> bool:
    job = _jobs.get(pid)
    if job and job.state == "running":
        job.cancel_requested = True
        return True
    return False
