"""
tracker.py - SAM 2 高精度トラッキング（バックグラウンドジョブ）

v2アーキテクチャの中核実装:
  1. ショット（カット）検出        … ヒストグラム相関。伝播はカットをまたがない
  2. アンカー発見                  … SCRFD検出 + ArcFace照合で「確実に本人」の点を集める
  3. SAM 2 時間伝播                … アンカーをシードにマスクを前後双方向へ伝播
  4. 自己検証                      … 伝播マスクを定期的に ArcFace 再照合（すり替わり検知）
  5. Claude 自動クリック           … 残った穴の頭部座標を Claude が回答 → 再シード
     （ANTHROPIC_API_KEY があるときのみ）

出力は回転付き楕円キーフレーム（マスクに cv2.fitEllipse を適用）で、
既存のセーブファイル形式・エディターにそのまま載る。
"""
from __future__ import annotations

import bisect
import gc
import shutil
import statistics
import tempfile
import threading
import traceback
from pathlib import Path

import cv2
import numpy as np

import math

from . import autoclick, landmarks3d, project as prj
from .detect import (
    MOUTH_ASPECT, detect_face_robust, ellipse_from_lmk68, get_face_app,
    mouth_to_ellipse, mouth_to_ellipse_lmk,
)
from .render import _lerp_angle, apply_blurs
from .video import manager as video_manager

BASE_DIR = Path(__file__).parent.parent
SAM2_CKPT = BASE_DIR / "models" / "sam2.1_hiera_base_plus.pt"
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# ── パラメータ ───────────────────────────────────────────────────────────────
SHOT_HIST_THRESH = 0.55   # ヒストグラム相関がこれ未満ならカットとみなす
ANCHOR_SCAN_EVERY = 3     # アンカー探索の検出間隔（フレーム）
ANCHOR_SIM = 0.40         # リファレンスと同一人物とみなす cosine 類似度
VERIFY_EVERY = 15         # 自己検証の間隔（フレーム）
VERIFY_REJECT_SIM = 0.18  # これ未満なら「別人にすり替わった」と判定
WINDOW = 240              # SAM 2 を一度に回すフレーム窓（メモリ上限対策）
LOW_RAM_WINDOW = 120      # 低RAMモード（夜間バッチ）の窓。SAM 2 のCPU保持テンソルが半減
SAM_MAX_SIDE = 960        # SAM 2 に渡すフレームの長辺
MIN_MASK_AREA = 120       # これ未満のマスクはノイズ（縮小後ピクセル数）
GAP_MIN = 12              # この長さ以上の穴を Claude 自動クリック対象にする
AUTOCLICK_MAX_CALLS = 12  # 1ジョブあたりの Claude 呼び出し上限（コスト対策）
MAX_PROMPTS_PER_WIN = 4   # 1窓・1オブジェクトあたりのアンカープロンプト数
AMBIG_COOLDOWN = 60       # 曖昧判定: 同一人物への再質問を控えるフレーム数
AMBIG_MAX_CALLS = 80      # 曖昧判定: 1ジョブあたりの Claude 呼び出し上限

# ── 口元: 3DDFA フィット破綻の足切り（信頼度ゲート） ──────────────────
# 破綻したフィットをそのまま楕円にすると、目・額・耳の上にマスクが乗る。
# landmarks3d.get_landmarks_68_conf の「怪しさ」で足切りし、捨てたフレームは
# 頭部楕円基準の相対補間に委ねる（頑健な頭部トラックへ位置を預ける）。
#
# しきい値は素材ごとに分布が動くため **絶対値で決め打ちしない**。トラック内の
# 中央値の MOUTH_GATE_K 倍を採り、FLOOR/CEIL で常識的な範囲に収める。
# 実測（hy_clip1/hy_clip2）: 中央値は 0.0095 / 0.0276 と3倍近く動くが、
# 中央値×2 なら双方で崩壊フレームだけを弾けた。
# 検証: .company/engineering/docs/mouth-mask-accuracy-hy.md
MOUTH_GATE_K = 2.0          # トラック内中央値の何倍までを信用するか
MOUTH_GATE_FLOOR = 0.012    # しきい値の下限（良素材で厳しくしすぎない）
MOUTH_GATE_CEIL = 0.060     # しきい値の上限（悪素材でも明らかな崩壊は必ず弾く）
MOUTH_GATE_MIN_SAMPLES = 20  # これ未満の実測数ではゲートを効かせない
# ⚠ トラック単位のゲート無効化（MOUTH_GATE_OFF_MEDIAN = 0.012）を 2026-07-17 に
# 一度入れたが、**同日中に revert した**。同じ過ちを繰り返さないための記録:
#
# 導入時の根拠は「median 0.011以上のトラックは手動/自動の弁別が逆転する
# （hy_clip1 0.79x）ので、ゲートを切れば無駄な指摘が42%減る」だった。
# だがこの弁別比も「無駄」も、**汚染されたラベルの産物**だった:
# hy_clip1 は 17,822 フレーム中 人が見たのは 13% だけで、残り 15,082 フレームが
# 「auto ＝ AIが成功した」として負例に入っていた。そこには実際にマスクが頬に
# 乗って口が露出したフレーム（f8390 等）が含まれていた。
#
# 負例をレビュー済み区間のみに限定して測り直すと:
#   拾えた失敗 320→317（-3） / **確実な無駄 251→251（±0）** /
#   抑制されたのは全て「人が見ていない＝正誤不明」の指摘 2,004件
# ＝ 確実な失敗検知を失うだけで、確実な無駄は1件も減らない。
#
# 教訓: **「auto」は「AIが成功した」ではなく「人が触らなかった」**。
# レビューが薄い動画ほどゲートが悪く見える。効果測定の前に
# tools/extract_failure_dataset.py の reviewed-only を必ず通すこと。
# 検証: docs/manual-correction-as-failure-labels.md
MOUTH_GATE_MIN_KEEP = 0.35  # 実測をこの割合より下に減らさない（補間の足場を残す）
MOUTH_GATE_DEVIATION = 0.5  # 信用フレームの補間からこの倍率(平均半径比)以上

# ── 要確認フラグ（レビュー送り）: 棄却とは別系統・配置に一切影響しない ──
#
# ゲート（棄却）とは目的が違う。棄却は「捨てて補間したほうがマシ」なフレームだけを
# 落とす保守的な操作で、緩めると悪化する（下の MOUTH_GATE_DEVIATION の根拠を参照:
# lmk_conf だけで捨てると良好区間で IoU -0.023）。
# こちらは **人に「ここを見て」と伝えるだけ**。外しても確認が数フレーム増えるだけで
# マスクは絶対に壊れない。だから棄却よりずっと緩く張れる。
#
# 値は手動修正ラベル 2,243件（レビュー率70%以上の7本）で較正した結果
# （tools/calibrate_gate.py, 2026-07-17）。**7本すべてが同じ組を選んだ**:
#   失敗リコール 14.1% → **35.4%**（別動画クロス検証の平均）/ レビュー予算 4.4%
# 予算爆発なし（最大 ht_clip8 の10.2%）。
#
# ⚠ この定数を「棄却」側に流用してはいけない。較正の目的関数は
# 「人が触ったフレームを拾えたか」であって「マスクが良くなったか」ではない。
# 設計: .company/engineering/docs/manual-correction-as-failure-labels.md
MOUTH_REVIEW_K = 0.5
MOUTH_REVIEW_FLOOR = 0.012
MOUTH_REVIEW_CEIL = 0.030
MOUTH_REVIEW_DEVIATION = 0.02
                            # 離れていて、かつ怪しい場合のみ棄却する。
                            # 怪しさ単独で捨てると「怪しいが当たっている」実測まで
                            # 落ちて逆に悪化する（実測: 良好区間で IoU -0.023）

# ── AI 楕円補正（幾何が苦手な低信頼フレームだけ Claude に範囲を補正させる） ──
AI_CORRECT_MAX_CALLS = 100  # 1動画あたりの Claude 呼び出し上限（オーナー指定）
AI_CORRECT_SUSP = 0.35      # この疑わしさ以上のフレームを補正候補にする
AI_CORRECT_MIN_SPAN = 3     # これ未満の孤立フレームは補正しない
AI_CORRECT_MERGE_GAP = 6    # この間隔以内の候補は1スパンに結合
AI_CORRECT_SCALE_CLAMP = (0.7, 2.5)  # 補正サイズ比の下限・上限
# 口元の AI 範囲補正は **無効**（2026-07-11）。
# 実測で Claude の口元矩形は幾何(3DDFA)より悪い（IoU中央 0.258 → 0.198、
# 中心ズレ 0.77 → 0.95〜1.2半径）うえ、**系統的に小さい**（面積比の中央 0.49〜0.79）。
# それがスパン全体に適用され、幾何が正しく出した楕円を縦に潰していた。
# 実害の証拠: hy_clip2 の f4180-4210 は jitter 一致性 0.006〜0.010（極めて健全）で
# 幾何は ry≈90-100 を出しているのに、保存マスクは ry≈56（縦 0.61倍）に縮み、
# 顎先がマスクから外れていた。1クリップあたり API 338回・$1.33 を払って悪化させていた。
# 検証: .company/engineering/docs/mouth-mask-accuracy-hy.md
AI_CORRECT_MOUTH = False

# 顔全体の AI 範囲補正も **無効**（2026-07-17）。口元と同じ壊れ方を実測した。
#
# a2_clip1 で追跡後の keyframes と、生SAM2楕円(detections)から幾何パスを再生した
# 値を突き合わせると、f62-99 の全域で
#   実測値 = 再生値 + 定数オフセット(-47.8, -127.3)px × 定数スケール 0.70
# が完全に一致した。0.70 は AI_CORRECT_SCALE_CLAMP の下限そのもの＝ Claude は
# さらに小さい矩形を返してクランプに張り付いていた（口元の「系統的に小さい」と同じ）。
# 結果、マスクは頭から 200px 以上離れた枕の上に居座った。
#
# 構造的な理由（決定7と同じ）: 顔で AI 補正が発火するのは**アンカー不在**の区間、
# つまり ArcFace が本人と照合できないほど崩れたフレーム（うつむき・横顔・ブラー）。
# 静止画1枚では人間でも曖昧で、VLM に届く情報が無い。さらに「1フレームで測った
# 平行移動デルタをスパン全体に配る」ため、頭が動いている区間ほど破綻が増幅する
# （taper はスパン端にしか効かず、worst からの時間距離では減衰しない）。
#
# ⚠ 2026-07-11 決定8 の「痛恨のミス」（Claude が幾何に負けると分かったのに本番で
# 動いていた同じ仕組みを止めなかった）を、口元だけ直して顔で繰り返していた。
# 検証: .company/engineering/debug-log/2026-07-17-face-mask-ai-correct-drift.md
AI_CORRECT_FACE = False

AI_CORRECT_TAPER = 4        # スパン端でデルタを0へ滑らかに戻すフレーム数

# ── 遮蔽ブリッジ（手・髪などでマスクが壊れた区間を前後の良フレームで補間） ──
BRIDGE_DIP_RATIO = 0.55   # 楕円面積が近傍中央値のこれ未満なら「壊れた」候補
BRIDGE_OK_RATIO = 0.80    # 端点として信頼する面積比の下限
BRIDGE_MED_WIN = 45       # 面積中央値を取る近傍キーフレーム数（片側）
BRIDGE_MAX_SPAN = 90      # これより長い壊れ区間はブリッジしない（自動クリック/QCに任せる）
BRIDGE_SEARCH = 24        # 良い端点を外側へ探す最大フレーム数
BRIDGE_VERIFY_TRIES = 3   # 1端点あたりの Sonnet 検証リトライ上限
BRIDGE_MAX_CALLS = 60     # 端点検証の Claude 呼び出し上限/ジョブ
BLOWUP_RATIO = 2.5        # 両軸がこの倍率を超えたら「マスク暴走」（中心ごと作り直し）
OVER_RATIO = 1.4          # 各軸の上限倍率。超過分は軸別に刈り込む（髪など）

_sam_predictor = None
_sam_lock = threading.Lock()


def _get_sam():
    global _sam_predictor
    with _sam_lock:
        if _sam_predictor is None:
            import torch
            from sam2.build_sam import build_sam2_video_predictor
            if not SAM2_CKPT.exists():
                raise FileNotFoundError(
                    f"SAM 2 チェックポイントがありません: {SAM2_CKPT}")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _sam_predictor = build_sam2_video_predictor(
                SAM2_CFG, str(SAM2_CKPT), device=device)
        return _sam_predictor


# ── ユーティリティ ───────────────────────────────────────────────────────────

def _hist(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def _mask_to_ellipse(mask: np.ndarray, scale: float) -> dict | None:
    """SAM 2 のマスク（縮小解像度）→ 元解像度の回転付き楕円。"""
    m = mask.astype(np.uint8)
    if int(m.sum()) < MIN_MASK_AREA:
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    (cx, cy), (dw, dh), ang = cv2.fitEllipse(cnt)
    rx, ry = dw / 2, dh / 2
    # 軸を正規化（常に ry >= rx）して fitEllipse の軸入れ替わりによる
    # 角度の飛び（0°↔90°）を抑える
    if rx > ry:
        rx, ry = ry, rx
        ang += 90.0
    ang = (ang + 90.0) % 180.0 - 90.0
    # わずかに広げて安全側へ（漏れ防止）
    return {
        "cx": round(cx / scale, 1),
        "cy": round(cy / scale, 1),
        "rx": round(rx * 1.08 / scale, 1),
        "ry": round(ry * 1.08 / scale, 1),
        "angle": round(ang, 1),
        "visible": True,
        "src": "auto",
    }


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _expand_to_face_bbox(ell: dict, bbox, cap: float = 2.0,
                         margin: float = 1.10) -> dict:
    """
    顔検出 bbox の上下左右の中点（額・顎・両頬）が楕円に収まるまで、
    中心と角度を保ったまま楕円を等方拡大する。

    SAM 2 のマスクは前髪・うつむき等で「見えている肌」だけに縮むことがあり、
    そのまま楕円化すると口元・顎がブラーから外れる。顔検出をサイズの下限として
    安全側（漏れ＝致命的なので拡大のみ、縮小はしない）に補正する。
    margin: 境界ぴったりだとフェザーでブラーが薄まる+検出誤差もあるため、
    対象点が縁ではなく内側に入るよう掛ける余裕率。
    拡大不要なら ell をそのまま返す。
    """
    x1, y1, x2, y2 = bbox
    pts = [((x1 + x2) / 2, y1), ((x1 + x2) / 2, y2),
           (x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2)]
    a = math.radians(ell.get("angle", 0.0))
    rx = max(ell["rx"], 2.0)
    ry = max(ell["ry"], 2.0)
    s = 1.0
    for px, py in pts:
        dx, dy = px - ell["cx"], py - ell["cy"]
        u = dx * math.cos(a) + dy * math.sin(a)
        v = -dx * math.sin(a) + dy * math.cos(a)
        need = math.sqrt((u / rx) ** 2 + (v / ry) ** 2)
        # はみ出す点だけでなく「縁ぎりぎり」の点もマージン帯の内側へ押し込む。
        # 拡大後は need/s = 1/margin になるため、再適用しても増殖しない
        if need * margin > 1.0:
            s = max(s, need * margin)
    s = min(s, cap)
    if s <= 1.02:  # 誤差程度なら触らない
        return ell
    out = dict(ell)
    out["rx"] = round(rx * s, 1)
    out["ry"] = round(ry * s, 1)
    return out


# _fit_face_ellipse の許容範囲。標準置き換え楕円（1.15/1.25）は必ずこの範囲に
# 収まるため、置き換え後の再判定で振動しない
FIT_MAX_X = 1.35   # 横の広がり / 顔bbox半幅 の上限
FIT_MAX_Y = 1.45   # 縦の広がり / 顔bbox半高 の上限


def _fit_face_ellipse(ell: dict, bbox) -> dict:
    """
    追跡楕円を顔検出 bbox と突き合わせる（顔全体モードの補正の入口）。

    スケール補正はしない。判定のみ:
    - カバーOK（bboxの額・顎・両頬の中点が楕円内）かつ 過大でない
      （回転込みの縦横の広がりが FIT_MAX 以内）→ マスク由来の楕円をそのまま採用
    - どちらかを満たさない → bbox から合成した標準顔楕円に置き換え

    以前の「均等拡大でカバーを満たす」方式は、回転楕円や縦長マスクで
    拡大が累積し、顔の1.5〜2倍のマスクを量産した（実測）。検出がある以上
    その形を信じて置き換える方が、小さくも大きくもならない。
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    hw = max((x2 - x1) / 2, 2.0)
    hh = max((y2 - y1) / 2, 2.0)
    cxb, cyb = (x1 + x2) / 2, (y1 + y2) / 2

    a = math.radians(ell.get("angle", 0.0))
    rx = max(float(ell["rx"]), 2.0)
    ry = max(float(ell["ry"]), 2.0)
    ext_x = math.hypot(rx * math.cos(a), ry * math.sin(a))
    ext_y = math.hypot(rx * math.sin(a), ry * math.cos(a))
    size_ok = ext_x <= hw * FIT_MAX_X and ext_y <= hh * FIT_MAX_Y

    cover_ok = True
    if size_ok:
        for px, py in ((cxb, y1), (cxb, y2), (x1, cyb), (x2, cyb)):
            dx, dy = px - float(ell["cx"]), py - float(ell["cy"])
            u = dx * math.cos(a) + dy * math.sin(a)
            v = -dx * math.sin(a) + dy * math.cos(a)
            if (u / rx) ** 2 + (v / ry) ** 2 > 1.0:
                cover_ok = False
                break

    if size_ok and cover_ok:
        return ell
    out = dict(ell)
    out.update(cx=round(cxb, 1), cy=round(cyb, 1),
               rx=round(hw * 1.15, 1), ry=round(hh * 1.25, 1), angle=0.0)
    return out


def _nearest_bbox(bboxes: dict[int, list], frame: int,
                  max_gap: int) -> list | None:
    """frame に最も近い顔bboxを ±max_gap の範囲で探す。"""
    for off in range(max_gap + 1):
        for f in (frame - off, frame + off):
            bb = bboxes.get(f)
            if bb is not None:
                return bb
    return None


def _shrink_to_face_bbox(ell: dict, bbox) -> dict:
    """
    サイズ上限ガード（_expand_to_face_bbox が下限、こちらが上限）。
    - 両軸が BLOWUP_RATIO 倍超 = マスク暴走（背景に流れた等）。中心も信用できない
      ため bbox から合成した標準顔楕円に置き換える
    - 片軸だけ OVER_RATIO 倍超 = マスクが髪・首などを取り込んで伸びた状態。
      その軸だけ上限まで刈り込む（後段の下限ガードが顔カバーを保証する）
    正常範囲なら ell をそのまま返す。
    """
    # bbox は numpy float の場合がある。キーフレームに書き込む値になるため
    # JSON化できるよう Python float に正規化する
    x1, y1, x2, y2 = [float(v) for v in bbox]
    hw = max((x2 - x1) / 2, 2.0)
    hh = max((y2 - y1) / 2, 2.0)
    if ell["rx"] > hw * BLOWUP_RATIO and ell["ry"] > hh * BLOWUP_RATIO:
        out = dict(ell)
        out.update(cx=round((x1 + x2) / 2, 1), cy=round((y1 + y2) / 2, 1),
                   rx=round(hw * 1.25, 1), ry=round(hh * 1.35, 1), angle=0.0)
        return out
    if ell["rx"] <= hw * OVER_RATIO and ell["ry"] <= hh * OVER_RATIO:
        return ell
    # 軸を刈り込む際は中心も bbox 中心へ寄せる（±0.25半径以内）。
    # マスクの重心が髪などで下にずれたまま刈ると額/顎が外れるため、
    # 「刈り込み後もbboxをカバーできる」を中心側で保証する
    # （1.25×半径 ≤ OVER_RATIO×半径 なので端点は必ず収まる）
    out = dict(ell)
    if ell["rx"] > hw * OVER_RATIO:
        out["rx"] = round(hw * OVER_RATIO, 1)
        cxb = (x1 + x2) / 2
        out["cx"] = round(min(max(float(ell["cx"]), cxb - hw * 0.25),
                              cxb + hw * 0.25), 1)
    if ell["ry"] > hh * OVER_RATIO:
        out["ry"] = round(hh * OVER_RATIO, 1)
        cyb = (y1 + y2) / 2
        out["cy"] = round(min(max(float(ell["cy"]), cyb - hh * 0.25),
                              cyb + hh * 0.25), 1)
    return out


def _clip_mask_to_bbox(mask: np.ndarray, bbox, scale: float,
                       pad: float = OVER_RATIO) -> np.ndarray | None:
    """
    持ち越しマスクを顔bbox（pad 倍に拡大、縮小解像度）の外で刈り込む。
    マスクが髪・首・体を取り込んだまま次窓へ渡ると膨張が持ち越され続けるため、
    窓境界でアンカーの顔サイズにリセットする。刈り込み後が空なら None。
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) / 2 * scale, (y1 + y2) / 2 * scale
    hw, hh = (x2 - x1) / 2 * scale * pad, (y2 - y1) / 2 * scale * pad
    h, w = mask.shape
    ax1, ay1 = max(0, int(cx - hw)), max(0, int(cy - hh))
    ax2, ay2 = min(w, int(cx + hw) + 1), min(h, int(cy + hh) + 1)
    if ax2 <= ax1 or ay2 <= ay1:
        return None
    out = np.zeros_like(mask)
    out[ay1:ay2, ax1:ax2] = mask[ay1:ay2, ax1:ax2]
    return out


def _kf_area(kf: dict | None) -> float | None:
    if kf is None or not kf.get("visible", True) or kf.get("parked"):
        return None
    return kf["rx"] * kf["ry"]


def _find_occlusion_spans(kfs: dict,
                          expected: dict[int, float] | None = None
                          ) -> list[tuple[int, int, float]]:
    """
    遮蔽（手・髪など）でマスクが壊れた候補区間を検出する。
    シグナル: 楕円面積が期待サイズより急減 / キーフレーム欠落。
    期待サイズは expected（フレーム -> アンカー由来の面積。信頼できる独立ソース）
    を優先し、無いフレームは近傍中央値へフォールバックする。
    返り値: [(開始, 終了, 期待面積)]。開始・終了は壊れたフレームを含む閉区間。
    """
    frames = sorted(int(f) for f in kfs)
    fs, areas = [], []
    for f in frames:
        a = _kf_area(kfs.get(str(f)))
        if a is not None:
            fs.append(f)
            areas.append(a)
    if len(fs) < BRIDGE_MED_WIN:
        return []

    exp_frames = sorted(expected) if expected else []

    def expected_at(f: int) -> float | None:
        """f に最も近いアンカー由来の期待面積（±BRIDGE_MAX_SPAN 以内）。"""
        if not exp_frames:
            return None
        i = bisect.bisect_left(exp_frames, f)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(exp_frames):
                d = abs(exp_frames[j] - f)
                if d <= BRIDGE_MAX_SPAN and (best is None or d < best[0]):
                    best = (d, expected[exp_frames[j]])
        return best[1] if best else None

    # 近傍中央値そのものが壊れている地帯（マスク暴走で巨大楕円が続く区間など）
    # では「正常フレームが壊れ扱い」になり誤ブリッジするため、全体中央値から
    # かけ離れた近傍は判定対象から外す（アンカー由来の期待値はこの汚染を受けない）
    global_med = statistics.median(areas)

    bad: dict[int, float] = {}  # 壊れたフレーム -> 期待面積
    for i, f in enumerate(fs):
        exp = expected_at(f)
        if exp is None:
            lo = max(0, i - BRIDGE_MED_WIN)
            med = statistics.median(areas[lo:i + BRIDGE_MED_WIN])
            if med > global_med * 4:
                continue
            exp = med
        if areas[i] < exp * BRIDGE_DIP_RATIO:
            bad[f] = exp
        # キーフレーム欠落（次の有効フレームまでの穴）も壊れ扱い
        if i + 1 < len(fs) and fs[i + 1] - f > 1:
            for g in range(f + 1, min(fs[i + 1], f + 1 + BRIDGE_MAX_SPAN)):
                bad[g] = exp

    # 近接する壊れフレームを1区間にまとめる（3フレーム以内は同一区間）
    spans: list[list] = []
    for f in sorted(bad):
        if spans and f - spans[-1][1] <= 3:
            spans[-1][1] = f
        else:
            spans.append([f, f, bad[f]])
    return [(s, e, med) for s, e, med in spans]


def _anchor_of(face) -> dict:
    return {
        "bbox": face.bbox.astype(float).tolist(),
        "kps": face.kps.astype(float).tolist() if face.kps is not None else None,
    }


def _to_head_local(x: float, y: float, hk: dict) -> tuple[float, float]:
    a = -math.radians(hk.get("angle", 0.0))
    return (x * math.cos(a) - y * math.sin(a),
            x * math.sin(a) + y * math.cos(a))


def _from_head_local(lx: float, ly: float, hk: dict) -> tuple[float, float]:
    a = math.radians(hk.get("angle", 0.0))
    return (hk["cx"] + lx * math.cos(a) - ly * math.sin(a),
            hk["cy"] + lx * math.sin(a) + ly * math.cos(a))


def convert_ident_to_mouth(ident: "_Identity",
                           refined: dict[int, dict] | None = None) -> None:
    """
    頭部マスク由来のキーフレームを「口元のみ」の楕円に変換する。

    refined: {フレーム番号: 口元楕円} — 頭部領域のランドマーク再検出で
    直接測定できたフレーム（_refine_mouth の結果）。これらはそのまま採用し、
    測定できなかったフレーム（横顔・後ろ向き等）だけを
    「頭部楕円に対する相対位置・相対サイズ」の補間で埋める。
    （口が見えない区間も、口があるはずの位置をブラーし続ける安全側設計）
    """
    refined = refined or {}

    # 相対パラメータの参照点: 直接測定できたフレームを優先、なければアンカー
    refs: list[tuple[int, dict]] = []

    def add_ref(fi: int, m: dict):
        hk = ident.keyframes.get(str(fi))
        if not hk or not hk.get("visible", True):
            return
        lx, ly = _to_head_local(m["cx"] - hk["cx"], m["cy"] - hk["cy"], hk)
        refs.append((fi, {
            "u": lx / max(1.0, hk["rx"]),
            "v": ly / max(1.0, hk["ry"]),
            "s": m["rx"] / max(1.0, hk["rx"]),
            "da": m["angle"] - hk.get("angle", 0.0),
        }))

    if refined:
        for fi in sorted(refined):
            add_ref(fi, refined[fi])
    else:
        for fi in sorted(ident.anchors):
            a = ident.anchors[fi]
            if not a.get("kps"):
                continue
            m = mouth_to_ellipse(a["bbox"], a["kps"])
            if m is not None:
                add_ref(fi, m)

    default = {"u": 0.0, "v": 0.55, "s": 0.5, "da": 0.0}

    def rel_at(frame: int) -> dict:
        if not refs:
            return default
        prev = nxt = None
        for fi, r in refs:
            if fi <= frame:
                prev = (fi, r)
            else:
                nxt = (fi, r)
                break
        if prev and nxt:
            t = (frame - prev[0]) / max(1, nxt[0] - prev[0])
            out = {k: prev[1][k] + (nxt[1][k] - prev[1][k]) * t
                   for k in ("u", "v", "s")}
            out["da"] = _lerp_angle(prev[1]["da"], nxt[1]["da"], t)
            return out
        return (prev or nxt)[1]

    new_kfs: dict[str, dict] = {}
    for fs, hk in ident.keyframes.items():
        if not hk.get("visible", True):
            new_kfs[fs] = hk
            continue
        fi = int(fs)
        if fi in refined:
            # 直接測定できたフレームは実測値をそのまま使う（寄り/引きに強い）
            m = refined[fi]
            new_kfs[fs] = dict(m, visible=True, src="auto")
            continue
        r = rel_at(fi)
        cx, cy = _from_head_local(r["u"] * hk["rx"], r["v"] * hk["ry"], hk)
        rx = max(4.0, r["s"] * hk["rx"])
        new_kfs[fs] = {
            "cx": round(cx, 1), "cy": round(cy, 1),
            "rx": round(rx, 1), "ry": round(rx * MOUTH_ASPECT, 1),
            "angle": round(hk.get("angle", 0.0) + r["da"], 1),
            "visible": True, "src": "auto",
        }
    ident.keyframes = new_kfs


def _detect_face_near(frame_img, hk: dict,
                      margin_factor: float = 0.5,
                      with_conf: bool = False) -> dict | None:
    """
    楕円 hk の周辺を切り出して顔を検出し、hk に最も近い顔の
    {"bbox": [...], "kps": [...]} を元解像度で返す（遠すぎれば None）。

    with_conf=True で 3DDFA フィットの信頼度も測る（摂動再フィット6回ぶんの
    コストがかかるので、口元の実測でのみ有効にする）。
    """
    h, w = frame_img.shape[:2]
    r = max(hk["rx"], hk["ry"])
    mg = int(r * margin_factor) + 8
    x1 = max(0, int(hk["cx"] - r) - mg)
    y1 = max(0, int(hk["cy"] - r) - mg)
    x2 = min(w, int(hk["cx"] + r) + mg)
    y2 = min(h, int(hk["cy"] + r) + mg)
    crop = frame_img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    cs = 1.0
    if max(crop.shape[:2]) < 240:
        cs = 240.0 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=cs, fy=cs)
    faces = get_face_app(with_rec=False).get(crop)
    best, best_d = None, None
    for f in faces:
        if f.kps is None:
            continue
        kps = [[q[0] / cs + x1, q[1] / cs + y1] for q in f.kps.astype(float)]
        bbox = [f.bbox[0] / cs + x1, f.bbox[1] / cs + y1,
                f.bbox[2] / cs + x1, f.bbox[3] / cs + y1]
        lmk = None
        lm106 = getattr(f, "landmark_2d_106", None)
        if lm106 is not None:
            lmk = [[q[0] / cs + x1, q[1] / cs + y1] for q in lm106.astype(float)]
        mx = (kps[3][0] + kps[4][0]) / 2
        my = (kps[3][1] + kps[4][1]) / 2
        d = math.hypot(mx - hk["cx"], my - hk["cy"])
        if best_d is None or d < best_d:
            best, best_d = {"bbox": bbox, "kps": kps, "lmk": lmk}, d
    if best is None or best_d > r * 1.8 + 24:
        return None
    # 3DDFA_V2 の 68点3D（大ポーズ対応の本命）。元解像度フレーム＋元座標bboxで実測。
    # lmk_conf は「フィットの怪しさ」（小さいほど信用できる）。破綻したフィットを
    # そのまま楕円にすると目・額・耳の上にマスクが乗るため、_refine_mouth で
    # トラック内の分布を基準に自己校正したしきい値で足切りする
    if with_conf:
        best["lmk68"], best["lmk_conf"] = landmarks3d.get_landmarks_68_conf(
            frame_img, best["bbox"])
    else:
        best["lmk68"] = landmarks3d.get_landmarks_68(frame_img, best["bbox"])
    return best


def _mouth_from_det(det: dict) -> dict | None:
    """検出結果から口元楕円を作る。
    優先順位: 3DDFA_V2 68点(大ポーズ対応) → 106点 → 5点式。"""
    m = None
    if det.get("lmk68") is not None:
        m = ellipse_from_lmk68(det["lmk68"], "mouth")
    if m is None and det.get("lmk"):
        m = mouth_to_ellipse_lmk(det.get("lmk"), det["kps"])
    if m is None:
        m = mouth_to_ellipse(det["bbox"], det["kps"])
    return m


def _frontal_conf(det: dict) -> float:
    """
    検出の信頼度を 0〜1 で返す。正面（目間隔 ≈ 目〜口距離）ほど高く、
    横顔ほど低い。106点が無い場合は割り引く。幾何が苦手なフレームを
    AI 補正へ回す判定に使う。
    """
    kps = det.get("kps")
    if kps is None or len(kps) < 5:
        return 0.0
    e1, e2, _n, ml, mr = kps
    inter_eye = math.hypot(e2[0] - e1[0], e2[1] - e1[1])
    mx, my = (ml[0] + mr[0]) / 2, (ml[1] + mr[1]) / 2
    ex, ey = (e1[0] + e2[0]) / 2, (e1[1] + e2[1]) / 2
    eye_mouth = math.hypot(mx - ex, my - ey)
    if eye_mouth < 1:
        return 0.0
    frontal = min(1.0, inter_eye / eye_mouth)   # 正面 ≈ 1.0, 真横 → 0
    return frontal if det.get("lmk") else frontal * 0.7


def _measure_mouth_on(frame_img, hk: dict, margin_factor: float = 0.5):
    """1フレーム分の口元実測。テンプレートが無いときの初期値用。"""
    det = _detect_face_near(frame_img, hk, margin_factor)
    if det is None:
        return None
    return _mouth_from_det(det)


# ── Mocha式: マスク形状のテンプレート化 ──────────────────────────────
# 「トラッキング（位置・スケール・回転）」と「マスクの形」を分離する。
# 手動で決めた楕円を顔座標系のテンプレートとして記憶し、以降のフレームは
# ランドマークから顔の 位置/スケール/傾き だけを取ってテンプレートを変換する。
# 形を毎フレーム作り直さないので、ピッチ（うつむき等）でサイズが暴走しない。

def _face_frame(kps) -> tuple[tuple[float, float], float, float]:
    """ランドマーク5点 → (原点=口の中点, スケール, 傾きroll度)。"""
    e1, e2 = kps[0], kps[1]
    ml, mr = kps[3], kps[4]
    mid_eye = ((e1[0] + e2[0]) / 2, (e1[1] + e2[1]) / 2)
    mid_mouth = ((ml[0] + mr[0]) / 2, (ml[1] + mr[1]) / 2)
    inter_eye = math.hypot(e2[0] - e1[0], e2[1] - e1[1])
    eye_mouth = math.hypot(mid_mouth[0] - mid_eye[0],
                           mid_mouth[1] - mid_eye[1])
    # 目間隔（ピッチに強い）と目〜口距離（ヨーに強い）の平均 = ポーズ変動に頑健
    scale = max(4.0, (inter_eye + eye_mouth) / 2)
    roll = math.degrees(math.atan2(e2[1] - e1[1], e2[0] - e1[0]))
    return mid_mouth, scale, roll


def make_mouth_template(manual: dict, kps) -> dict:
    """手動修正した楕円を、顔座標系の相対テンプレートとして記憶する。"""
    org, sc, roll = _face_frame(kps)
    a = -math.radians(roll)
    dx, dy = manual["cx"] - org[0], manual["cy"] - org[1]
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    return {
        "du": lx / sc, "dv": ly / sc,
        "ru": manual["rx"] / sc, "rv": manual["ry"] / sc,
        "dang": manual.get("angle", 0.0) - roll,
    }


def apply_mouth_template(kps, tpl: dict) -> dict:
    """テンプレートを現在フレームの顔座標系に変換して楕円を得る。"""
    org, sc, roll = _face_frame(kps)
    a = math.radians(roll)
    lx, ly = tpl["du"] * sc, tpl["dv"] * sc
    return {
        "cx": round(org[0] + lx * math.cos(a) - ly * math.sin(a), 1),
        "cy": round(org[1] + lx * math.sin(a) + ly * math.cos(a), 1),
        "rx": round(max(4.0, tpl["ru"] * sc), 1),
        "ry": round(max(4.0, tpl["rv"] * sc), 1),
        "angle": round(roll + tpl["dang"], 1),
        "visible": True,
        "src": "auto",
    }


def _face_bbox_near(frame_img, ell: dict,
                    margin_factor: float = 2.5) -> list | None:
    """
    楕円 ell の周辺から「最寄りの顔」の bbox（元解像度）を返す。
    口元人物の伝播シードに使う: SAM 2 には模様の少ない唇パッチより
    顔全体を追わせた方が圧倒的に安定する。
    """
    h, w = frame_img.shape[:2]
    r = max(ell["rx"], ell["ry"])
    mg = int(r * margin_factor) + 8
    x1 = max(0, int(ell["cx"] - r) - mg)
    y1 = max(0, int(ell["cy"] - r) - mg)
    x2 = min(w, int(ell["cx"] + r) + mg)
    y2 = min(h, int(ell["cy"] + r) + mg)
    crop = frame_img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    cs = 1.0
    if max(crop.shape[:2]) < 240:
        cs = 240.0 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=cs, fy=cs)
    faces = get_face_app(with_rec=False).get(crop)
    best, best_d = None, None
    for f in faces:
        bb = [f.bbox[0] / cs + x1, f.bbox[1] / cs + y1,
              f.bbox[2] / cs + x1, f.bbox[3] / cs + y1]
        d = math.hypot((bb[0] + bb[2]) / 2 - ell["cx"],
                       (bb[1] + bb[3]) / 2 - ell["cy"])
        if best_d is None or d < best_d:
            best, best_d = bb, d
    if best is None or best_d > r * 2.5 + 32:
        return None
    return best


def calibrate_mouth_adjust(manual: dict, measured: dict) -> dict:
    """
    手動修正した楕円と実測楕円の差分を「相対調整値」として学習する。
    du/dv: 実測楕円ローカル座標での中心オフセット（半径比）
    sx/sy: 半径のスケール比
    """
    a = -math.radians(measured.get("angle", 0.0))
    dx = manual["cx"] - measured["cx"]
    dy = manual["cy"] - measured["cy"]
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    return {
        "du": clamp(lx / max(1.0, measured["rx"]), -1.5, 1.5),
        "dv": clamp(ly / max(1.0, measured["ry"]), -1.5, 1.5),
        "sx": clamp(manual["rx"] / max(1.0, measured["rx"]), 0.3, 4.0),
        "sy": clamp(manual["ry"] / max(1.0, measured["ry"]), 0.3, 4.0),
    }


def apply_mouth_adjust(m: dict, adj: dict | None) -> dict:
    """学習済みの相対調整値を実測楕円に適用する。"""
    if not adj:
        return m
    a = math.radians(m.get("angle", 0.0))
    lx = adj.get("du", 0.0) * m["rx"]
    ly = adj.get("dv", 0.0) * m["ry"]
    out = dict(m)
    out["cx"] = round(m["cx"] + lx * math.cos(a) - ly * math.sin(a), 1)
    out["cy"] = round(m["cy"] + lx * math.sin(a) + ly * math.cos(a), 1)
    out["rx"] = round(m["rx"] * adj.get("sx", 1.0), 1)
    out["ry"] = round(m["ry"] * adj.get("sy", 1.0), 1)
    return out


class _Identity:
    """追跡対象1人分（リファレンスモードでは常に1つ）。"""

    def __init__(self, iid: int):
        self.id = iid
        # frame -> {"bbox": [...], "kps": [...] or None}（元解像度）
        self.anchors: dict[int, dict] = {}
        self.keyframes: dict[str, dict] = {}
        # frame(str) -> 生の検出楕円（SAM2 マスクbbox）。エディタのオレンジ
        # 「検出枠」を常時表示するため、口元へ精緻化する前の頭部検出領域を残す
        self.detections: dict[str, dict] = {}
        # frame(str) -> 追跡時の失敗検知用特徴量（lmk_conf/conf/dev/gated/gate_th）。
        # 非破壊。手動修正(src=manual)を失敗ラベルとして結合するための教師データ源。
        # 設計: .company/engineering/docs/manual-correction-as-failure-labels.md
        self.track_features: dict[str, dict] = {}
        self.last_bbox: list | None = None
        self.last_frame = -10**9


# ── ジョブ本体 ───────────────────────────────────────────────────────────────

class TrackJob:
    def __init__(self, pid: str, mode: str = "all",
                 ref_images: list[str] | None = None,
                 use_autoclick: bool = True, region: str = "face",
                 low_ram: bool = False):
        self.pid = pid
        self.mode = mode if mode in ("all", "reference") else "all"
        self.ref_images = ref_images or []
        self.use_autoclick = use_autoclick
        self.region = region if region in ("face", "mouth") else "face"
        self.window = LOW_RAM_WINDOW if low_ram else WINDOW
        self.state = "running"
        self.progress = 0.0
        self.message = "初期化中..."
        self.stats = {"anchors": 0, "shots": 0, "verify_rejects": 0,
                      "autoclicks": 0, "autoclick_hits": 0,
                      "ambig_checks": 0, "ambig_rejects": 0,
                      "ai_corrections": 0, "ai_correct_hits": 0}
        self.cancel_requested = False
        self.ref_crop = None            # リファレンス顔クロップ（②/③で使用）
        self._ambig_last: dict[int, int] = {}   # oid -> 最後に質問したフレーム
        # ライブ表示用: {"frame": int, "ellipses": [{cx,cy,rx,ry,angle}]}
        self.live: dict | None = None
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start_job(self):
        self._thread.start()

    def status(self) -> dict:
        return {"state": self.state, "progress": round(self.progress, 3),
                "message": self.message, "stats": self.stats,
                "live": self.live}

    def _check_cancel(self):
        if self.cancel_requested:
            raise InterruptedError("キャンセルされました")

    def _run(self):
        try:
            self._track()
        except InterruptedError:
            self.state = "cancelled"
            self.message = "キャンセルされました"
        except Exception as e:
            traceback.print_exc()
            self.state = "error"
            self.message = f"エラー: {e}"

    # ── リファレンス読み込み ──────────────────────────────────────────
    def _load_reference(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """リファレンス画像群 → (平均埋め込み, 顔クロップ画像)。"""
        if self.mode != "reference":
            return None, None
        app = get_face_app(with_rec=True)
        embs, crop = [], None
        for path in self.ref_images:
            img = cv2.imread(str(Path(path).expanduser()))
            if img is None:
                continue
            faces = app.get(img)
            if not faces:
                continue
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            embs.append(face.normed_embedding)
            if crop is None:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                mx, my = int((x2 - x1) * 0.3), int((y2 - y1) * 0.3)
                crop = img[max(0, y1 - my):y2 + my, max(0, x1 - mx):x2 + mx].copy()
        if not embs:
            raise ValueError(
                "リファレンス画像から顔を検出できませんでした。"
                "正面に近い顔が写った画像を指定してください。")
        mean = np.mean(embs, axis=0)
        return mean / np.linalg.norm(mean), crop

    # ── Pass 1: ショット検出 + アンカー発見 ──────────────────────────
    def _scan(self, video_path: str, total: int, ref_emb):
        src = video_manager.get(video_path)
        app = get_face_app(with_rec=(self.mode == "reference"))
        cuts = [0]
        identities: dict[int, _Identity] = {}
        next_iid = 1
        prev_hist = None

        if self.mode == "reference":
            identities[1] = _Identity(1)

        for fi in range(total):
            self._check_cancel()
            frame = src.get_frame(fi)
            if frame is None:
                total = fi
                break

            h = _hist(frame)
            if prev_hist is not None:
                corr = cv2.compareHist(prev_hist, h, cv2.HISTCMP_CORREL)
                if corr < SHOT_HIST_THRESH:
                    cuts.append(fi)
            prev_hist = h

            if fi % ANCHOR_SCAN_EVERY == 0:
                faces = app.get(frame)
                if self.mode == "reference":
                    ident = identities[1]
                    best, best_sim = None, ANCHOR_SIM
                    for f in faces:
                        sim = _cos(f.normed_embedding, ref_emb)
                        if sim > best_sim:
                            best, best_sim = f, sim
                    if best is not None:
                        ident.anchors[fi] = _anchor_of(best)
                else:
                    # 全員モード: IoU で既存 identity に割り当て
                    for f in faces:
                        bbox = f.bbox.astype(float).tolist()
                        best_id, best_iou = None, 0.2
                        for ident in identities.values():
                            if fi - ident.last_frame > ANCHOR_SCAN_EVERY * 12:
                                continue
                            if ident.last_bbox is None:
                                continue
                            v = _bbox_iou(ident.last_bbox, bbox)
                            if v > best_iou:
                                best_id, best_iou = ident.id, v
                        if best_id is None:
                            ident = _Identity(next_iid)
                            identities[next_iid] = ident
                            next_iid += 1
                        else:
                            ident = identities[best_id]
                        ident.anchors[fi] = _anchor_of(f)
                        ident.last_bbox = bbox
                        ident.last_frame = fi

            if fi % ANCHOR_SCAN_EVERY == 0:
                # ライブ表示: 現時点で追跡中の顔bbox
                self.live = {
                    "frame": fi,
                    "ellipses": [
                        {"cx": (a["bbox"][0] + a["bbox"][2]) / 2,
                         "cy": (a["bbox"][1] + a["bbox"][3]) / 2,
                         "rx": (a["bbox"][2] - a["bbox"][0]) / 2,
                         "ry": (a["bbox"][3] - a["bbox"][1]) / 2,
                         "angle": 0}
                        for ident in identities.values()
                        for a in ([ident.anchors[fi]] if fi in ident.anchors else [])
                    ],
                }
            if fi % 30 == 0:
                self.progress = 0.35 * fi / max(1, total)
                self.message = f"Pass1: アンカー探索中 {fi}/{total}"

        # ノイズ identity（アンカーが少なすぎる）を除去
        identities = {k: v for k, v in identities.items()
                      if len(v.anchors) >= 2 or self.mode == "reference"}
        cuts.append(total)
        shots = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)
                 if cuts[i + 1] - cuts[i] >= 2]
        self.stats["shots"] = len(shots)
        self.stats["anchors"] = sum(len(v.anchors) for v in identities.values())
        return shots, identities, total

    # ── SAM 2: 1窓分の伝播 ────────────────────────────────────────────
    def _run_window(self, video_path: str, w_start: int, w_end: int,
                    prompts: dict[int, dict[int, dict]], scale: float,
                    ref_emb, ident_map: dict[int, _Identity]
                    ) -> dict[int, np.ndarray]:
        """
        prompts: obj_id -> {窓内フレーム番号:
                 {"box": [...]} / {"point": [x,y]} / {"mask": ndarray}}
                 （座標・マスクは縮小解像度）
        検証で棄却されたフレーム以降のキーフレームは書かない。
        返り値: obj_id -> 窓の最終フレームのマスク（次窓への持ち越し用）。
        """
        import contextlib

        import torch

        # SAM 2 はマスクメモリを bfloat16 で保存する実装のため、
        # 公式デモと同様に bf16 autocast の中で動かす（fp32のままだと
        # メモリアテンションで dtype 不一致エラーになる）
        amp = (torch.autocast("cuda", dtype=torch.bfloat16)
               if torch.cuda.is_available() else contextlib.nullcontext())

        src = video_manager.get(video_path)
        predictor = _get_sam()
        n = w_end - w_start

        tmpdir = Path(tempfile.mkdtemp(prefix="sam2_win_",
                                       dir=tempfile.gettempdir()))
        try:
            # 縮小フレームは自己検証のクロップに使う分だけ保持する
            # （全保持だと窓240で数百MBのRAMを占有する）
            keep_verify = self.mode == "reference" and ref_emb is not None
            verify_frames: dict[int, np.ndarray] = {}
            for i in range(n):
                self._check_cancel()
                frame = src.get_frame(w_start + i)
                if frame is None:
                    n = i
                    break
                small = cv2.resize(frame, None, fx=scale, fy=scale)
                if keep_verify and i % VERIFY_EVERY == 0:
                    verify_frames[i] = small
                cv2.imwrite(str(tmpdir / f"{i:05d}.jpg"), small,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
            if n <= 0:
                return {}

            results: dict[tuple[int, int], np.ndarray] = {}
            with amp:
                state = predictor.init_state(
                    video_path=str(tmpdir),
                    offload_video_to_cpu=True,
                    offload_state_to_cpu=False,
                )
                for oid, pm in prompts.items():
                    for lf, p in pm.items():
                        if lf >= n:
                            continue
                        if "mask" in p:
                            # 前窓からのマスク持ち越し。box にすると SAM 2 が
                            # 「見えている肌だけ」に再解釈して縮むことがある
                            predictor.add_new_mask(
                                state, frame_idx=lf, obj_id=oid,
                                mask=p["mask"].astype(bool))
                        elif "box" in p:
                            predictor.add_new_points_or_box(
                                state, frame_idx=lf, obj_id=oid,
                                box=np.array(p["box"], dtype=np.float32))
                        else:
                            predictor.add_new_points_or_box(
                                state, frame_idx=lf, obj_id=oid,
                                points=np.array([p["point"]], dtype=np.float32),
                                labels=np.array([1], dtype=np.int32))

                # 前後双方向に伝播して結果を統合
                for reverse in (False, True):
                    for f_idx, obj_ids, logits in predictor.propagate_in_video(
                            state, reverse=reverse):
                        masks = (logits > 0.0).cpu().numpy()
                        live_ells = []
                        for k, oid in enumerate(obj_ids):
                            key = (f_idx, int(oid))
                            m = masks[k, 0]
                            if key not in results or m.sum() > results[key].sum():
                                results[key] = m
                            bb = cv2.boundingRect(m.astype(np.uint8))
                            if bb[2] > 2 and bb[3] > 2:
                                live_ells.append({
                                    "cx": (bb[0] + bb[2] / 2) / scale,
                                    "cy": (bb[1] + bb[3] / 2) / scale,
                                    "rx": bb[2] / 2 / scale,
                                    "ry": bb[3] / 2 / scale,
                                    "angle": 0,
                                })
                        self.live = {"frame": w_start + f_idx,
                                     "ellipses": live_ells}

            predictor.reset_state(state)
            del state
            gc.collect()
            torch.cuda.empty_cache()

            # 自己検証（リファレンスモードのみ）: すり替わりフレームを特定
            rejected: dict[int, set[int]] = {}
            if self.mode == "reference" and ref_emb is not None:
                app = get_face_app(with_rec=True)

                def _reject_span(oid: int, lf: int):
                    ident = ident_map.get(oid)
                    next_anchor = min(
                        (a for a in ident.anchors if a > w_start + lf),
                        default=w_end) if ident else w_end
                    rej = rejected.setdefault(oid, set())
                    rej.update(range(lf, min(next_anchor - w_start, n)))
                    self.stats["verify_rejects"] += 1

                for (lf, oid), m in sorted(results.items()):
                    if lf % VERIFY_EVERY != 0:
                        continue
                    bb = _mask_bbox(m)
                    if bb is None:
                        continue
                    x1, y1, x2, y2 = bb
                    fimg = verify_frames.get(lf)
                    if fimg is None:
                        continue
                    mx, my = int((x2 - x1) * 0.4) + 8, int((y2 - y1) * 0.4) + 8
                    crop = fimg[max(0, y1 - my):y2 + my,
                                max(0, x1 - mx):x2 + mx]
                    if crop.size == 0:
                        continue
                    faces = app.get(crop)
                    if not faces:
                        continue  # 顔が見えない（後ろ向き等）→ 判定不能は継続
                    best = max(_cos(f.normed_embedding, ref_emb) for f in faces)
                    if best < VERIFY_REJECT_SIM:
                        # すり替わり検知（確信あり）: 次のアンカーまで棄却
                        _reject_span(oid, lf)
                    elif best < ANCHOR_SIM:
                        # 曖昧ゾーン: ArcFace では白黒つかない → Claude 二次判定
                        # （髪型・服装など顔以外の手がかりで判断できる）
                        verdict = self._ambig_check(oid, w_start + lf, crop)
                        if verdict is False:
                            _reject_span(oid, lf)
                            self.stats["ambig_rejects"] += 1

            # マスク → 楕円キーフレーム
            for (lf, oid), m in results.items():
                if lf in rejected.get(oid, ()):
                    continue
                ident = ident_map.get(oid)
                if ident is None:
                    continue
                gf = w_start + lf
                ell = _mask_to_ellipse(m, scale)
                if ell is not None and str(gf) not in ident.keyframes:
                    ident.keyframes[str(gf)] = ell
                    # 生の検出領域を保存（この後 口元に精緻化されても残す）
                    ident.detections[str(gf)] = dict(ell)

            # 次窓への持ち越し用: 最終フレームのマスク
            carry_masks: dict[int, np.ndarray] = {}
            for oid in prompts:
                m = results.get((n - 1, oid))
                if (m is not None and int(m.sum()) >= MIN_MASK_AREA
                        and (n - 1) not in rejected.get(oid, ())):
                    carry_masks[oid] = m
            return carry_masks
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    # ── 曖昧ゾーンの Claude 二次判定 ──────────────────────────────────
    def _ambig_check(self, oid: int, frame: int, crop) -> bool | None:
        """
        ArcFace 類似度が曖昧（0.18〜0.40）なケースを Claude に仲裁させる。
        クールダウン + 回数上限でコストを制御。判定不能は None（追跡続行）。
        """
        if (not self.use_autoclick or self.ref_crop is None
                or not autoclick.available()):
            return None
        if self.stats["ambig_checks"] >= AMBIG_MAX_CALLS:
            return None
        last = self._ambig_last.get(oid, -10**9)
        if frame - last < AMBIG_COOLDOWN:
            return None
        self._ambig_last[oid] = frame
        self.stats["ambig_checks"] += 1
        self.message = f"曖昧照合を Claude で二次判定中 frame {frame}"
        try:
            return autoclick.same_person(crop, self.ref_crop, pid=self.pid)
        except Exception:
            return None  # APIエラーは判定保留（追跡は続行）

    # ── 口元の実測リファイン ──────────────────────────────────────────
    def _refine_mouth(self, video_path: str, ident: _Identity,
                      margin_factor: float = 0.5,
                      adjust: dict | None = None,
                      template: dict | None = None
                      ) -> tuple[dict[int, dict], dict[int, float]]:
        """
        追跡済みキーフレームの領域を毎フレームランドマーク検出し、
        口元楕円を返す（検出できたフレームのみ）。

        template: 手動修正から学習した形状テンプレート（Mocha式）。
        あればテンプレートを顔の位置/スケール/傾きに載せて配置する（形は不変）。
        無ければ式ベースの実測（+ 旧 adjust 互換）。

        戻り値: (refined, conf)。conf[fi] は 0〜1 の信頼度
        （正面ほど高い。横顔・106点無しは低い）。未測定フレームは conf 未登録。
        """
        src = video_manager.get(video_path)
        refined: dict[int, dict] = {}
        conf: dict[int, float] = {}
        lmk_conf: dict[int, float] = {}  # fi -> 3DDFAフィットの怪しさ（小=良）
        pos: dict[int, tuple] = {}      # fi -> 口中心（Pass2の位置予測に使う）
        frames = sorted(int(k) for k in ident.keyframes)
        vis = [fi for fi in frames
               if ident.keyframes[str(fi)].get("visible", True)]

        # ── Pass 1: 頭部キーフレーム周辺のローカル探索 ──
        for i, fi in enumerate(vis):
            self._check_cancel()
            frame = src.get_frame(fi)
            if frame is None:
                continue
            det = _detect_face_near(frame, ident.keyframes[str(fi)],
                                    margin_factor, with_conf=True)
            m = self._mouth_from_det_or_tpl(det, template, adjust) if det else None
            if m is not None:
                refined[fi] = m
                conf[fi] = _frontal_conf(det)
                # 既定は「測っていない」。fit を試して破綻した場合は
                # get_landmarks_68_conf 側が CONF_FAILED を入れてくる
                lmk_conf[fi] = det.get("lmk_conf", landmarks3d.CONF_UNAVAILABLE)
                pos[fi] = (m["cx"], m["cy"])
            if i % 60 == 0:
                self.message = f"口元を実測中... {i}/{len(vis)}"
                self.live = {"frame": fi,
                             "ellipses": [refined[fi]] if fi in refined else []}

        # ── Pass 2: ローカル失敗フレームを全画面検出で回収 ──
        # 頭部追跡が背景へドリフトした区間は周辺探索では顔を見つけられない。
        # 前後の良フレームから口位置を予測し、全画面検出のうち最も近い顔を採用する。
        good = sorted(pos)
        missing = [fi for fi in vis if fi not in refined]
        if missing:
            app = get_face_app(with_rec=False)
            h, w = None, None
            for j, fi in enumerate(missing):
                self._check_cancel()
                frame = src.get_frame(fi)
                if frame is None:
                    continue
                if h is None:
                    h, w = frame.shape[:2]
                predict = self._predict_pos(good, pos, fi)
                det = self._detect_face_fullframe(frame, app, predict,
                                                  max(h, w) * 0.5)
                m = self._mouth_from_det_or_tpl(det, template, adjust) if det else None
                if m is not None:
                    refined[fi] = m
                    conf[fi] = _frontal_conf(det) * 0.9
                    lmk_conf[fi] = det.get("lmk_conf",
                                           landmarks3d.CONF_UNAVAILABLE)
                if j % 30 == 0:
                    self.message = f"口元を回収中... {j}/{len(missing)}"
                    self.live = {"frame": fi,
                                 "ellipses": [refined[fi]] if fi in refined else []}

        # ── Pass 3: 口位置予測が外れて Pass2 も落ちたフレームを、信頼できる
        # SAM2 頭部トラックの位置を頼りに回収する。全画面検出のうち頭部KF中心に
        # 最も近い顔を採用（顔さえ映っていれば拾える）。相対補間フォールバック前の
        # 最後の砦。上向き等で「顔は検出できるのに測れず誤配置」を防ぐ。
        still = [fi for fi in vis if fi not in refined]
        if still:
            app3 = get_face_app(with_rec=False)
            for j, fi in enumerate(still):
                self._check_cancel()
                frame = src.get_frame(fi)
                if frame is None:
                    continue
                hk = ident.keyframes[str(fi)]
                cands = [f for f in app3.get(frame) if f.kps is not None]
                if not cands:
                    continue
                hcx, hcy = hk["cx"], hk["cy"]
                hr = max(1.0, hk.get("rx", 1.0), hk.get("ry", 1.0))

                def _bc(f):
                    b = f.bbox
                    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2

                face = min(cands, key=lambda f: math.hypot(
                    _bc(f)[0] - hcx, _bc(f)[1] - hcy))
                bcx, bcy = _bc(face)
                if math.hypot(bcx - hcx, bcy - hcy) > hr * 1.6:
                    continue  # 頭部から遠すぎる＝別人/誤検出
                bbox = face.bbox.astype(float).tolist()
                lm106 = getattr(face, "landmark_2d_106", None)
                l68, lc = landmarks3d.get_landmarks_68_conf(frame, bbox)
                det = {
                    "bbox": bbox,
                    "kps": face.kps.astype(float).tolist(),
                    "lmk": lm106.astype(float).tolist() if lm106 is not None else None,
                    "lmk68": l68,
                    "lmk_conf": lc,
                }
                m = self._mouth_from_det_or_tpl(det, template, adjust)
                if m is not None:
                    refined[fi] = m
                    conf[fi] = _frontal_conf(det) * 0.85
                    lmk_conf[fi] = det["lmk_conf"]
                if j % 30 == 0:
                    self.message = f"口元を再回収中... {j}/{len(still)}"
                    self.live = {"frame": fi,
                                 "ellipses": [refined[fi]] if fi in refined else []}

        # ── Pass 4: SCRFD が顔を1つも返さなかったフレームの救済 ──
        # ブラー＋あご上げ＋大きな面内回転で検出が全滅する。位置が既知なので
        # 局所を拡大・回転して低しきい値で拾い直す（検出率 68%→96%）。
        # ここで拾えるのは 3DDFA が壊れやすいフレームそのものなので、
        # 直後の信頼度ゲートが崩壊を弾くことを前提に有効化している
        rescue = [fi for fi in vis if fi not in refined]
        if rescue:
            for j, fi in enumerate(rescue):
                self._check_cancel()
                frame = src.get_frame(fi)
                if frame is None:
                    continue
                hk = ident.keyframes[str(fi)]
                det = detect_face_robust(frame, hk["cx"], hk["cy"],
                                         max(hk["rx"], hk["ry"]))
                m = self._mouth_from_det_or_tpl(det, template, adjust) if det else None
                if m is not None:
                    refined[fi] = m
                    conf[fi] = _frontal_conf(det) * 0.8
                    lmk_conf[fi] = det["lmk_conf"]
                    self.stats["mouth_rescued"] = (
                        self.stats.get("mouth_rescued", 0) + 1)
                if j % 30 == 0:
                    self.message = f"口元を救済中... {j}/{len(rescue)}"

        # ── 信頼度ゲート: 破綻したフィットを捨てる ──
        # 捨てたフレームは refined に残さない → convert_ident_to_mouth が
        # 頭部楕円基準の相対補間で埋める（頑健な頭部トラックに位置を預ける）。
        # テンプレート運用時は口元が 5点式から作られ 68点を使わないので対象外

        # ── 失敗検知用の特徴量を非破壊で記録（手動修正ラベルとの結合用）──
        # ゲートより前に組む（棄却されるフレームも gated=True で残すため）。
        # 設計: .company/engineering/docs/manual-correction-as-failure-labels.md
        feats: dict[str, dict] = {}
        for fi in refined:
            feats[str(fi)] = {
                "lmk_conf": round(float(lmk_conf.get(
                    fi, landmarks3d.CONF_UNAVAILABLE)), 4),
                "conf": round(float(conf.get(fi, 0.0)), 4),
                "gated": False,
            }

        # 3DDFA が死んだまま走り切っていないか記録する。lmk_conf が全滅した
        # ときそれが「素材が難しい」のか「モデルが動いていない」のかを後から
        # 判別できないと、分析が壊れる（2026-07-16 の負の結果がこれで汚染された）
        st3d = landmarks3d.status()
        self.stats["lmk3d_loaded"] = st3d["loaded"]
        if not st3d["loaded"]:
            self.stats["lmk3d_error"] = st3d["last_error"]
        n_unavail = sum(1 for v in feats.values()
                        if v["lmk_conf"] >= landmarks3d.CONF_UNAVAILABLE)
        self.stats["lmk3d_unmeasured"] = n_unavail
        if feats and n_unavail == len(feats):
            self.message = ("⚠ 3DDFA の68点が1フレームも取得できませんでした"
                            "（106点/5点フォールバックで配置）")

        if template is None:
            self._gate_mouth(refined, conf, lmk_conf, feats=feats)
        ident.track_features = feats

        return refined, conf

    def _gate_mouth(self, refined: dict, conf: dict,
                    lmk_conf: dict[int, float],
                    feats: dict | None = None) -> None:
        """3DDFA のフィット破綻フレームを refined から除く。

        判定は **2条件のAND**:
          (1) jitter 一致性が悪い（フィットが摂動に対して暴れている）
          (2) その楕円が、信用できる近傍フレームの補間と大きく食い違う

        (1) だけで捨てると「怪しいが実は当たっている」フレームまで落ちて、
        補間で置き換えたぶん逆に悪化する（実測: 良好区間で IoU -0.023）。
        棄却は「捨てて補間したほうがマシ」なときだけ意味がある。

        しきい値(1)は絶対値で決め打ちしない。素材ごとに分布が動くため
        （hy_clip1 の中央値 0.0115 に対し hy_clip2 は 0.0276）、トラック内の
        中央値を基準に相対で決める。
        検証: .company/engineering/docs/mouth-mask-accuracy-hy.md
        """
        vals = [c for c in lmk_conf.values() if c < landmarks3d.CONF_FAILED]
        if len(vals) < MOUTH_GATE_MIN_SAMPLES:
            return
        med = statistics.median(vals)
        th = min(MOUTH_GATE_CEIL, max(MOUTH_GATE_FLOOR, MOUTH_GATE_K * med))
        # トラックの中央値は失敗検知の分析で使うので記録しておく
        # （ゲートの有効/無効判定には使わない。上の ⚠ 参照）
        self.stats["mouth_lmk_median"] = round(med, 4)

        # (1) 怪しいフレーム。信用できるフレームを補間の足場にする
        suspect = sorted(fi for fi in refined if lmk_conf.get(fi, 0.0) > th)
        trusted = sorted(fi for fi in refined if fi not in set(suspect))
        if not trusted:
            self.stats["mouth_gated"] = 0
            return

        drop = []
        suspect_set = set(suspect)
        # 要確認フラグ用のしきい値（棄却とは別。MOUTH_REVIEW_* 参照）
        th_rev = min(MOUTH_REVIEW_CEIL,
                     max(MOUTH_REVIEW_FLOOR, MOUTH_REVIEW_K * med))
        n_review = 0
        # dev（信用フレーム補間からの食い違い）は全フレームに付ける＝密な失敗
        # 検知特徴量。trusted フレームは自己補間で dev≈0、怪しいフレームほど大。
        # 棄却は従来どおり「怪しい(suspect) かつ dev 大」のみ。
        for fi in refined:
            ref = self._interp_from(trusted, refined, fi)
            if ref is None:
                continue
            m = refined[fi]
            R = max(1.0, (ref["rx"] + ref["ry"]) / 2)
            dev = math.hypot(m["cx"] - ref["cx"], m["cy"] - ref["cy"]) / R
            lc = lmk_conf.get(fi, 0.0)
            if feats is not None and str(fi) in feats:
                feats[str(fi)]["dev"] = round(dev, 4)
            # (2) 近傍と整合していれば、怪しくても実測を残す
            if fi in suspect_set and dev > MOUTH_GATE_DEVIATION:
                drop.append(fi)
            # 要確認フラグ: 棄却より緩い。配置には一切影響せず、人に見せるだけ
            if lc > th_rev and dev > MOUTH_REVIEW_DEVIATION:
                n_review += 1
                if feats is not None and str(fi) in feats:
                    feats[str(fi)]["review"] = True

        # 全滅させない（補間の足場が無くなる）
        min_keep = int(len(refined) * MOUTH_GATE_MIN_KEEP)
        if len(refined) - len(drop) < min_keep:
            drop = sorted(drop, key=lambda f: -lmk_conf[f])[
                :max(0, len(refined) - min_keep)]

        for fi in drop:
            refined.pop(fi, None)
            # AI 範囲補正が優先的に見に行くよう信頼度も落としておく
            conf[fi] = min(conf.get(fi, 1.0), 0.15)
            if feats is not None and str(fi) in feats:
                feats[str(fi)]["gated"] = True
        if feats is not None:
            th_r = round(th, 4)
            for f in feats.values():
                f["gate_th"] = th_r
        self.stats["mouth_gated"] = len(drop)
        self.stats["mouth_suspect"] = len(suspect)
        self.stats["mouth_gate_th"] = round(th, 4)
        self.stats["mouth_review"] = n_review
        self.stats["mouth_review_th"] = round(th_rev, 4)

    @staticmethod
    def _interp_from(keys: list[int], vals: dict[int, dict],
                     fi: int) -> dict | None:
        """keys（昇順）の前後フレームから fi の楕円を線形補間する。"""
        if not keys:
            return None
        j = bisect.bisect_left(keys, fi)
        prev = keys[j - 1] if j > 0 else None
        nxt = keys[j] if j < len(keys) else None
        if prev is None and nxt is None:
            return None
        if prev is None:
            return vals[nxt]
        if nxt is None:
            return vals[prev]
        t = (fi - prev) / max(1, nxt - prev)
        a, b = vals[prev], vals[nxt]
        out = {k: a[k] + (b[k] - a[k]) * t for k in ("cx", "cy", "rx", "ry")}
        out["angle"] = _lerp_angle(a.get("angle", 0.0), b.get("angle", 0.0), t)
        out["visible"] = True
        out["src"] = "auto"
        return out

    @staticmethod
    def _mouth_from_det_or_tpl(det, template, adjust):
        """検出結果から口元楕円（テンプレートあれば配置、無ければ実測+調整）。"""
        if det is None:
            return None
        if template:
            return apply_mouth_template(det["kps"], template)
        m = _mouth_from_det(det)
        return apply_mouth_adjust(m, adjust) if m is not None else None

    @staticmethod
    def _predict_pos(good: list, pos: dict, fi: int):
        """前後の良フレームの口中心を線形補間して fi の予測位置を返す。"""
        j = bisect.bisect_left(good, fi)
        prev = good[j - 1] if j > 0 else None
        nxt = good[j] if j < len(good) else None
        if prev is not None and nxt is not None:
            t = (fi - prev) / max(1, nxt - prev)
            (px, py), (nx, ny) = pos[prev], pos[nxt]
            return (px + (nx - px) * t, py + (ny - py) * t)
        if prev is not None:
            return pos[prev]
        if nxt is not None:
            return pos[nxt]
        return None

    @staticmethod
    def _detect_face_fullframe(frame, app, predict, max_dist: float):
        """全画面検出。predict に最も近い口の顔を選ぶ（遠すぎれば不採用）。"""
        faces = app.get(frame)
        if not faces:
            return None

        def mouthc(f):
            k = f.kps
            return ((k[3][0] + k[4][0]) / 2, (k[3][1] + k[4][1]) / 2)

        cands = [f for f in faces if f.kps is not None]
        if not cands:
            return None
        if predict is not None:
            face = min(cands, key=lambda f: math.hypot(
                mouthc(f)[0] - predict[0], mouthc(f)[1] - predict[1]))
            mx, my = mouthc(face)
            if math.hypot(mx - predict[0], my - predict[1]) > max_dist:
                return None            # 予測から遠すぎる＝別人/誤検出
        else:
            face = max(cands, key=lambda f: (f.bbox[2] - f.bbox[0])
                       * (f.bbox[3] - f.bbox[1]))
        bbox = face.bbox.astype(float).tolist()
        lm106 = getattr(face, "landmark_2d_106", None)
        l68, lconf = landmarks3d.get_landmarks_68_conf(frame, bbox)
        return {
            "bbox": bbox,
            "kps": face.kps.astype(float).tolist(),
            "lmk": lm106.astype(float).tolist() if lm106 is not None else None,
            "lmk68": l68,
            "lmk_conf": lconf,
        }

    def _expand_face_coverage(self, video_path: str,
                              ident: _Identity) -> dict[int, list]:
        """
        アンカーを持たない経路（再シード等）用の顔カバー検証。
        ANCHOR_SCAN_EVERY 間隔で追跡楕円の周辺の顔を実測し、その bbox を
        サイズ下限（+暴走上限）として ident.keyframes を安全側に補正する。
        実測した {フレーム: bbox} を返す（遮蔽ブリッジの期待サイズに使う）。
        """
        src = video_manager.get(video_path)
        frames = sorted(int(f) for f in ident.keyframes)
        bbs: dict[int, list] = {}
        for i, fi in enumerate(frames):
            if fi % ANCHOR_SCAN_EVERY != 0:
                continue
            self._check_cancel()
            kf = ident.keyframes[str(fi)]
            if not kf.get("visible", True):
                continue
            img = src.get_frame(fi)
            if img is None:
                continue
            det = _detect_face_near(img, kf, 0.8)
            if det is not None:
                bbs[fi] = det["bbox"]
            if i % 60 == 0:
                self.message = f"顔全体カバーを検証中... {i}/{len(frames)}"
        fixes = blowups = 0
        for fi in frames:
            kf = ident.keyframes[str(fi)]
            if not kf.get("visible", True):
                continue
            bb = _nearest_bbox(bbs, fi, ANCHOR_SCAN_EVERY)
            if bb is None:
                continue
            if _shrink_to_face_bbox(kf, bb) is not kf:
                blowups += 1
            fitted = _fit_face_ellipse(kf, bb)
            if fitted is not kf:
                ident.keyframes[str(fi)] = fitted
                fixes += 1
        self.stats["coverage_fixes"] = (
            self.stats.get("coverage_fixes", 0) + fixes)
        self.stats["blowup_fixes"] = (
            self.stats.get("blowup_fixes", 0) + blowups)
        return bbs

    # ── 遮蔽ブリッジ ──────────────────────────────────────────────────
    def _verify_endpoint(self, video_path: str, f: int, kf: dict,
                         blur_cfg: dict | None) -> bool | None:
        """
        端点候補をブラー合成して Sonnet に検品させる。
        True=漏れなし（信頼できる端点）/ False=漏れあり / None=検証不能。
        """
        if not autoclick.available():
            return None
        if self.stats.get("bridge_calls", 0) >= BRIDGE_MAX_CALLS:
            return None
        img = video_manager.get(video_path).get_frame(f)
        if img is None:
            return None
        persons = [{"id": "_v", "enabled": True,
                    "blur": blur_cfg or prj.default_blur()}]
        rendered = apply_blurs(img, persons, {"_v": {str(f): kf}}, f)
        self.stats["bridge_calls"] = self.stats.get("bridge_calls", 0) + 1
        leak = autoclick.check_leak(rendered, self.ref_crop, "face", self.pid)
        return leak is None

    def _bridge_occlusions(self, video_path: str, ident: _Identity,
                           blur_cfg: dict | None = None,
                           expected: dict[int, float] | None = None):
        """
        遮蔽（手・髪など）でマスクが壊れた区間を、前後の「良い端点」の
        線形補間で埋める。端点はローカル判定（面積が期待サイズ近傍）で選び、
        API があれば Sonnet の漏れ検品で確認する（安全側: 血の通った端点
        だけを信じ、ダメなら外側へずらして再検証）。
        ブリッジできなかった壊れ区間には suspect フラグを付け、
        エディターのタイムラインで要注意表示する。
        """
        kfs = ident.keyframes
        spans = _find_occlusion_spans(kfs, expected)
        if not spans:
            return

        def endpoint(edge: int, step: int, med: float) -> int | None:
            """edge から step 方向に「良い端点」を探す。"""
            tries = 0
            for off in range(1, BRIDGE_SEARCH + 1):
                f = edge + step * off
                kf = kfs.get(str(f))
                a = _kf_area(kf)
                if a is None or kf.get("bridged"):
                    continue
                if a < med * BRIDGE_OK_RATIO:
                    continue
                if tries < BRIDGE_VERIFY_TRIES:
                    ok = self._verify_endpoint(video_path, f, kf, blur_cfg)
                    if ok is False:  # Sonnet が漏れを検出 → さらに外側へ
                        tries += 1
                        continue
                return f
            return None

        def mark_suspect(s: int, e: int):
            for f in range(s, e + 1):
                kf = kfs.get(str(f))
                if kf is not None and kf.get("src") != "manual":
                    kf["suspect"] = True
                    suspects[0] += 1

        suspects = [0]
        bridged = 0
        for s, e, med in spans:
            self._check_cancel()
            if e - s > BRIDGE_MAX_SPAN:
                mark_suspect(s, e)
                continue
            self.message = f"遮蔽ブリッジ検証中 frame {s}-{e}"
            f0 = endpoint(s, -1, med)
            f1 = endpoint(e, +1, med)
            if f0 is None or f1 is None:
                mark_suspect(s, e)
                continue
            a, b = kfs[str(f0)], kfs[str(f1)]
            for f in range(f0 + 1, f1):
                old = kfs.get(str(f))
                if old is not None and old.get("src") == "manual":
                    continue
                t = (f - f0) / (f1 - f0)
                kfs[str(f)] = {
                    "cx": round(a["cx"] + (b["cx"] - a["cx"]) * t, 1),
                    "cy": round(a["cy"] + (b["cy"] - a["cy"]) * t, 1),
                    "rx": round(a["rx"] + (b["rx"] - a["rx"]) * t, 1),
                    "ry": round(a["ry"] + (b["ry"] - a["ry"]) * t, 1),
                    "angle": round(_lerp_angle(a.get("angle", 0.0),
                                               b.get("angle", 0.0), t), 1),
                    "visible": True,
                    "src": "auto",
                    "bridged": True,
                }
            bridged += 1
        self.stats["bridges"] = self.stats.get("bridges", 0) + bridged
        self.stats["suspects"] = (
            self.stats.get("suspects", 0) + suspects[0])

    # ── メインフロー ──────────────────────────────────────────────────
    def _track(self):
        proj = prj.load_project(self.pid)
        video = proj["video"]
        video_path = video["path"]
        total = video["frame_count"]
        scale = min(1.0, SAM_MAX_SIDE / max(video["width"], video["height"]))

        self.message = "リファレンス読み込み中..."
        ref_emb, ref_crop = self._load_reference()
        self.ref_crop = ref_crop

        shots, identities, total = self._scan(video_path, total, ref_emb)
        ident_map = {i.id: i for i in identities.values()}
        if not self.stats["anchors"]:
            raise ValueError(
                "アンカーを1つも発見できませんでした。"
                "対象人物が動画に映っているか、リファレンス画像を確認してください。")

        # ── Pass 2: ショットごと・窓ごとに SAM 2 伝播 ────────────────
        done_frames = 0
        for si, (s0, s1) in enumerate(shots):
            self._check_cancel()
            w = s0
            # 窓またぎのシード（obj -> {"mask": ...} または {"box": ...}）
            carry: dict[int, dict] = {}
            while w < s1:
                w_end = min(w + self.window, s1)
                self.message = (f"Pass2: SAM 2 伝播 ショット{si + 1}/{len(shots)} "
                                f"frame {w}-{w_end}")

                prompts: dict[int, dict[int, dict]] = {}
                for ident in identities.values():
                    pm = {}
                    # 窓内アンカー（信頼できる順に間引いて最大数まで）
                    in_win = sorted(a for a in ident.anchors
                                    if w <= a < w_end)
                    step = max(1, len(in_win) // MAX_PROMPTS_PER_WIN)
                    for a in in_win[::step][:MAX_PROMPTS_PER_WIN]:
                        bb = ident.anchors[a]["bbox"]
                        pm[a - w] = {"box": [v * scale for v in bb]}
                    # 前窓からの持ち越しシード
                    if ident.id in carry and 0 not in pm:
                        pm[0] = carry[ident.id]
                    if pm:
                        prompts[ident.id] = pm

                carry_masks: dict[int, np.ndarray] = {}
                if prompts:
                    carry_masks = self._run_window(
                        video_path, w, w_end, prompts, scale,
                        ref_emb, ident_map) or {}

                # 次窓への持ち越しを計算（マスク優先。box は SAM 2 が
                # 「見えている肌だけ」に再解釈して縮むことがあるため予備）
                carry = {}
                for ident in identities.values():
                    kf = ident.keyframes.get(str(w_end - 1))
                    if not kf or not kf.get("visible"):
                        continue
                    m = carry_masks.get(ident.id)
                    if m is not None:
                        # 髪・体を取り込んだ膨張を持ち越さないよう、
                        # 近くにアンカーがあれば顔サイズに刈り込む
                        bb = _nearest_bbox(
                            {f: a["bbox"] for f, a in ident.anchors.items()},
                            w_end - 1, ANCHOR_SCAN_EVERY * 2)
                        if bb is not None:
                            m = _clip_mask_to_bbox(m, bb, scale)
                    if m is not None and int(m.sum()) >= MIN_MASK_AREA:
                        carry[ident.id] = {"mask": m}
                    else:
                        carry[ident.id] = {"box": [
                            (kf["cx"] - kf["rx"]) * scale,
                            (kf["cy"] - kf["ry"]) * scale,
                            (kf["cx"] + kf["rx"]) * scale,
                            (kf["cy"] + kf["ry"]) * scale,
                        ]}
                done_frames += w_end - w
                self.progress = 0.35 + 0.5 * done_frames / max(1, total)
                w = w_end

        # ── Pass 3: Claude 自動クリックで穴埋め ──────────────────────
        if (self.use_autoclick and self.mode == "reference"
                and autoclick.available()):
            self._autoclick_pass(video_path, shots, identities[1],
                                 ref_emb, ref_crop, scale)
        elif self.use_autoclick and self.mode == "reference":
            self.message = "自動クリックはスキップ（ANTHROPIC_API_KEY 未設定）"

        # ── 顔全体モード: 楕円がアンカー（顔検出bbox）を覆うよう安全側に拡大 ──
        if self.region == "face":
            self.message = "顔全体カバーを検証中..."
            fixes = blowups = 0
            src_cov = video_manager.get(video_path)
            for ident in identities.values():
                bbs = {f: a["bbox"] for f, a in ident.anchors.items()}
                # アンカーが無い区間（うつむき等で照合スコアが届かない）は
                # 追跡楕円の周辺を実測してサイズ基準を補う。アンカー優先なので
                # 別人に引っ張られるリスクは楕円のすぐ近くに限られる
                cov_frames = sorted(int(fs) for fs in ident.keyframes)
                for i, fi in enumerate(cov_frames):
                    if fi % ANCHOR_SCAN_EVERY != 0:
                        continue
                    if _nearest_bbox(bbs, fi, ANCHOR_SCAN_EVERY) is not None:
                        continue
                    self._check_cancel()
                    kf = ident.keyframes[str(fi)]
                    if not kf.get("visible", True):
                        continue
                    img = src_cov.get_frame(fi)
                    if img is None:
                        continue
                    det = _detect_face_near(img, kf, 0.8)
                    if det is not None:
                        bbs[fi] = det["bbox"]
                    if i % 600 == 0:
                        self.message = f"顔全体カバーを検証中... {fi}"
                for fs, kf in ident.keyframes.items():
                    if not kf.get("visible", True):
                        continue
                    bb = _nearest_bbox(bbs, int(fs), ANCHOR_SCAN_EVERY)
                    if bb is None:
                        continue
                    if _shrink_to_face_bbox(kf, bb) is not kf:
                        blowups += 1
                    fitted = _fit_face_ellipse(kf, bb)
                    if fitted is not kf:
                        ident.keyframes[fs] = fitted
                        fixes += 1
            self.stats["coverage_fixes"] = fixes
            self.stats["blowup_fixes"] = blowups

            # 遮蔽（手・髪）で壊れた区間を良い端点間の補間でブリッジ
            for ident in identities.values():
                expected = {f: (a["bbox"][2] - a["bbox"][0])
                            * (a["bbox"][3] - a["bbox"][1]) / 4 * 1.15
                            for f, a in ident.anchors.items()}
                self._bridge_occlusions(video_path, ident, expected=expected)

            # 顔の AI 範囲補正は無効（AI_CORRECT_FACE 参照）。
            # Claude の矩形は系統的に小さくクランプ下限に張り付き、その平行移動
            # デルタがスパン全体に配られてマスクを頭から引き剥がしていた。
            if AI_CORRECT_FACE:
                for ident in identities.values():
                    fconf = {int(fs): 1.0 for fs in ident.keyframes}
                    for f in ident.anchors:
                        fconf[f] = 1.0
                    for fs in ident.keyframes:
                        fi = int(fs)
                        if _nearest_bbox({f: a["bbox"] for f, a in
                                          ident.anchors.items()}, fi,
                                         ANCHOR_SCAN_EVERY) is None:
                            fconf[fi] = 0.2   # アンカー不在 = 低信頼
                    self._ai_correct_region(video_path, ident, "face",
                                            conf=fconf)

        # ── 口元モードなら頭部楕円を口元楕円へ変換 ────────────────────
        if self.region == "mouth":
            # SAM2 がロストした区間を、口元へ変換する**前**に頭部トラックの
            # 段階で埋める。
            #
            # ⚠ これが無いと `_refine_mouth` も `convert_ident_to_mouth` も
            # 「そのフレームは存在しない」ものとして扱い、**マスクが1枚も
            # 出ない＝ブラーが抜ける**。相対補間は既存キーフレームの間しか
            # 埋めないので救えない。
            # 実測: a2_clip1 の f100-113（14フレーム/0.47秒）で detections も
            # keyframes も欠損し、f110 は顔が正面を向いて口が開いているのに
            # マスクが無かった（2026-07-17）。顔モードは以前からブリッジして
            # おり同じ区間が bridged で埋まっていた＝**口元モードだけの穴**。
            # ブリッジ後は顔が見えるフレームで口元を実測できる点でも有利。
            self.message = "遮蔽区間をブリッジ中..."
            for ident in identities.values():
                expected = {f: (a["bbox"][2] - a["bbox"][0])
                            * (a["bbox"][3] - a["bbox"][1]) / 4 * 1.15
                            for f, a in ident.anchors.items()}
                self._bridge_occlusions(video_path, ident, expected=expected)

            self.message = "口元を実測中..."
            _tg = prj.load_project(self.pid).get("targets") or {}
            mouth_adjust = _tg.get("mouth_adjust")
            mouth_template = _tg.get("mouth_template")
            for ident in identities.values():
                # 各フレームの頭部領域からランドマークを検出し、
                # テンプレート（手動学習済みの形）があればそれを配置、
                # なければ式ベースの実測。検出不能フレームは相対補間で埋める
                refined, conf = self._refine_mouth(video_path, ident,
                                                   adjust=mouth_adjust,
                                                   template=mouth_template)
                convert_ident_to_mouth(ident, refined)
                # 要確認フラグをキーフレームへ渡す（エディタのタイムラインが
                # keyframes の review を読んで帯表示する）。**配置は変えない**。
                # 手動フレームは人が既に直したので対象外。
                for fs, feat in (ident.track_features or {}).items():
                    if not feat.get("review"):
                        continue
                    kf = ident.keyframes.get(fs)
                    if kf is not None and kf.get("src") != "manual":
                        kf["review"] = True
                # 口元の AI 範囲補正は無効（AI_CORRECT_MOUTH 参照）。
                # Claude の矩形は幾何より悪く、しかも小さいためマスクを縦に潰し、
                # 顎先を切り落としていた
                if AI_CORRECT_MOUTH:
                    self._ai_correct_region(video_path, ident, "mouth",
                                            conf=conf, measured=set(refined))

        # ── 保存 ──────────────────────────────────────────────────────
        self.message = "結果を保存中..."
        proj = prj.load_project(self.pid)

        proj.setdefault("detections", {})
        proj.setdefault("track_features", {})
        keep = []
        for p in proj["persons"]:
            kfs = proj["keyframes"].get(p["id"], {})
            has_manual = any(kf.get("src") == "manual" for kf in kfs.values())
            if has_manual or not kfs:
                keep.append(p)
            else:
                proj["keyframes"].pop(p["id"], None)
                proj["detections"].pop(p["id"], None)
                proj["track_features"].pop(p["id"], None)
        proj["persons"] = keep

        added = 0
        for ident in identities.values():
            if len(ident.keyframes) < 3:
                continue
            person = prj.new_person(proj["persons"])
            person["region"] = self.region
            person["engine"] = "sam2"
            if self.mode == "reference":
                person["label"] = "ターゲット"
            if self.region == "mouth":
                person["label"] = (person["label"].replace("人物", "口元")
                                   if "人物" in person["label"]
                                   else person["label"] + "（口元）")
                _tg2 = proj.get("targets") or {}
                if _tg2.get("mouth_adjust"):
                    person["mouth_adjust"] = _tg2["mouth_adjust"]
                if _tg2.get("mouth_template"):
                    person["mouth_template"] = _tg2["mouth_template"]
            proj["persons"].append(person)
            proj["keyframes"][person["id"]] = ident.keyframes
            proj["detections"][person["id"]] = ident.detections
            proj["track_features"][person["id"]] = ident.track_features
            added += 1

        prj.save_project(proj)
        self.progress = 1.0
        self.state = "done"
        st = self.stats
        self.message = (
            f"完了: {added}人物 / ショット{st['shots']} / アンカー{st['anchors']} / "
            f"すり替わり棄却{st['verify_rejects']} / "
            f"曖昧判定{st['ambig_checks']}回(棄却{st['ambig_rejects']}) / "
            f"自動クリック{st['autoclick_hits']}/{st['autoclicks']} / "
            f"AI範囲補正{st.get('ai_correct_hits', 0)}/"
            f"{st.get('ai_corrections', 0)} / "
            f"顔カバー補正{st.get('coverage_fixes', 0)} / "
            f"遮蔽ブリッジ{st.get('bridges', 0)}"
            f"(検証{st.get('bridge_calls', 0)}回)")

    # ── Claude 自動クリック ───────────────────────────────────────────
    def _autoclick_pass(self, video_path: str, shots, ident: _Identity,
                        ref_emb, ref_crop, scale: float):
        src = video_manager.get(video_path)
        calls = 0
        for s0, s1 in shots:
            if calls >= AUTOCLICK_MAX_CALLS:
                break
            # ショット内でターゲットが確認されていなければ探さない
            # （不在ショットに Claude を呼ぶだけ無駄なため）
            has_any = any(s0 <= int(f) < s1 for f in ident.keyframes)
            if not has_any:
                continue

            for g0, g1 in self._gaps(ident, s0, s1):
                if calls >= AUTOCLICK_MAX_CALLS:
                    break
                mid = (g0 + g1) // 2
                frame = src.get_frame(mid)
                if frame is None:
                    continue
                self.message = f"Pass3: Claude 自動クリック frame {mid}"
                calls += 1
                self.stats["autoclicks"] = calls
                try:
                    hit = autoclick.find_head(frame, ref_crop, pid=self.pid)
                except Exception as e:
                    # 認証・残高などのAPIエラーは以降も失敗するため打ち切る
                    self.stats["autoclick_error"] = str(e)[:200]
                    self.message = f"自動クリックを中断（トラッキングは続行）: {e}"
                    return
                if hit is None:
                    continue
                self.stats["autoclick_hits"] += 1
                # 穴区間だけを SAM 2 で再伝播（点プロンプト）
                w0, w1 = max(s0, g0 - 5), min(s1, g1 + 5)
                prompts = {ident.id: {mid - w0: {
                    "point": [hit["x"] * scale, hit["y"] * scale]}}}
                self._run_window(video_path, w0, w1, prompts, scale,
                                 ref_emb, {ident.id: ident})

    @staticmethod
    def _gaps(ident: _Identity, s0: int, s1: int):
        """ショット [s0, s1) 内のキーフレームの穴（長さ GAP_MIN 以上）を列挙。"""
        frames = sorted(f for f in (int(k) for k in ident.keyframes)
                        if s0 <= f < s1)
        gaps = []
        prev = s0 - 1
        for f in frames + [s1]:
            if f - prev - 1 >= GAP_MIN:
                gaps.append((prev + 1, f - 1))
            prev = f
        return gaps

    # ── AI 楕円補正 ───────────────────────────────────────────────────
    @staticmethod
    def _suspicion(ident: _Identity, frames: list[int],
                   conf: dict[int, float] | None) -> dict[int, float]:
        """
        各フレームの「疑わしさ」を返す。近傍中央値からの位置・サイズの逸脱
        （追跡ドリフト）と、低信頼（横顔・未測定）を合成する。
        """
        import statistics
        kf = ident.keyframes
        cx = {f: kf[str(f)]["cx"] for f in frames}
        cy = {f: kf[str(f)]["cy"] for f in frames}
        rx = {f: kf[str(f)]["rx"] for f in frames}
        ry = {f: kf[str(f)]["ry"] for f in frames}
        W = 7
        susp: dict[int, float] = {}
        for i, f in enumerate(frames):
            nb = frames[max(0, i - W):i + W + 1]
            mcx = statistics.median(cx[g] for g in nb)
            mcy = statistics.median(cy[g] for g in nb)
            mrx = max(1.0, statistics.median(rx[g] for g in nb))
            mry = max(1.0, statistics.median(ry[g] for g in nb))
            dpos = math.hypot(cx[f] - mcx, cy[f] - mcy) / mrx
            dsize = abs(rx[f] - mrx) / mrx + abs(ry[f] - mry) / mry
            lowconf = 0.0 if conf is None else (1.0 - conf.get(f, 0.0))
            susp[f] = max(dpos, 0.7 * dsize) + 0.8 * lowconf
        return susp

    @staticmethod
    def _susp_spans(frames: list[int], susp: dict[int, float]
                    ) -> list[tuple[int, int, int, float]]:
        """
        疑わしいフレームを連続スパンにまとめる。
        戻り値: [(start, end, worst_frame, severity)] を severity 降順で。
        """
        flagged = [f for f in frames if susp[f] >= AI_CORRECT_SUSP]
        if not flagged:
            return []
        spans = []
        a = prev = flagged[0]
        for f in flagged[1:]:
            if f - prev <= AI_CORRECT_MERGE_GAP:
                prev = f
                continue
            spans.append((a, prev))
            a = prev = f
        spans.append((a, prev))
        out = []
        for a, b in spans:
            inner = [f for f in frames if a <= f <= b]
            if len(inner) < AI_CORRECT_MIN_SPAN:
                continue
            worst = max(inner, key=lambda f: susp[f])
            severity = sum(susp[f] for f in inner)
            out.append((a, b, worst, severity))
        out.sort(key=lambda s: -s[3])
        return out

    def _ai_correct_region(self, video_path: str, ident: _Identity,
                           region: str, conf: dict[int, float] | None = None,
                           measured: set | None = None) -> None:
        """
        幾何が苦手な低信頼スパンを Claude に範囲補正させる（コスト上限つき）。

        各スパンの最悪フレームだけ Claude に矩形を問い合わせ、得た補正
        （中心オフセット + サイズ比）をスパン全体へ適用する（端はテーパー
        して近傍の良フレームと滑らかに繋ぐ）。SAM 2 の再伝播は不要。
        """
        if (not self.use_autoclick or self.ref_crop is None
                or not autoclick.available()):
            return
        vis = [int(k) for k, v in ident.keyframes.items()
               if v.get("visible", True)]
        vis.sort()
        if len(vis) < AI_CORRECT_MIN_SPAN:
            return
        conf = dict(conf or {})
        # 未測定フレーム（横顔等で実測できず補間されたもの）は信頼度0扱い
        if measured is not None:
            for f in vis:
                if f not in measured:
                    conf.setdefault(f, 0.0)
        susp = self._suspicion(ident, vis, conf)
        spans = self._susp_spans(vis, susp)
        if not spans:
            return
        src = video_manager.get(video_path)
        lo, hi = AI_CORRECT_SCALE_CLAMP
        calls = self.stats.get("ai_corrections", 0)
        for a, b, worst, _sev in spans:
            if calls >= AI_CORRECT_MAX_CALLS:
                break
            img = src.get_frame(worst)
            if img is None:
                continue
            ell = ident.keyframes[str(worst)]
            self.message = f"AI範囲補正 frame {worst}"
            calls += 1
            self.stats["ai_corrections"] = calls
            try:
                box = autoclick.refine_region(img, ell, self.ref_crop,
                                              region, self.pid)
            except Exception as e:
                self.stats["ai_correct_error"] = str(e)[:200]
                self.message = f"AI範囲補正を中断（追跡は続行）: {e}"
                return
            if box is None:
                continue
            dcx = box["cx"] - ell["cx"]
            dcy = box["cy"] - ell["cy"]
            sx = min(hi, max(lo, box["rx"] / max(1.0, ell["rx"])))
            sy = min(hi, max(lo, box["ry"] / max(1.0, ell["ry"])))
            span = [f for f in vis if a <= f <= b]
            n = len(span)
            for j, f in enumerate(span):
                # 端 AI_CORRECT_TAPER フレームでデルタを0へ線形に戻す
                edge = min(j + 1, n - j)
                t = min(1.0, edge / (AI_CORRECT_TAPER + 1))
                kf = dict(ident.keyframes[str(f)])
                kf["cx"] = round(kf["cx"] + dcx * t, 1)
                kf["cy"] = round(kf["cy"] + dcy * t, 1)
                kf["rx"] = round(kf["rx"] * (1 + (sx - 1) * t), 1)
                kf["ry"] = round(kf["ry"] * (1 + (sy - 1) * t), 1)
                kf["src"] = "auto"
                ident.keyframes[str(f)] = kf
            self.stats["ai_correct_hits"] = self.stats.get(
                "ai_correct_hits", 0) + 1


def _bbox_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1])
                    + (b[2] - b[0]) * (b[3] - b[1]) - inter)


class ReseedJob(TrackJob):
    """
    手動修正キーフレームをシードに、以降の区間を SAM 2 で再伝播する。

    「1フレーム目の修正を引き継ぎながら以降を検出し直す」機能の実体。
    次の手動キーフレームの手前（最大 MAX_SPAN フレーム）まで置き換える。
    """

    MAX_SPAN = 240

    def __init__(self, pid: str, person_id: str, frame: int,
                 end_frame: int | None = None):
        super().__init__(pid, mode="all", ref_images=None, use_autoclick=False)
        self.person_id = person_id
        self.seed_frame = frame
        self.end_frame = end_frame   # 区間限定の再追跡（このフレームまで）

    def _track(self):
        proj = prj.load_project(self.pid)
        video = proj["video"]
        person = next((p for p in proj["persons"]
                       if p["id"] == self.person_id), None)
        if person is None:
            raise ValueError("人物が見つかりません")
        kfs = proj["keyframes"].get(self.person_id, {})
        seed = kfs.get(str(self.seed_frame))
        if seed is None or not seed.get("visible", True):
            raise ValueError("このフレームにシードになるキーフレームがありません")
        self.region = person.get("region", "face")

        total = video["frame_count"]
        next_manual = min(
            (int(f) for f, k in kfs.items()
             if int(f) > self.seed_frame and k.get("src") == "manual"),
            default=total)
        end = min(self.seed_frame + self.MAX_SPAN, next_manual, total)
        if self.end_frame is not None:
            end = min(end, self.end_frame + 1)
        if end - self.seed_frame < 2:
            raise ValueError("再追跡する区間がありません（直後に手動キーフレームがあります）")

        scale = min(1.0, SAM_MAX_SIDE / max(video["width"], video["height"]))
        # シード決定: 口元人物は「顔全体」を SAM 2 に追わせる方が安定
        # （唇パッチは模様が少なくマスクが崩壊・漂流しやすい）
        calibrated = False
        adjust = person.get("mouth_adjust")
        template = person.get("mouth_template")
        head_seeded = False
        seed_box_src = seed
        if self.region == "mouth":
            src_v = video_manager.get(video["path"])
            fimg = src_v.get_frame(self.seed_frame)
            if fimg is not None:
                det = _detect_face_near(fimg, seed, 2.5)
                if det is not None:
                    # 手動シードなら形状テンプレートを学習（Mocha式）
                    if seed.get("src") == "manual":
                        template = make_mouth_template(seed, det["kps"])
                        calibrated = True
                    head_seeded = True
                    seed_box_src = None
                    fb = det["bbox"]
                    box = [fb[0] * scale, fb[1] * scale,
                           fb[2] * scale, fb[3] * scale]
        if seed_box_src is not None:
            a = math.radians(seed_box_src.get("angle", 0.0))
            dx = math.hypot(seed_box_src["rx"] * math.cos(a),
                            seed_box_src["ry"] * math.sin(a))
            dy = math.hypot(seed_box_src["rx"] * math.sin(a),
                            seed_box_src["ry"] * math.cos(a))
            box = [(seed_box_src["cx"] - dx) * scale,
                   (seed_box_src["cy"] - dy) * scale,
                   (seed_box_src["cx"] + dx) * scale,
                   (seed_box_src["cy"] + dy) * scale]

        ident = _Identity(1)
        self.message = f"修正を引き継いで再追跡中 frame {self.seed_frame}〜{end}"
        self._run_window(video["path"], self.seed_frame, end,
                         {1: {0: {"box": box}}}, scale, None, {1: ident})
        self.progress = 0.8

        # 顔全体モード: マスク縮小で口元・顎が外れないよう顔検出を下限に拡大し、
        # 遮蔽で壊れた区間は良い端点間の補間でブリッジする
        if self.region == "face":
            bbs = self._expand_face_coverage(video["path"], ident)
            expected = {f: (bb[2] - bb[0]) * (bb[3] - bb[1]) / 4 * 1.15
                        for f, bb in bbs.items()}
            self._bridge_occlusions(video["path"], ident,
                                    blur_cfg=person.get("blur"),
                                    expected=expected)

        if self.region == "mouth":
            refined, _conf = self._refine_mouth(
                video["path"], ident,
                margin_factor=(1.0 if head_seeded else 2.5),
                adjust=adjust, template=template)
            if head_seeded:
                # 頭部追跡 + 実測 + 相対補間（フル追跡と同じ変換）
                convert_ident_to_mouth(ident, refined)
            else:
                # フォールバック: 実測できたフレームだけ置き換え、
                # 実測できないフレームは信用せず捨てる（壊れた値を書かない）
                ident.keyframes = {
                    str(fi): dict(m, visible=True, src="auto")
                    for fi, m in refined.items()
                }

        # 保存: シードの後ろ〜end の自動キーフレームを置き換え（手動は保護）
        proj = prj.load_project(self.pid)
        if calibrated:
            for p in proj["persons"]:
                if p["id"] == self.person_id:
                    p["mouth_template"] = template
                    p.pop("mouth_adjust", None)
            proj.setdefault("targets", {})["mouth_template"] = template
            proj["targets"].pop("mouth_adjust", None)
        kfs = proj["keyframes"].setdefault(self.person_id, {})
        # 失敗検知の特徴量も keyframe と歩調を合わせて更新。
        tf = proj.setdefault("track_features", {}).setdefault(
            self.person_id, {})
        updated = 0
        for f in range(self.seed_frame + 1, end):
            old = kfs.get(str(f))
            if old is not None and old.get("src") == "manual":
                # keyframe は保護（手動を上書きしない）。ただし失敗ラベル
                # (=手動修正) に対応する AI 測定の特徴量が未記録なら残す。
                # 既存があれば当時の値を優先保持。これが正例の教師になる。
                nf = ident.track_features.get(str(f))
                if nf is not None and str(f) not in tf:
                    tf[str(f)] = nf
                continue
            new = ident.keyframes.get(str(f))
            if new is not None:
                kfs[str(f)] = new
                nf = ident.track_features.get(str(f))
                if nf is not None:
                    tf[str(f)] = nf
                else:
                    tf.pop(str(f), None)
                updated += 1
            elif old is not None:
                kfs.pop(str(f), None)  # 追跡が途切れた区間は非表示へ
                tf.pop(str(f), None)
        prj.save_project(proj)

        self.progress = 1.0
        self.state = "done"
        self.message = (f"再追跡完了: {updated}フレーム更新 "
                        f"({self.seed_frame + 1}〜{end - 1})"
                        + ("｜手動の形を学習し以降に適用" if calibrated else ""))


class MeasureFeaturesJob(TrackJob):
    """keyframe を一切変更せず、既存 mouth 人物の各フレームで _refine_mouth を
    走らせて track_features（失敗検知の特徴量）だけを記録するレトロ用パス。

    既存プロジェクト（コード変更前に追跡＝特徴量なし）の手動修正ラベルに
    当時相当の AI 特徴量を後付けする。頭部検出領域(detections)をシードにするため
    手動フレーム=正例にも特徴量が付く（reseed では窓の境界になり付かなかった）。
    設計: .company/engineering/docs/manual-correction-as-failure-labels.md
    """

    def __init__(self, pid: str):
        super().__init__(pid, region="mouth")

    def _track(self):
        proj = prj.load_project(self.pid)
        video = proj["video"]
        persons = [p for p in proj.get("persons", [])
                   if p.get("region") == "mouth"]
        if not persons:
            self.state = "done"
            self.progress = 1.0
            self.message = "口元人物がありません"
            return

        tf_all = proj.setdefault("track_features", {})
        targets = proj.get("targets") or {}
        total_new = 0
        for pi, person in enumerate(persons):
            self._check_cancel()
            pid_p = person["id"]
            dets = proj.get("detections", {}).get(pid_p, {})
            if not dets:
                continue
            # 頭部検出領域をシードに毎フレーム顔検出→特徴量算出（keyframeは不変）
            ident = _Identity(1)
            ident.keyframes = dets
            adjust = person.get("mouth_adjust") or targets.get("mouth_adjust")
            self.message = f"特徴量を測定中... 人物{pi + 1}/{len(persons)}"
            # template を敢えて渡さずゲートを走らせる。measure は refined 楕円を
            # 捨てるので配置には無影響。狙いは dev/gate_th を全フレームに付けること。
            # （template運用だとゲートがskipされ dev が欠損する＝素材で最強の
            #  弁別特徴が埋もれる問題を回避する。CONF_FAILED フレームも suspect と
            #  なり近傍補間からの dev が計算される。）
            self._refine_mouth(video["path"], ident,
                               adjust=adjust, template=None)
            # 再測定: 対象人物の track_features を今回の測定で再構築する
            # （既存優先だと2回目に dev が足せない。measure は決定論的な
            #  再測定なので置換で一貫する。keyframe は依然不変）。
            tf_all[pid_p] = ident.track_features
            total_new += len(ident.track_features)

        prj.save_project(proj)
        self.progress = 1.0
        self.state = "done"
        self.message = f"特徴量測定完了: {total_new}フレーム"


def start_measure_features(pid: str) -> dict:
    with _jobs_lock:
        job = _jobs.get(pid)
        if job and job.state == "running":
            return {"error": "ジョブが実行中です"}
        job = MeasureFeaturesJob(pid)
        _jobs[pid] = job
        job.start_job()
        return job.status()


# ── ジョブ管理 ───────────────────────────────────────────────────────────────

_jobs: dict[str, TrackJob] = {}
_jobs_lock = threading.Lock()


def start_track(pid: str, mode: str = "all",
                ref_images: list[str] | None = None,
                use_autoclick: bool = True, region: str = "face",
                low_ram: bool = False) -> dict:
    with _jobs_lock:
        job = _jobs.get(pid)
        if job and job.state == "running":
            return {"error": "トラッキングジョブが実行中です"}
        job = TrackJob(pid, mode, ref_images, use_autoclick, region, low_ram)
        _jobs[pid] = job
        job.start_job()
        return job.status()


_step_lock = threading.Lock()


def step_propagate(pid: str, person_id: str, frame: int) -> dict:
    """
    現在フレームの楕円をシードに、次の1フレームだけ SAM 2 伝播する（Sキー）。
    同期実行（1〜3秒程度）。手動キーフレームは上書きしない。
    """
    job = _jobs.get(pid)
    if job and job.state == "running":
        return {"error": "ジョブ実行中は1フレーム伝播を使えません"}

    with _step_lock:
        from .render import interp_ellipse

        proj = prj.load_project(pid)
        video = proj["video"]
        total = video["frame_count"]
        if frame + 1 >= total:
            return {"error": "最終フレームです"}
        person = next((p for p in proj["persons"]
                       if p["id"] == person_id), None)
        if person is None:
            return {"error": "人物が見つかりません"}
        kfs = proj["keyframes"].get(person_id, {})
        ell = kfs.get(str(frame)) or interp_ellipse(kfs, frame)
        if (ell is None or not ell.get("visible", True)
                or ell.get("parked")):
            return {"error": "このフレームに楕円がありません（先にキーフレームを打ってください）"}

        scale = min(1.0, SAM_MAX_SIDE / max(video["width"], video["height"]))
        is_mouth = person.get("region") == "mouth"
        calibrated = False
        adjust = person.get("mouth_adjust")
        head_seeded = False
        box = None
        src_v = video_manager.get(video["path"])
        next_is_manual = ((kfs.get(str(frame + 1)) or {}).get("src")
                          == "manual")

        template = person.get("mouth_template")
        if is_mouth:
            fimg = src_v.get_frame(frame)
            if fimg is None:
                return {"error": f"フレーム {frame} を読めません"}
            det = _detect_face_near(fimg, ell, 2.5)
            if det is not None:
                # 手動シードなら形状テンプレートを学習（Mocha式:
                # 形は固定し、顔の位置/スケール/傾きにだけ追従させる）
                if (kfs.get(str(frame)) or {}).get("src") == "manual":
                    template = make_mouth_template(ell, det["kps"])
                    calibrated = True
                head_seeded = True
                fb = det["bbox"]
                box = [fb[0] * scale, fb[1] * scale,
                       fb[2] * scale, fb[3] * scale]

        def _persist_template():
            if not calibrated:
                return
            proj2 = prj.load_project(pid)
            for p2 in proj2["persons"]:
                if p2["id"] == person_id:
                    p2["mouth_template"] = template
                    p2.pop("mouth_adjust", None)
            proj2.setdefault("targets", {})["mouth_template"] = template
            proj2["targets"].pop("mouth_adjust", None)
            prj.save_project(proj2)

        # 次フレームが手動KFなら伝播不要: 学習だけ保存して移動する
        if next_is_manual:
            _persist_template()
            return {"ok": True, "frame": frame + 1,
                    "kf": kfs[str(frame + 1)], "skipped": "next_manual",
                    "calibrated": calibrated}
        if box is None:
            a = math.radians(ell.get("angle", 0.0))
            dxr = math.hypot(ell["rx"] * math.cos(a), ell["ry"] * math.sin(a))
            dyr = math.hypot(ell["rx"] * math.sin(a), ell["ry"] * math.cos(a))
            box = [(ell["cx"] - dxr) * scale, (ell["cy"] - dyr) * scale,
                   (ell["cx"] + dxr) * scale, (ell["cy"] + dyr) * scale]

        helper = TrackJob(pid)
        ident = _Identity(1)
        helper._run_window(video["path"], frame, frame + 2,
                           {1: {0: {"box": box}}}, scale, None, {1: ident})

        if is_mouth:
            # 実測できたときだけ書き込む（壊れた値でマスクが飛ぶのを防ぐ）
            refined, _conf = helper._refine_mouth(
                video["path"], ident,
                margin_factor=(1.0 if head_seeded else 2.5),
                adjust=adjust, template=template)
            m = refined.get(frame + 1)
            if m is None:
                return {"error": "次フレームで口元を実測できませんでした。"
                                 "書き込みを中止します — 手動で調整してください"}
            newkf = dict(m, visible=True, src="auto")
        else:
            newkf = ident.keyframes.get(str(frame + 1))
            if newkf is None:
                return {"error": "伝播に失敗しました（次フレームでマスクが得られません）"}
            # 顔全体: マスク縮小・暴走で口元・顎が外れないよう顔検出で補正
            fimg2 = src_v.get_frame(frame + 1)
            if fimg2 is not None:
                det2 = _detect_face_near(fimg2, newkf, 0.8)
                if det2 is not None:
                    newkf = _fit_face_ellipse(newkf, det2["bbox"])

        # 大ジャンプ・潰れガード: 1フレームでこの移動/縮小はあり得ない
        rmax = max(ell["rx"], ell["ry"])
        jump = math.hypot(newkf["cx"] - ell["cx"], newkf["cy"] - ell["cy"])
        if (jump > rmax * 1.5 + 24
                or max(newkf["rx"], newkf["ry"]) < rmax * 0.3):
            return {"error": f"追跡が大きく外れたため書き込みを中止しました"
                             f"（移動{int(jump)}px）。手動で調整してください"}

        _persist_template()
        proj = prj.load_project(pid)
        pkfs = proj["keyframes"].setdefault(person_id, {})
        old = pkfs.get(str(frame + 1))
        if old is not None and old.get("src") == "manual":
            return {"error": "次フレームは手動キーフレームです（上書きしません）"}
        pkfs[str(frame + 1)] = newkf
        # 口元は _refine_mouth 経由なので当該フレームの特徴量を非破壊で残す
        if is_mouth:
            nf = ident.track_features.get(str(frame + 1))
            if nf is not None:
                proj.setdefault("track_features", {}).setdefault(
                    person_id, {})[str(frame + 1)] = nf
        prj.save_project(proj)
        return {"ok": True, "frame": frame + 1, "kf": newkf,
                "calibrated": calibrated}


def start_reseed(pid: str, person_id: str, frame: int,
                 end_frame: int | None = None) -> dict:
    with _jobs_lock:
        job = _jobs.get(pid)
        if job and job.state == "running":
            return {"error": "トラッキングジョブが実行中です"}
        job = ReseedJob(pid, person_id, frame, end_frame)
        _jobs[pid] = job
        job.start_job()
        return job.status()


def track_status(pid: str) -> dict | None:
    job = _jobs.get(pid)
    return job.status() if job else None


def cancel_track(pid: str) -> bool:
    job = _jobs.get(pid)
    if job and job.state == "running":
        job.cancel_requested = True
        return True
    return False
