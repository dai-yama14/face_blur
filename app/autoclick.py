"""
autoclick.py - Claude vision による自動クリック（再シード座標の取得）

SAM 2 伝播で埋められなかった区間のフレームを Claude に見せ、
対象人物の頭部座標を回答させて SAM 2 の再シードに使う。

Claude = 自動クリック係（だいたいの位置）、SAM 2 = 精密な範囲決め係、という分業。
ANTHROPIC_API_KEY が未設定の場合は利用不可（呼び出し側でスキップする）。
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

MODEL = os.environ.get("CLAUDE_AUTOCLICK_MODEL", "claude-sonnet-5")
MAX_SIDE = 1024  # 送信画像の長辺上限（コスト対策）

# 累計使用量の保存先
USAGE_PATH = Path(__file__).parent.parent / "api_usage.json"
# 1Mトークンあたりの料金 (USD, input/output)。未知モデルは高め(Opus相当)に見積もる
PRICES = {
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-opus-4-8": (5.0, 25.0),
}
DEFAULT_PRICE = (5.0, 25.0)

_client = None
_usage_lock = threading.Lock()
_active_calls = 0


def _empty_bucket() -> dict:
    return {"calls": 0, "input_tokens": 0, "output_tokens": 0,
            "est_cost_usd": 0.0, "last_call": None}


def _load_usage() -> dict:
    u = _empty_bucket()
    u["projects"] = {}
    if USAGE_PATH.exists():
        try:
            with open(USAGE_PATH, "r", encoding="utf-8") as f:
                u.update(json.load(f))
            u.setdefault("projects", {})
        except Exception:
            pass
    return u


def _add(bucket: dict, input_tokens: int, output_tokens: int,
         price_in: float, price_out: float) -> None:
    bucket["calls"] += 1
    bucket["input_tokens"] += input_tokens
    bucket["output_tokens"] += output_tokens
    bucket["est_cost_usd"] = round(
        bucket["est_cost_usd"]
        + input_tokens / 1e6 * price_in
        + output_tokens / 1e6 * price_out, 6)
    bucket["last_call"] = datetime.now().isoformat(timespec="seconds")


def _record_usage(model: str, input_tokens: int, output_tokens: int,
                  pid: str | None = None) -> None:
    price_in, price_out = PRICES.get(model, DEFAULT_PRICE)
    with _usage_lock:
        u = _load_usage()
        _add(u, input_tokens, output_tokens, price_in, price_out)
        if pid:
            b = u["projects"].setdefault(pid, _empty_bucket())
            _add(b, input_tokens, output_tokens, price_in, price_out)
        tmp = USAGE_PATH.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(u, f)
        tmp.replace(USAGE_PATH)


def get_usage() -> dict:
    """累計 + プロジェクト別使用量 + 現在呼び出し中かどうか。"""
    with _usage_lock:
        u = _load_usage()
    u["active"] = _active_calls > 0
    u["model"] = MODEL
    return u


def drop_project_usage(pid: str) -> None:
    """プロジェクト削除時に紐づく集計を消す（全体累計は残す）。"""
    with _usage_lock:
        u = _load_usage()
        if u["projects"].pop(pid, None) is not None:
            tmp = USAGE_PATH.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(u, f)
            tmp.replace(USAGE_PATH)


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


def _to_b64_jpeg(img: np.ndarray, max_side: int = MAX_SIDE) -> tuple[str, float]:
    """画像を縮小して base64 JPEG にする。縮小率も返す。"""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("JPEGエンコード失敗")
    return base64.standard_b64encode(buf.tobytes()).decode(), scale


def _img_block(img: np.ndarray) -> dict:
    b64, _ = _to_b64_jpeg(img)
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
    }


def _with_grid(img: np.ndarray) -> np.ndarray:
    """
    正規化座標(0-1000)のグリッドを描き込む。VLM は目盛りを読んで座標を
    答えられるため、素の画像より座標回答の精度が大幅に上がる。
    """
    out = img.copy()
    h, w = out.shape[:2]
    color = (80, 255, 80)
    for i in range(1, 10):
        x, y = int(w * i / 10), int(h * i / 10)
        cv2.line(out, (x, 0), (x, h), color, 1)
        cv2.line(out, (0, y), (w, y), color, 1)
        cv2.putText(out, str(i * 100), (x + 3, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(out, str(i * 100), (3, y + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return out


_GRID_NOTE = """\
フレームには緑のグリッド線が描かれています。縦線・横線に付いた数字は
正規化座標（左上0,0〜右下1000,1000）の目盛りです。
座標はこの目盛りを読み取って答えてください。"""


def _call(content: list, max_tokens: int = 200,
          pid: str | None = None) -> str:
    """Claude を呼び出して使用量を記録し、テキストを返す。"""
    global _active_calls
    client = _get_client()
    _active_calls += 1
    try:
        resp = client.messages.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": content}],
        )
    finally:
        _active_calls -= 1
    _record_usage(MODEL, resp.usage.input_tokens, resp.usage.output_tokens, pid)
    return "".join(b.text for b in resp.content if b.type == "text")


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def same_person(crop: np.ndarray, ref_face: np.ndarray,
                pid: str | None = None) -> bool | None:
    """
    追跡中の人物クロップとリファレンス顔が同一人物かを判定する（②曖昧ゾーン仲裁）。
    髪型・服装・体格など顔認識が使えない手がかりも判断材料にできる。
    戻り値: True=同一人物 / False=別人 / None=判断不能・エラー
    """
    if not available():
        return None
    content = [
        {"type": "text", "text": "画像1: リファレンス人物の顔写真"},
        _img_block(ref_face),
        {"type": "text", "text": "画像2: 動画から切り出した追跡中の人物"},
        _img_block(crop),
        {"type": "text", "text": """\
画像2の人物は画像1と同一人物ですか？
横顔・後ろ向きでも、髪型・髪色・服装・体格から判断してください。
判断できない場合は unknown にしてください。

以下のJSONのみを出力してください:
{"same": true/false, "confidence": "high"/"low", "unknown": true/false}"""},
    ]
    data = _parse_json(_call(content, pid=pid))
    if data is None or data.get("unknown"):
        return None
    if data.get("confidence") == "low":
        return None  # 低確信は判定保留（誤棄却防止）
    return bool(data.get("same"))


def check_leak(rendered: np.ndarray, ref_face: np.ndarray | None,
               region: str = "face", pid: str | None = None) -> dict | None:
    """
    ブラー合成済みフレームに対象人物の顔（または口元）が生で見えていないかを
    検品する（①最終QC）。
    戻り値: {"x": px, "y": px}（漏れ位置）/ None（漏れなし・判断不能）
    """
    if not available():
        return None
    h, w = rendered.shape[:2]
    content = []
    if ref_face is not None:
        content.append({"type": "text", "text": "リファレンス: ブラーすべき対象人物の顔"})
        content.append(_img_block(ref_face))
    content.append({"type": "text",
                    "text": "次はブラー処理済みの動画フレームです:"})
    content.append(_img_block(_with_grid(rendered)))
    target = ("リファレンスと同一人物" if ref_face is not None else "いずれかの人物")
    if region == "mouth":
        criteria = f"""\
このフレームで、{target}の「口元（小鼻・口・顎の周辺）」がブラー（ぼかし/モザイク）
されずにはっきり見えていないか検品してください。
- 小鼻から顎までがぼけている / モザイクがかかっている → 問題なし
- ※目元や鼻筋より上が見えているのは正常です（口元だけを隠す設定のため）
- 口・唇・顎・小鼻（鼻の膨らみ）がはっきり識別できる → 漏れ

漏れがある場合はその口元の中心座標をグリッド目盛りで答えてください。"""
    else:
        criteria = f"""\
このフレームで、{target}の顔がブラー（ぼかし/モザイク）されずに
はっきり識別できる状態で映っていないか検品してください。
- 顔全体がぼけている / モザイクがかかっている → 問題なし
- 顔の特徴（目鼻立ち）が識別できる → 漏れ

漏れがある場合はその頭部中心座標をグリッド目盛りで答えてください。"""
    content.append({"type": "text", "text": f"""\
{criteria}

{_GRID_NOTE}

以下のJSONのみを出力してください:
{{"leak": true/false, "x": 0-1000, "y": 0-1000}}"""})
    data = _parse_json(_call(content, pid=pid))
    if not data or not data.get("leak"):
        return None
    x, y = float(data.get("x", -1)), float(data.get("y", -1))
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return {"x": x / 1000.0 * w, "y": y / 1000.0 * h}


def find_head(frame: np.ndarray, ref_face: np.ndarray | None,
              pid: str | None = None) -> dict | None:
    """
    フレーム内の対象人物の頭部中心を Claude に尋ねる。

    戻り値: {"x": px, "y": px, "confidence": "high|low"}（フレーム座標）
            見つからない/判断不能なら None。
    """
    if not available():
        return None

    h, w = frame.shape[:2]
    content = []
    if ref_face is not None:
        content.append({"type": "text",
                        "text": "これが対象人物のリファレンス顔写真です:"})
        content.append(_img_block(ref_face))
        content.append({"type": "text", "text": "次が動画のフレームです:"})
    content.append(_img_block(_with_grid(frame)))
    target_desc = ("リファレンス顔写真と同一人物" if ref_face is not None
                   else "最も目立つ人物")
    content.append({"type": "text", "text": f"""\
この動画フレームに{target_desc}が映っているか判定してください。
横顔・後ろ向き・一部隠れでも、髪型・服装・体格から判断してください。

{_GRID_NOTE}

映っている場合はその人物の頭部の中心座標をグリッド目盛りで答えてください。

以下のJSONのみを出力してください（説明文は不要）:
{{"found": true/false, "x": 0-1000, "y": 0-1000, "confidence": "high"/"low"}}"""})

    data = _parse_json(_call(content, pid=pid))
    if data is None or not data.get("found"):
        return None
    x = float(data.get("x", -1))
    y = float(data.get("y", -1))
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        return None
    return {
        "x": x / 1000.0 * w,
        "y": y / 1000.0 * h,
        "confidence": data.get("confidence", "low"),
    }


def refine_region(frame: np.ndarray, ell: dict, ref_face: np.ndarray | None,
                  region: str = "mouth", pid: str | None = None) -> dict | None:
    """
    幾何（ランドマーク）が苦手なフレームの領域楕円を Claude に補正させる。

    現在の楕円 ell の周辺を切り出してグリッド付きで見せ、対象領域
    （口元 or 顔全体）を囲む最小矩形をグリッド座標で答えさせる。
    find_head が「中心点」しか返さないのに対し、こちらは「範囲（矩形）」を
    返すので、横顔・遮蔽でズレた楕円の位置とサイズを補正できる。

    戻り値: {"cx","cy","rx","ry"}（フレーム座標）/ None（不在・低確信・異常）。
    """
    if not available():
        return None
    h, w = frame.shape[:2]
    r = max(ell["rx"], ell["ry"])
    # 楕円がズレ／過小のときも真の領域が窓内に入るよう、余裕を持って切り出す
    # （最低でも約220px四方の窓を確保）
    half = max(r * 2.6, 110)
    x1 = max(0, int(ell["cx"] - half))
    y1 = max(0, int(ell["cy"] - half))
    x2 = min(w, int(ell["cx"] + half))
    y2 = min(h, int(ell["cy"] + half))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return None
    ch, cw = crop.shape[:2]

    content = []
    if ref_face is not None:
        content.append({"type": "text", "text": "対象人物のリファレンス顔:"})
        content.append(_img_block(ref_face))
    content.append({"type": "text", "text": "次は動画から切り出した領域です:"})
    content.append(_img_block(_with_grid(crop)))
    what = ("対象人物の口元（小鼻・口・顎を含む範囲）" if region == "mouth"
            else "対象人物の顔全体（額から顎まで）")
    content.append({"type": "text", "text": f"""\
この切り出し画像の中で、{what}を囲む最小の長方形をグリッド座標で答えてください。
横顔・うつむき・手や物で一部隠れていても、見えている手がかりから推定してください。
対象が写っていない場合は found=false にしてください。

{_GRID_NOTE}

以下のJSONのみを出力してください:
{{"found": true/false, "x1": 0-1000, "y1": 0-1000, "x2": 0-1000, "y2": 0-1000, \
"confidence": "high"/"low"}}"""})

    data = _parse_json(_call(content, max_tokens=120, pid=pid))
    # ⚠ confidence=="low" の棄却を「バグ」だと思って外さないこと（2026-07-11 検証済み）。
    # 夜間・ブラー・横顔で Claude は低確信を返すため、この行のせいで
    # 「最も助けが必要なフレームでだけ AI 補正が発火しない」構造になっている。
    # しかし低確信も採用して実測した結果、Claude の回答は幾何(3DDFA)より悪かった
    # （IoU中央 0.258 → 0.198、中心ズレ 0.77 → 0.95〜1.2 半径）。しかも Claude の
    # 自己申告 confidence は当てにならない（"high" 回答が最悪だった）。
    # つまりこの棄却は結果的にシステムを守っている。
    # 検証: .company/engineering/docs/mouth-mask-accuracy-hy.md
    if not data or not data.get("found") or data.get("confidence") == "low":
        return None
    try:
        bx1, by1, bx2, by2 = (float(data[k]) for k in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError):
        return None
    # 目盛りを少しはみ出す回答は範囲内へクランプ（棄却しない）
    bx1, bx2 = sorted((min(1000.0, max(0.0, bx1)), min(1000.0, max(0.0, bx2))))
    by1, by2 = sorted((min(1000.0, max(0.0, by1)), min(1000.0, max(0.0, by2))))
    if bx2 - bx1 < 20 or by2 - by1 < 20:
        return None                   # 極小ボックスは不採用
    # クロップのほぼ全体を囲んだ回答は「見つけられず全体を指した」とみなし棄却
    if (bx2 - bx1) * (by2 - by1) > 0.88 * 1000 * 1000:
        return None
    fx1, fx2 = x1 + bx1 / 1000.0 * cw, x1 + bx2 / 1000.0 * cw
    fy1, fy2 = y1 + by1 / 1000.0 * ch, y1 + by2 / 1000.0 * ch
    return {
        "cx": round((fx1 + fx2) / 2, 1),
        "cy": round((fy1 + fy2) / 2, 1),
        "rx": round((fx2 - fx1) / 2, 1),
        "ry": round((fy2 - fy1) / 2, 1),
    }
