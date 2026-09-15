"""
landmarks3d.py - 3DDFA_V2 による大ポーズ対応 3D ランドマーク（68点）。

InsightFace(SCRFD) の顔 bbox を入力に 3D 顔モデル(3DMM)をフィットし、
横顔・真横でも顎輪郭と唇を正確に取る。2D の 106点が射影で崩れる問題
（奥側のハルシネーション）を避けるのが目的。

モデルが無い / 初期化や推論に失敗した場合は None を返し、呼び出し側が
既存の 106点式ランドマークへフォールバックする（AI補正と同じ安全側設計）。

vendored: app/tddfa/（3DDFA_V2, MIT License, cleardusk）
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

_TDDFA_DIR = os.path.join(os.path.dirname(__file__), "tddfa")
_tddfa = None
_lock = threading.Lock()

# 初期化失敗の扱い。
#
# ⚠ 以前は初回の例外で _unavailable=True を立て、プロセス寿命の間ずっと
# None を返し続けていた。常駐 uvicorn では「一度こけたらサーバを再起動する
# まで全クリップが 68点なしで追跡される」という致命的な挙動になり、実際に
# a2_clip1 / hy_clip3 ほか9本が全フレーム CONF_FAILED のまま記録された
# （2026-07-17 に判明。conf は正常値なのに lmk_conf だけ 9.9 で飽和）。
# 失敗の証拠は stdout のトレースバックだけで、プロセスと共に消えていた。
#
# 対策: 一過性の失敗は時間を空けて再試行し、恒久的な失敗（重みが無い等）
# だけを permanent として諦める。いずれの場合も最後のエラーを保持して
# status() で外から観測できるようにする。
_RETRY_INTERVAL = 60.0     # 一過性失敗の再試行間隔（秒）
_permanent = False         # 恒久的に使用不可（重みが無い等）
_last_error: str | None = None
_last_attempt = 0.0
_attempts = 0


class MissingModelError(RuntimeError):
    """重み/設定が存在しない = 再試行しても無駄な恒久的失敗。"""

# 使用する 3DDFA_V2 のモデル。先頭が既定。
#
# ⚠ resnet22（18.5M params）への格上げは **実測で棄却した**（2026-07-11）。
# パラメータは mb1（3.3M）の5.6倍だが、hy_clip1/hy_clip2 の手動修正フレームを
# 正解として測ると精度は落ちる:
#   hy_clip1 IoU中央 0.774 → 0.735
#   hy_clip2 IoU中央 0.728 → 0.262（崩壊率 0% → 76%）
# 横顔で口元楕円が小さくなり鼻側へ寄る（＝「鼻だけマスク」そのもの）。
# しかも jitter 一致性は resnet22 のほうが良く（0.0276→0.0233）、
# 「より安定して、より間違っている」ため信頼度ゲートでも検知できない。
# 重みは残してあるので TDDFA_CONFIG=resnet_120x120.yml で再検証は可能。
# 検証: .company/engineering/docs/mouth-mask-accuracy-hy.md
_CONFIGS = ["mb1_120x120.yml", "resnet_120x120.yml"]


def _pick_config() -> str:
    """使える設定を優先順に選ぶ（重みファイルが実在するものだけ）。

    どれも使えなければ MissingModelError（= 再試行しても無駄）。
    """
    env = os.environ.get("TDDFA_CONFIG")
    cands = [env] + _CONFIGS if env else _CONFIGS
    for name in cands:
        path = os.path.join(_TDDFA_DIR, "configs", name)
        if not os.path.exists(path):
            continue
        import yaml
        with open(path) as f:
            cfg = yaml.safe_load(f)
        if os.path.exists(os.path.join(_TDDFA_DIR, cfg["checkpoint_fp"])):
            return name
    raise MissingModelError(
        f"3DDFA の重みが見つかりません（探した設定: {cands} / "
        f"場所: {_TDDFA_DIR}）")


def available() -> bool:
    """3DDFA_V2 が使えるか（初期化を試みる）。"""
    return _get() is not None


def status() -> dict:
    """3DDFA の生死を外から観測するための診断情報。

    「黙って死んで気づかない」を防ぐのが目的。サーバ起動時の警告と
    追跡ジョブの stats から参照する。
    """
    return {
        "loaded": _tddfa is not None,
        "permanent_failure": _permanent,
        "attempts": _attempts,
        "last_error": _last_error,
    }


def _get():
    """3DDFA を返す（使用不可なら None）。

    一過性の失敗は _RETRY_INTERVAL 秒後に再試行する。毎フレーム呼ばれる
    ため、失敗直後に再初期化を連打しないよう時間で間引く。
    """
    global _tddfa, _permanent, _last_error, _last_attempt, _attempts
    if _tddfa is not None or _permanent:
        return _tddfa
    now = time.monotonic()
    if _attempts and now - _last_attempt < _RETRY_INTERVAL:
        return None                       # 直近に失敗済み。再試行はまだ待つ
    with _lock:
        if _tddfa is not None or _permanent:
            return _tddfa
        if _attempts and time.monotonic() - _last_attempt < _RETRY_INTERVAL:
            return None
        _last_attempt = time.monotonic()
        _attempts += 1
        try:
            _tddfa = _init()
            _last_error = None
            if _attempts > 1:
                print(f"[landmarks3d] 3DDFA を復旧しました（{_attempts}回目）")
        except Exception as e:
            import traceback
            _last_error = f"{type(e).__name__}: {e}"
            _permanent = isinstance(e, MissingModelError)
            traceback.print_exc()
            print(f"[landmarks3d] ⚠ 3DDFA 初期化に失敗 "
                  f"({'恒久的' if _permanent else f'{_RETRY_INTERVAL:.0f}秒後に再試行'}): "
                  f"{_last_error}\n"
                  f"[landmarks3d] ⚠ このまま追跡すると 68点3Dランドマークは使われず、"
                  f"106点/5点フォールバックで口元が配置されます（精度が落ちます）")
    return _tddfa


def _init():
    import sys
    import yaml
    import torch

    # 旧repo対応: torch>=2.6 は torch.load の weights_only 既定が True で
    # pickle 化された設定/重みの読み込みが失敗するため False を強制する
    _orig_load = torch.load

    def _patched_load(*a, **k):
        k.setdefault("weights_only", False)
        return _orig_load(*a, **k)

    torch.load = _patched_load

    # 3DDFA_V2 は絶対インポート（import models / from bfm import ... など）なので
    # ベンダリング先を sys.path に載せる
    if _TDDFA_DIR not in sys.path:
        sys.path.insert(0, _TDDFA_DIR)

    name = _pick_config()
    with open(os.path.join(_TDDFA_DIR, "configs", name)) as f:
        cfg = yaml.safe_load(f)
    print(f"[landmarks3d] 3DDFA config: {name} (arch={cfg.get('arch')})")
    # yml 内の相対パス（cwd 基準で解決される）を絶対パス化して chdir を不要にする
    cfg["checkpoint_fp"] = os.path.join(_TDDFA_DIR, cfg["checkpoint_fp"])
    cfg["bfm_fp"] = os.path.join(_TDDFA_DIR, cfg["bfm_fp"])
    cfg["param_mean_std_fp"] = os.path.join(
        _TDDFA_DIR, "configs", "param_mean_std_62d_120x120.pkl")

    from TDDFA import TDDFA
    gpu = _cuda_ok()
    return TDDFA(gpu_mode=gpu, **cfg)


def _cuda_ok() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _no_grad():
    """推論を必ず no_grad で囲むためのコンテキスト。

    ⚠ **必須**。3DDFA_V2 の `TDDFA.__init__` は `torch.set_grad_enabled(False)`
    を呼ぶが（TDDFA.py:31）、**これはスレッドローカル**。一方 `TDDFA.__call__` は
    `param.squeeze().cpu().numpy()`（TDDFA.py:117）を実行するため、grad が有効な
    スレッドでは毎回

        RuntimeError: Can't call numpy() on Tensor that requires grad.

    で落ちる。追跡ジョブはジョブごとに新しいスレッドで走るので、
    **初期化したスレッド以外では 3DDFA が全フレーム失敗する**という挙動になっていた
    （2026-07-17 判明）。例外は呼び出し側の `except Exception` に握り潰され、
    CONF_FAILED として記録されるため、素材が難しいのと区別がつかなかった。

    実害: ht_clip9(初回ジョブ)=成功 / 直後のバッチ8本=全滅、
    ht_clip7(再起動直後の初回)=成功 / hy_clip3(2番目)=全滅、というように
    「サーバ内で最初に3DDFAを触ったジョブだけが成功する」状態だった。

    初期化スレッドに依存しないよう、呼び出し側で明示的に no_grad を張る。
    """
    import torch
    return torch.no_grad()


def get_landmarks_68(img: np.ndarray, bbox) -> np.ndarray | None:
    """
    顔 bbox=[x1,y1,x2,y2] を入力に、iBUG 68点(3D)を画像座標で返す (68,3)。
    使用不可・失敗時は None。
    """
    t = _get()
    if t is None:
        return None
    try:
        with _no_grad():
            param_lst, roi_box_lst = t(img, [list(bbox)])
            ver = t.recon_vers(param_lst, roi_box_lst,
                               dense_flag=False)[0]  # (3,68)
        return np.asarray(ver, dtype=float).T  # (68,3)
    except Exception:
        return None


# ── フィット破綻の検知（jitter 一致性） ─────────────────────────────
# 3DDFA は横顔・大ピッチ・モーションブラーが重なると 3DMM フィットが破綻し、
# 68点が画像中に散乱する。それでも「それらしい値」を返すため、下流の楕円は
# 顔から外れた場所（目・額・耳）に描かれてしまう。
#
# bbox をわずかに摂動して再フィットし、結果のばらつきを見る。健全なフィットは
# 摂動に対して安定し、破綻したフィットは大きく暴れる。姿勢に依存しない
# 自己教師あり信頼度で、hy_clip1/hy_clip2 の実測で IoU と負相関
# （-0.44 / -0.77、両クリップ同符号）を確認済み。
# 検証: .company/engineering/docs/mouth-mask-accuracy-hy.md
_JITTER = (
    (0.0, 0.0, 1.0),      # 素の bbox（この結果を採用値として返す）
    (0.06, 0.0, 1.0),
    (-0.06, 0.0, 1.0),
    (0.0, 0.06, 1.0),
    (0.0, -0.06, 1.0),
    (0.0, 0.0, 1.12),
)
CONF_FAILED = 9.9      # フィットを試みたが破綻した。しきい値より必ず大きい値

# 「3DDFA がそもそも動いていない」＝ フィットを試してすらいない。
#
# ⚠ CONF_FAILED と混同してはいけない。両者を同じ 9.9 で記録していたため、
# 2026-07-16 の分析は「難所では fit が構造的に失敗する（CONF_FAILED 100%）」
# と結論づけたが、実際にはモデルが死んでいて一度も測っていなかった
# （2026-07-17 に判明）。素材の性質と実行環境の故障を取り違えないこと。
# CONF_FAILED より大きいので、既存の `c < CONF_FAILED` 判定はそのまま
# 「使えない値」として除外できる。
CONF_UNAVAILABLE = 10.0


def get_landmarks_68_conf(img: np.ndarray, bbox) -> tuple:
    """
    68点(3D) と「フィットの怪しさ」を返す (lmk68 | None, conf)。

    conf は摂動再フィット間の点のばらつき（bbox サイズで正規化）。
    **小さいほど信用できる**。しきい値は素材ごとに分布が動くため、
    呼び出し側でトラック内の中央値を基準に自己校正すること
    （絶対値で決め打ちすると素材をまたいだ瞬間に破綻する）。

    3DDFA が使用不可なら conf は CONF_UNAVAILABLE（測っていない）。
    フィットを試みて破綻したときだけ CONF_FAILED（測ったが駄目）。
    """
    t = _get()
    if t is None:
        return None, CONF_UNAVAILABLE
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    w, h = x2 - x1, y2 - y1
    if w < 2 or h < 2:
        return None, CONF_FAILED
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    fits = []
    with _no_grad():   # スレッドローカルな grad 設定に依存しない（_no_grad 参照）
        for dx, dy, ds in _JITTER:
            px, py = cx + dx * w, cy + dy * h
            bw, bh = w * ds, h * ds
            try:
                param_lst, roi_box_lst = t(
                    img, [[px - bw / 2, py - bh / 2, px + bw / 2, py + bh / 2]])
                ver = t.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]
                fits.append(np.asarray(ver, dtype=float).T)
            except Exception:
                fits.append(None)

    base = fits[0]
    if base is None:
        return None, CONF_FAILED
    ok = [f for f in fits if f is not None]
    if len(ok) < 3:
        return base, CONF_FAILED

    pts = np.stack([f[:, :2] for f in ok])          # (n, 68, 2)
    med = np.median(pts, axis=0)
    scale = max(1.0, w, h)
    conf = float(np.linalg.norm(pts - med, axis=2).mean() / scale)
    return base, conf
