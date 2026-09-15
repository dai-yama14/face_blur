"""
project.py - プロジェクト（セーブファイル）管理

セーブファイルは projects/{id}.mvproj.json に保存する。
動画情報・人物トラック・キーフレーム・ブラー設定・UI状態をすべて含み、
エディターを閉じても開き直せば続きから作業できる。
"""
from __future__ import annotations

import gzip
import json
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).parent.parent
PROJECTS_DIR = BASE_DIR / "projects"

PROJECT_VERSION = 1

# 人物に順番に割り当てる色
PERSON_COLORS = [
    "#ff5555", "#50b0ff", "#5fd068", "#f5a623",
    "#c678dd", "#2ec4b6", "#ff7ab2", "#d4c05a",
]

_lock = threading.Lock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _slugify(name: str) -> str:
    s = re.sub(r"[^\w\-]+", "-", name.strip().lower()).strip("-")
    return s or "project"


def probe_video(video_path: str) -> dict:
    """動画のメタ情報を取得する。開けなければ ValueError。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"動画を開けません: {video_path}")
    try:
        info = {
            "path": str(Path(video_path).resolve()),
            "fps": cap.get(cv2.CAP_PROP_FPS) or 30.0,
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()
    if info["frame_count"] <= 0 or info["width"] <= 0:
        raise ValueError(f"動画情報を取得できません: {video_path}")
    return info


def default_blur() -> dict:
    return {"type": "gaussian", "strength": 30, "feather": 0.25}


def new_person(persons: list, label: str | None = None) -> dict:
    idx = len(persons)
    return {
        "id": f"p{idx + 1}_{uuid.uuid4().hex[:6]}",
        "label": label or f"人物{idx + 1}",
        "color": PERSON_COLORS[idx % len(PERSON_COLORS)],
        "enabled": True,
        "blur": default_blur(),
    }


def create_project(video_path: str, name: str | None = None) -> dict:
    video = probe_video(video_path)
    name = name or Path(video_path).stem
    pid = f"{_slugify(name)}-{uuid.uuid4().hex[:6]}"
    proj = {
        "version": PROJECT_VERSION,
        "id": pid,
        "name": name,
        "created": _now(),
        "updated": _now(),
        "video": video,
        "persons": [],
        # person_id -> { "フレーム番号(str)": {cx,cy,rx,ry,visible,src} }
        "keyframes": {},
        # person_id -> { "フレーム番号(str)": {cx,cy,rx,ry,angle} }
        # 生の顔検出領域（オレンジ「検出枠」の常時表示用）。keyframes とは別管理
        "detections": {},
        "ui": {"frame": 0},
    }
    save_project(proj)
    return proj


def project_path(pid: str) -> Path:
    # パストラバーサル防止
    if not re.fullmatch(r"[\w\-]+", pid):
        raise ValueError(f"不正なプロジェクトID: {pid}")
    return PROJECTS_DIR / f"{pid}.mvproj.json"


# ── 変更履歴（undo 用スナップショット）────────────────────────────
# 破壊的操作（区間補間・再追跡など）の「直前の状態」を projects/<id>.history/
# に gzip で退避し、最大 HISTORY_MAX 件のリングバッファとして保持する。
# 誤操作しても任意の地点まで戻せるようにするのが目的。
HISTORY_MAX = 50
# 同じラベルの保存がこの秒数内に連続したら 1 件に集約する（手動ドラッグの
# 連打などで履歴が溢れるのを防ぐ。最初の＝真の直前状態を残す）。
HISTORY_COALESCE_SEC = 8.0


def history_dir(pid: str) -> Path:
    return project_path(pid).with_suffix(".history")


def _history_index_path(pid: str) -> Path:
    return history_dir(pid) / "index.json"


def _read_history_index(pid: str) -> dict:
    p = _history_index_path(pid)
    if not p.exists():
        return {"next": 0, "entries": []}
    try:
        with open(p, "r", encoding="utf-8") as f:
            idx = json.load(f)
        idx.setdefault("next", 0)
        idx.setdefault("entries", [])
        return idx
    except (json.JSONDecodeError, OSError):
        return {"next": 0, "entries": []}


def _write_history_index(pid: str, idx: dict) -> None:
    p = _history_index_path(pid)
    tmp = p.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False)
    tmp.replace(p)


def _elapsed_sec(ts_old: str, ts_new: str) -> float:
    try:
        return (datetime.fromisoformat(ts_new)
                - datetime.fromisoformat(ts_old)).total_seconds()
    except ValueError:
        return 1e9


def _push_history(pid: str, label: str) -> None:
    """現在ディスク上のセーブファイル（＝この保存で上書きされる直前の状態）を
    履歴に退避する。ファイルがまだ無い初回保存では何もしない。"""
    with _lock:
        src = project_path(pid)
        if not src.exists():
            return
        hdir = history_dir(pid)
        hdir.mkdir(parents=True, exist_ok=True)
        idx = _read_history_index(pid)
        entries = idx["entries"]
        now = _now()
        if entries:
            last = entries[-1]
            if (last.get("label") == label
                    and _elapsed_sec(last.get("ts", ""), now)
                    < HISTORY_COALESCE_SEC):
                return  # 連続した同一操作は集約（直前状態はすでに退避済み）
        seq = idx["next"]
        idx["next"] = seq + 1
        data = src.read_bytes()
        with gzip.open(hdir / f"{seq:04d}.json.gz", "wb") as f:
            f.write(data)
        entries.append({"seq": seq, "ts": now, "label": label,
                        "size": len(data)})
        while len(entries) > HISTORY_MAX:
            old = entries.pop(0)
            try:
                (hdir / f"{old['seq']:04d}.json.gz").unlink()
            except FileNotFoundError:
                pass
        _write_history_index(pid, idx)


def list_history(pid: str) -> list[dict]:
    """履歴エントリを新しい順で返す。各エントリは、そのラベルの操作を行う
    「直前の状態」を指す（＝復元すればその操作を取り消せる）。"""
    idx = _read_history_index(pid)
    return list(reversed(idx["entries"]))


def restore_history(pid: str, seq: int) -> dict:
    """指定した履歴地点にセーブファイルを戻す。戻す前の現在状態も
    「復元前の状態」として履歴に積むので、復元自体もやり直せる。"""
    snap = history_dir(pid) / f"{seq:04d}.json.gz"
    if not snap.exists():
        raise FileNotFoundError(f"履歴が見つかりません: {pid}#{seq}")
    with gzip.open(snap, "rb") as f:
        proj = json.loads(f.read().decode("utf-8"))
    _push_history(pid, "復元前の状態")   # 復元をやり直せるように現状を退避
    proj["updated"] = _now()
    path = project_path(pid)
    with _lock:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(proj, f, ensure_ascii=False)
        tmp.replace(path)
    return proj


def save_project(proj: dict, label: str | None = None) -> dict:
    """一時ファイル経由で原子的に保存する（書き込み途中のクラッシュ対策）。

    label を渡すと、上書き前の状態を変更履歴に退避する（undo 用）。"""
    if label:
        _push_history(proj["id"], label)
    proj["updated"] = _now()
    path = project_path(proj["id"])
    with _lock:
        PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(proj, f, ensure_ascii=False)
        tmp.replace(path)
    return proj


def load_project(pid: str) -> dict:
    path = project_path(pid)
    if not path.exists():
        raise FileNotFoundError(f"プロジェクトが見つかりません: {pid}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# フレーム供給のようにリクエスト毎に呼ばれる読み取り専用パス用のキャッシュ。
# キーフレームが多いとJSONパースだけで数十msかかるため、mtimeで鮮度を判定する。
_load_cache: dict[str, tuple[int, dict]] = {}


def load_project_cached(pid: str) -> dict:
    """load_project のキャッシュ版。返り値は共有物なので変更しないこと。"""
    path = project_path(pid)
    if not path.exists():
        raise FileNotFoundError(f"プロジェクトが見つかりません: {pid}")
    mtime = path.stat().st_mtime_ns
    with _lock:
        hit = _load_cache.get(pid)
        if hit is not None and hit[0] == mtime:
            return hit[1]
    proj = load_project(pid)
    with _lock:
        _load_cache[pid] = (mtime, proj)
    return proj


def delete_project(pid: str) -> None:
    path = project_path(pid)
    if path.exists():
        path.unlink()


def list_projects() -> list[dict]:
    """一覧用のサマリーを更新日降順で返す。"""
    items = []
    for path in PROJECTS_DIR.glob("*.mvproj.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                p = json.load(f)
            t = p.get("targets") or {}
            items.append({
                "id": p["id"],
                "name": p["name"],
                "updated": p.get("updated", ""),
                "video_path": p["video"]["path"],
                "video_exists": Path(p["video"]["path"]).exists(),
                "frame_count": p["video"]["frame_count"],
                "fps": p["video"]["fps"],
                "person_count": len(p.get("persons", [])),
                # バッチ用: 保存済みの追跡設定サマリー
                "targets_mode": t.get("mode"),
                "targets_refs": len(t.get("ref_images") or []),
                "targets_region": t.get("region"),
            })
        except Exception:
            continue
    items.sort(key=lambda x: x["updated"], reverse=True)
    return items
