"""
batch.py - 複数プロジェクトの連続処理（夜間バッチ）

各プロジェクトに保存済みの追跡設定（targets: mode / ref_images /
use_autoclick / region）で高精度追跡を1本ずつ順番に実行する。
オプションで追跡完了後に QC（漏れ検品+自動修正）も続けて実行する。
夜間の無人運用が前提なので、1本の失敗では止まらず記録して次へ進む。
"""
from __future__ import annotations

import threading
import time
from datetime import datetime

from . import project as prj, qc, tracker
from .video import manager as video_manager

_lock = threading.Lock()
_runner: "BatchRunner | None" = None


class BatchRunner:
    def __init__(self, items: list[dict], run_qc: bool):
        self.run_qc = run_qc
        self.state = "running"
        self.started = datetime.now().isoformat(timespec="seconds")
        self.finished: str | None = None
        self.cancel_requested = False
        self.items = []
        for it in items:
            self.items.append({
                "pid": it["pid"],
                "name": it.get("name") or it["pid"],
                "state": "pending",    # pending/tracking/qc/done/error/skipped
                "message": "",
            })
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def status(self) -> dict:
        items = []
        for it in self.items:
            d = dict(it)
            if it["state"] in ("tracking", "qc"):
                # 実行中ジョブ（追跡/QC共通で tracker._jobs に載る）の進捗を転記
                st = tracker.track_status(it["pid"])
                if st and st.get("state") == "running":
                    d["progress"] = st.get("progress", 0.0)
                    d["job_message"] = st.get("message", "")
            items.append(d)
        return {
            "state": self.state,
            "started": self.started,
            "finished": self.finished,
            "run_qc": self.run_qc,
            "items": items,
        }

    def _wait_job(self, pid: str) -> dict | None:
        """pid のジョブ（追跡 or QC）が終わるまで待つ。キャンセル要求は転送する。"""
        cancelled = False
        while True:
            st = tracker.track_status(pid)
            if st is None or st["state"] != "running":
                return st
            if self.cancel_requested and not cancelled:
                tracker.cancel_track(pid)
                cancelled = True
            time.sleep(2)

    def _run(self):
        for item in self.items:
            if self.cancel_requested:
                item["state"] = "skipped"
                continue
            video_path = None
            try:
                proj = prj.load_project(item["pid"])
                item["name"] = proj["name"]
                video_path = proj["video"]["path"]
                t = proj.get("targets") or {}
                mode = t.get("mode", "all")
                refs = t.get("ref_images") or []
                if mode == "reference" and not refs:
                    raise ValueError("リファレンス画像が未設定です"
                                     "（エディターで対象人物を登録してください）")

                item["state"] = "tracking"
                res = tracker.start_track(
                    item["pid"], mode, refs,
                    t.get("use_autoclick", True) is not False,
                    t.get("region", "face"),
                    low_ram=True)
                if "error" in res:
                    raise RuntimeError(res["error"])
                st = self._wait_job(item["pid"])
                if st is None or st["state"] != "done":
                    raise RuntimeError((st or {}).get("message", "追跡が完了しませんでした"))
                item["message"] = st.get("message", "")

                if self.run_qc and not self.cancel_requested:
                    item["state"] = "qc"
                    res = qc.start_qc(item["pid"], 1.0, True)
                    if "error" in res:
                        raise RuntimeError(res["error"])
                    st = self._wait_job(item["pid"])
                    if st is None or st["state"] != "done":
                        raise RuntimeError((st or {}).get("message", "QCが完了しませんでした"))
                    item["message"] += " ／ QC: " + st.get("message", "")

                item["state"] = "done"
            except Exception as e:
                item["state"] = "error"
                item["message"] = str(e)
            finally:
                # 1本ごとに動画のデコーダとフレームキャッシュ（最大256MB）を
                # 解放する。夜間に多本回してもRAMが積み上がらないようにする
                if video_path:
                    video_manager.close(video_path)
        self.state = "cancelled" if self.cancel_requested else "done"
        self.finished = datetime.now().isoformat(timespec="seconds")


def start_batch(items: list[dict], run_qc: bool = True) -> dict:
    global _runner
    with _lock:
        if _runner is not None and _runner.state == "running":
            return {"error": "バッチが実行中です"}
        if not items:
            return {"error": "対象プロジェクトがありません"}
        _runner = BatchRunner(items, run_qc)
        return _runner.status()


def batch_status() -> dict | None:
    return _runner.status() if _runner else None


def cancel_batch() -> bool:
    if _runner is not None and _runner.state == "running":
        _runner.cancel_requested = True
        return True
    return False
