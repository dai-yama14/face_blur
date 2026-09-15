"""
export.py - mp4 書き出し（バックグラウンドジョブ）

render.apply_blurs でフレームを合成し、rawvideo を ffmpeg にパイプする。
音声は元動画からパススルー。NVENC が使えれば優先し、なければ libx264。
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import traceback
from datetime import datetime
from pathlib import Path

from . import project as prj
from .render import apply_blurs
from .video import manager as video_manager

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "output"

_nvenc_available: bool | None = None


def _has_nvenc() -> bool:
    global _nvenc_available
    if _nvenc_available is None:
        try:
            out = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=15,
            ).stdout
            _nvenc_available = "h264_nvenc" in out
        except Exception:
            _nvenc_available = False
    return _nvenc_available


class ExportJob:
    def __init__(self, pid: str, out_path: str | None = None):
        self.pid = pid
        self.out_path = out_path
        self.state = "running"
        self.progress = 0.0
        self.message = "初期化中..."
        self.result_path: str | None = None
        self.cancel_requested = False
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start_job(self):
        self._thread.start()

    def status(self) -> dict:
        return {
            "state": self.state, "progress": round(self.progress, 3),
            "message": self.message, "result_path": self.result_path,
        }

    def _run(self):
        try:
            self._export()
        except Exception as e:
            traceback.print_exc()
            self.state = "error"
            self.message = f"エラー: {e}"

    def _export(self):
        proj = prj.load_project(self.pid)
        video = proj["video"]
        src_path = video["path"]
        if not Path(src_path).exists():
            raise FileNotFoundError(f"元動画が見つかりません: {src_path}")
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg が見つかりません")

        w, h = video["width"], video["height"]
        fps = video["fps"]
        total = video["frame_count"]

        if self.out_path:
            out_path = Path(self.out_path)
        else:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = OUTPUT_DIR / f"{proj['name']}_blurred_{stamp}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        vcodec = (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "19"]
                  if _has_nvenc() else
                  ["-c:v", "libx264", "-preset", "medium", "-crf", "18"])

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            # 合成済みフレーム（rawvideo, stdin）
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
            "-r", f"{fps:.6f}", "-i", "pipe:0",
            # 音声用に元動画
            "-i", src_path,
            "-map", "0:v:0", "-map", "1:a:0?",
            *vcodec,
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-shortest",
            str(out_path),
        ]

        self.message = "書き出し中..."
        src = video_manager.get(src_path)
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stderr=subprocess.PIPE)
        try:
            for fi in range(total):
                if self.cancel_requested:
                    proc.stdin.close()
                    proc.terminate()
                    self.state = "cancelled"
                    self.message = "キャンセルされました"
                    out_path.unlink(missing_ok=True)
                    return
                frame = src.get_frame(fi)
                if frame is None:
                    break
                out = apply_blurs(frame, proj["persons"], proj["keyframes"], fi)
                proc.stdin.write(out.tobytes())
                self.progress = (fi + 1) / max(1, total)
            proc.stdin.close()
            rc = proc.wait()
            if rc != 0:
                err = proc.stderr.read().decode(errors="replace")[-800:]
                raise RuntimeError(f"ffmpeg 失敗 (exit={rc}): {err}")
        finally:
            if proc.poll() is None:
                proc.terminate()

        self.progress = 1.0
        self.state = "done"
        self.result_path = str(out_path)
        self.message = f"完了: {out_path}"


_jobs: dict[str, ExportJob] = {}
_jobs_lock = threading.Lock()


def start_export(pid: str, out_path: str | None = None) -> dict:
    with _jobs_lock:
        job = _jobs.get(pid)
        if job and job.state == "running":
            return {"error": "書き出しジョブが実行中です"}
        job = ExportJob(pid, out_path)
        _jobs[pid] = job
        job.start_job()
        return job.status()


def export_status(pid: str) -> dict | None:
    job = _jobs.get(pid)
    return job.status() if job else None


def cancel_export(pid: str) -> bool:
    job = _jobs.get(pid)
    if job and job.state == "running":
        job.cancel_requested = True
        return True
    return False
