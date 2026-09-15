"""
video.py - 動画フレーム供給

プロジェクトごとに cv2.VideoCapture をキャッシュし、ランダムアクセスを提供する。
連続再生（次フレーム読み）は seek せずに read だけで済ませて高速化する。
デコード済みフレームはメモリ上限付きの LRU にキャッシュし、エディターの
フレーム送り・戻り（同じ付近を行き来する操作）を高速化する。
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import cv2
import numpy as np

# デコード済みフレームキャッシュの合計メモリ上限（1080pで約40フレーム分）
CACHE_BYTES = 256 * 1024 * 1024
# 逆方向シーク時にまとめてデコードする長さ。seek は GOP 先頭からの再デコードで
# 1回 50ms 前後かかるため、手前から読み進めてキャッシュし戻り連打を1回で済ませる
BACK_RUN = 8
# この範囲内の前方ジャンプは seek せず読み進める（read は 1-2ms/フレーム）
FORWARD_RUN = 30


class VideoSource:
    def __init__(self, path: str):
        self.path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise ValueError(f"動画を開けません: {path}")
        self._next_pos: int | None = 0  # None = デコーダ位置が不明
        self._lock = threading.Lock()
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_max = 0  # 最初のフレームサイズから決定

    def _put(self, index: int, frame: np.ndarray):
        if self._cache_max == 0:
            self._cache_max = max(4, CACHE_BYTES // frame.nbytes)
        self._cache[index] = frame
        self._cache.move_to_end(index)
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

    def get_frame(self, index: int) -> np.ndarray | None:
        with self._lock:
            cached = self._cache.get(index)
            if cached is not None:
                self._cache.move_to_end(index)
                return cached.copy()  # キャッシュ破壊防止のためコピーを渡す

            if (self._next_pos is None or index < self._next_pos
                    or index > self._next_pos + FORWARD_RUN):
                start = (max(0, index - (BACK_RUN - 1))
                         if self._next_pos is not None and index < self._next_pos
                         else index)
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                self._next_pos = start

            frame = None
            while self._next_pos <= index:
                ok, f = self._cap.read()
                if not ok:
                    # seek 失敗時のリトライ（コーデックによる位置ズレ対策）
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, f = self._cap.read()
                    if not ok:
                        self._next_pos = None
                        return None
                    self._put(index, f)
                    self._next_pos = index + 1
                    return f.copy()
                self._put(self._next_pos, f)
                frame = f
                self._next_pos += 1
            return frame.copy() if frame is not None else None

    def release(self):
        with self._lock:
            self._cap.release()
            self._cache.clear()


class VideoManager:
    """path -> VideoSource のキャッシュ。"""

    def __init__(self):
        self._sources: dict[str, VideoSource] = {}
        self._lock = threading.Lock()

    def get(self, path: str) -> VideoSource:
        with self._lock:
            src = self._sources.get(path)
            if src is None:
                src = VideoSource(path)
                self._sources[path] = src
            return src

    def close(self, path: str):
        """1本だけ閉じてキャッシュを解放する（バッチで動画を渡り歩く際のRAM対策）。"""
        with self._lock:
            src = self._sources.pop(path, None)
        if src is not None:
            src.release()

    def close_all(self):
        with self._lock:
            for src in self._sources.values():
                src.release()
            self._sources.clear()


manager = VideoManager()
