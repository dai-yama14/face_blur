"""
qc.py - 最終QC: ブラー漏れの検品と自動修正（バックグラウンドジョブ）

書き出しと同じ合成結果をサンプリング検品し、対象人物の顔が
生で見えているフレーム（漏れ）を見つけて自動修正する。

2段構えの検品（長尺動画対応）:
  Stage A（ローカル・無料）: ブラー合成後のフレームに SCRFD をかける。
    ブラー済みなら顔は検出できないはず → 検出できて ArcFace が本人と
    一致したら確実な漏れ（bboxが正確なのでそのまま修正シードになる）
  Stage B（Claude API・要APIキー）: 対象の keyframe が無い「未カバー区間」
    のサンプル + 定期的な抜き取り分だけを Claude に送り、
    横顔などローカル検出が拾えない漏れを検品する

修正: 漏れを区間にまとめ、SAM 2 で再伝播してキーフレームを追加 →
     修正した区間を再検品（最大 MAX_ROUNDS 周）。
     直らなかった漏れはセーブファイルに「要確認マーク」として保存し、
     タイムラインに赤く表示する。
"""
from __future__ import annotations

import math

import numpy as np

from . import autoclick, project as prj
from .detect import get_face_app, mouth_to_ellipse
from .render import apply_blurs, interp_ellipse
from .tracker import (SAM_MAX_SIDE, TrackJob, _cos, _jobs, _jobs_lock,
                      convert_ident_to_mouth, _Identity)
from .video import manager as video_manager

# 検品パラメータ
LEAK_SIM = 0.35          # StageA: 検出顔がこの類似度以上なら本人の漏れ
DET_SCORE_MIN = 0.55     # StageA: 全員モードでの顔検出スコア閾値
SPOT_CHECK_EVERY = 20    # StageB: カバー済み区間の抜き取り間隔（サンプル数）
QC_MAX_CALLS = 200       # StageB: 1周あたりの Claude 呼び出し上限
SPAN_MERGE_GAP_SEC = 1.5 # 漏れフレームをこの秒数以内なら同一区間にまとめる
FIX_MARGIN = 12          # 修正時に漏れ区間の前後へ広げるフレーム数
FIX_MAX_SPAN = 240       # 1回の修正窓の上限


def _point_covered(persons: list, keyframes: dict, fi: int,
                   x: float, y: float) -> bool:
    """点 (x, y) がいずれかの有効人物のブラー楕円に覆われているか（回転考慮）。"""
    for p in persons:
        if not p.get("enabled", True):
            continue
        ell = interp_ellipse(keyframes.get(p["id"], {}), fi)
        if ell is None:
            continue
        a = -math.radians(ell.get("angle", 0.0))
        dx, dy = x - ell["cx"], y - ell["cy"]
        lx = dx * math.cos(a) - dy * math.sin(a)
        ly = dx * math.sin(a) + dy * math.cos(a)
        if ((lx / max(1e-6, ell["rx"])) ** 2
                + (ly / max(1e-6, ell["ry"])) ** 2) <= 1.0:
            return True
    return False


class QCJob(TrackJob):
    """漏れ検品 + 自動修正ジョブ。TrackJob の SAM 2 機構を流用する。"""

    def __init__(self, pid: str, every_sec: float = 1.0,
                 use_api: bool = True, max_rounds: int = 2):
        super().__init__(pid, mode="all", ref_images=None, use_autoclick=use_api)
        self.every_sec = max(0.2, float(every_sec))
        self.max_rounds = max(1, int(max_rounds))
        self.stats = {"sampled": 0, "leaks_local": 0, "leaks_api": 0,
                      "api_calls": 0, "fixed_spans": 0, "marks": 0}

    def _track(self):
        proj = prj.load_project(self.pid)
        video = proj["video"]
        total = video["frame_count"]
        fps = video["fps"] or 30.0
        step = max(1, int(round(fps * self.every_sec)))

        # リファレンス（特定人物モード）の準備
        targets = proj.get("targets") or {}
        ref_emb = ref_crop = None
        if targets.get("mode") == "reference" and targets.get("ref_images"):
            self.mode = "reference"
            self.ref_images = targets["ref_images"]
            self.message = "リファレンス読み込み中..."
            ref_emb, ref_crop = self._load_reference()
            self.ref_crop = ref_crop

        # 修正の書き込み先: リファレンスモードならターゲット人物、
        # それ以外は最初の有効な人物
        person = self._find_fix_person(proj)

        rounds = 0
        remaining: list[dict] = []
        while rounds < self.max_rounds:
            self._check_cancel()
            leaks = self._scan_for_leaks(proj, step, total, fps, ref_emb,
                                         ref_crop, person)
            if not leaks:
                remaining = []
                break
            spans = self._merge_spans(leaks, fps)
            rounds += 1
            if rounds >= self.max_rounds or person is None:
                remaining = leaks
                break
            self._fix_spans(proj, person, spans, total)
            self.stats["fixed_spans"] += len(spans)
            proj = prj.load_project(self.pid)  # 修正後の状態を再読込

        # 直らなかった漏れは「要確認マーク」として保存
        proj = prj.load_project(self.pid)
        marks = [{"frame": lk["frame"],
                  "note": "ブラー漏れの可能性（" +
                          ("ローカル検出" if lk["source"] == "local"
                           else "Claude検品") + "）"}
                 for lk in remaining]
        proj["qc"] = {"marks": marks}
        prj.save_project(proj)
        self.stats["marks"] = len(marks)

        st = self.stats
        self.progress = 1.0
        self.state = "done"
        self.message = (
            f"QC完了: 検品{st['sampled']}フレーム / "
            f"漏れ検出 ローカル{st['leaks_local']}+API{st['leaks_api']} / "
            f"自動修正{st['fixed_spans']}区間 / 要確認{st['marks']}件"
            + (f"（API {st['api_calls']}回）" if st['api_calls'] else ""))

    # ── 検品 ──────────────────────────────────────────────────────────
    def _scan_for_leaks(self, proj, step, total, fps, ref_emb, ref_crop,
                        person) -> list[dict]:
        src = video_manager.get(proj["video"]["path"])
        app = get_face_app(with_rec=(ref_emb is not None))
        persons = proj["persons"]
        keyframes = proj["keyframes"]
        api_ok = (self.use_autoclick and autoclick.available())

        leaks: list[dict] = []
        sample_i = 0
        for fi in range(0, total, step):
            self._check_cancel()
            frame = src.get_frame(fi)
            if frame is None:
                break
            rendered = apply_blurs(frame, persons, keyframes, fi)
            sample_i += 1
            self.stats["sampled"] += 1

            # Stage A: ブラー後の映像で顔が検出できてしまう = 漏れ候補
            # （口元モードでは目元が見えるのは正常。口元中心が楕円で
            #   覆われているかで判定する）
            region = (person or {}).get("region", "face")
            faces = app.get(rendered)
            found_local = False
            for f in faces:
                if ref_emb is not None:
                    if _cos(f.normed_embedding, ref_emb) < LEAK_SIM:
                        continue  # 本人以外（第三者）は対象外
                elif float(f.det_score) < DET_SCORE_MIN:
                    continue
                bb = f.bbox.astype(float).tolist()
                if region == "mouth":
                    kps = (f.kps.astype(float).tolist()
                           if f.kps is not None else None)
                    m = mouth_to_ellipse(bb, kps)
                    if m is None:
                        continue
                    if _point_covered(persons, keyframes, fi,
                                      m["cx"], m["cy"]):
                        continue  # 口元は覆われている → 正常
                    bb = [m["cx"] - m["rx"], m["cy"] - m["ry"],
                          m["cx"] + m["rx"], m["cy"] + m["ry"]]
                leaks.append({"frame": fi, "bbox": bb, "source": "local"})
                self.stats["leaks_local"] += 1
                found_local = True
                self.live = {"frame": fi, "ellipses": [{
                    "cx": (bb[0] + bb[2]) / 2, "cy": (bb[1] + bb[3]) / 2,
                    "rx": (bb[2] - bb[0]) / 2, "ry": (bb[3] - bb[1]) / 2,
                    "angle": 0}]}
                break

            # Stage B: Claude 検品（未カバー区間 + 抜き取り）
            if (api_ok and not found_local and ref_emb is not None
                    and self.stats["api_calls"] < QC_MAX_CALLS):
                covered = (person is not None and interp_ellipse(
                    keyframes.get(person["id"], {}), fi) is not None)
                spot = (sample_i % SPOT_CHECK_EVERY == 0)
                if (not covered) or spot:
                    self.stats["api_calls"] += 1
                    self.message = (f"QC: Claude 検品中 frame {fi} "
                                    f"({self.stats['api_calls']}/{QC_MAX_CALLS})")
                    try:
                        hit = autoclick.check_leak(rendered, ref_crop,
                                                   region=region, pid=self.pid)
                    except Exception as e:
                        self.message = f"QC: API エラーのため Claude 検品を打切り: {e}"
                        api_ok = False
                        hit = None
                    if hit is not None:
                        leaks.append({"frame": fi,
                                      "point": [hit["x"], hit["y"]],
                                      "source": "api"})
                        self.stats["leaks_api"] += 1

            if sample_i % 20 == 0:
                self.progress = min(0.95, fi / max(1, total) * 0.9)
                if not found_local:
                    self.message = f"QC: 検品中 {fi}/{total}"
                    self.live = {"frame": fi, "ellipses": []}
        return leaks

    # ── 漏れフレーム → 区間へまとめる ─────────────────────────────────
    @staticmethod
    def _merge_spans(leaks: list[dict], fps: float) -> list[dict]:
        gap = int(fps * SPAN_MERGE_GAP_SEC)
        spans: list[dict] = []
        for lk in sorted(leaks, key=lambda x: x["frame"]):
            if spans and lk["frame"] - spans[-1]["end"] <= gap:
                spans[-1]["end"] = lk["frame"]
                spans[-1]["leaks"].append(lk)
            else:
                spans.append({"start": lk["frame"], "end": lk["frame"],
                              "leaks": [lk]})
        return spans

    # ── 自動修正: 漏れ区間を SAM 2 で再伝播 ──────────────────────────
    def _fix_spans(self, proj, person, spans, total):
        video = proj["video"]
        scale = min(1.0, SAM_MAX_SIDE / max(video["width"], video["height"]))
        kfs_all = prj.load_project(self.pid)["keyframes"]

        for si, span in enumerate(spans):
            self._check_cancel()
            w0 = max(0, span["start"] - FIX_MARGIN)
            w1 = min(total, span["end"] + FIX_MARGIN + 1)
            if w1 - w0 > FIX_MAX_SPAN:
                w1 = w0 + FIX_MAX_SPAN
            self.message = f"QC: 自動修正中 {si + 1}/{len(spans)} frame {w0}-{w1}"

            # シード: bbox（ローカル検出=正確）優先、なければ Claude の点
            prompts = {}
            for lk in span["leaks"][:3]:
                lf = lk["frame"] - w0
                if not (0 <= lf < w1 - w0):
                    continue
                if "bbox" in lk:
                    b = lk["bbox"]
                    mx = (b[2] - b[0]) * 0.25
                    my = (b[3] - b[1]) * 0.35
                    prompts[lf] = {"box": [(b[0] - mx) * scale,
                                           (b[1] - my) * scale,
                                           (b[2] + mx) * scale,
                                           (b[3] + my) * scale]}
                else:
                    prompts[lf] = {"point": [lk["point"][0] * scale,
                                             lk["point"][1] * scale]}
            if not prompts:
                continue

            ident = _Identity(1)
            self._run_window(video["path"], w0, w1, {1: prompts}, scale,
                             None, {1: ident})
            if person.get("region") == "mouth":
                refined, _conf = self._refine_mouth(
                    video["path"], ident,
                    margin_factor=1.0,
                    adjust=person.get("mouth_adjust"),
                    template=person.get("mouth_template"))
                convert_ident_to_mouth(ident, refined)

            # 手動キーフレームは保護しつつ追記
            proj2 = prj.load_project(self.pid)
            pkfs = proj2["keyframes"].setdefault(person["id"], {})
            for fs, kf in ident.keyframes.items():
                old = pkfs.get(fs)
                if old is not None and old.get("src") == "manual":
                    continue
                pkfs[fs] = kf
            prj.save_project(proj2)

    @staticmethod
    def _find_fix_person(proj) -> dict | None:
        persons = [p for p in proj["persons"] if p.get("enabled", True)]
        if not persons:
            return None
        for p in persons:
            if p.get("label") in ("ターゲット", "ターゲット（口元）") \
                    or p.get("engine") == "sam2":
                return p
        return persons[0]


def start_qc(pid: str, every_sec: float = 1.0, use_api: bool = True) -> dict:
    with _jobs_lock:
        job = _jobs.get(pid)
        if job and job.state == "running":
            return {"error": "ジョブが実行中です"}
        job = QCJob(pid, every_sec, use_api)
        _jobs[pid] = job
        job.start_job()
        return job.status()
