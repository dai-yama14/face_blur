"""
main.py - 顔ブラーエディター バックエンド (FastAPI)

起動: python run_editor.py  →  http://localhost:8765
"""
from __future__ import annotations

from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import os

import json

from . import (ae_export, autoclick, batch, detect, export, landmarks3d,
               project as prj, qc, tracker)
from .render import apply_blurs
from .video import manager as video_manager

STATIC_DIR = Path(__file__).parent / "static"
# 新規プロジェクト作成時に動画を探すデフォルトディレクトリ
DEFAULT_VIDEO_DIR = Path.home() / "projects" / "movieedit2" / "input"
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}

app = FastAPI(title="movieedit2 - 顔ブラーエディター")


@app.on_event("startup")
def _warmup_landmarks3d():
    """3DDFA を起動時に読み込み、死んでいれば即座に大声で知らせる。

    遅延初期化のままだと、失敗は「追跡を50分回した後、口元の精度が落ちる」
    という形でしか現れず、しかも track_features 上は素材起因の CONF_FAILED と
    見分けがつかなかった（2026-07-17 に判明）。起動時に確定させる。
    """
    if landmarks3d.available():
        print("[startup] 3DDFA: OK（68点3Dランドマーク有効）")
        return
    st = landmarks3d.status()
    print("=" * 70)
    print("[startup] ⚠ 3DDFA が使用できません: " f"{st['last_error']}")
    print("[startup] ⚠ このまま追跡すると口元は 106点/5点フォールバックで")
    print("[startup] ⚠ 配置され、精度が落ちます。GET /api/health で確認可能。")
    print("=" * 70)


@app.get("/api/health")
def api_health():
    """依存モデルの生死。3DDFA が黙って死ぬのを検知するための窓口。"""
    return {"landmarks3d": landmarks3d.status()}


# ── プロジェクト（セーブファイル） ──────────────────────────────────────────

class CreateProjectBody(BaseModel):
    video_path: str
    name: str | None = None


@app.get("/api/projects")
def api_list_projects():
    return prj.list_projects()


@app.post("/api/projects")
def api_create_project(body: CreateProjectBody):
    try:
        return prj.create_project(body.video_path, body.name)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/projects/{pid}")
def api_get_project(pid: str):
    try:
        return prj.load_project(pid)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))


@app.put("/api/projects/{pid}")
def api_save_project(pid: str, body: dict, label: str | None = None):
    if body.get("id") != pid:
        raise HTTPException(400, "プロジェクトIDが一致しません")
    prj.save_project(body, label=label)
    return {"ok": True, "updated": body["updated"]}


@app.post("/api/projects/{pid}")
def api_save_project_beacon(pid: str, body: dict):
    """タブを閉じる際の sendBeacon 用（PUT と同じ保存処理）。"""
    return api_save_project(pid, body)


# ── 変更履歴（undo）─────────────────────────────────────────────────────────

@app.post("/api/projects/{pid}/snapshot")
def api_snapshot(pid: str, label: str = "編集"):
    """現在のセーブファイルを変更履歴に退避する。サーバー側でセーブファイルを
    上書きするジョブ（追跡・検出・再追跡）の直前に呼ぶ。"""
    prj._push_history(pid, label)
    return {"ok": True}


@app.get("/api/projects/{pid}/history")
def api_list_history(pid: str):
    return {"entries": prj.list_history(pid), "max": prj.HISTORY_MAX}


@app.post("/api/projects/{pid}/history/{seq}/restore")
def api_restore_history(pid: str, seq: int):
    try:
        proj = prj.restore_history(pid, seq)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"ok": True, "project": proj}


@app.delete("/api/projects/{pid}")
def api_delete_project(pid: str):
    import shutil
    prj.delete_project(pid)
    shutil.rmtree(prj.PROJECTS_DIR / f"{pid}.refs", ignore_errors=True)
    shutil.rmtree(prj.history_dir(pid), ignore_errors=True)
    autoclick.drop_project_usage(pid)
    return {"ok": True}


# ── 動画ファイル一覧 ─────────────────────────────────────────────────────────

@app.get("/api/videos")
def api_list_videos(dir: str | None = None):
    base = Path(dir).expanduser() if dir else DEFAULT_VIDEO_DIR
    if not base.is_dir():
        return {"dir": str(base), "videos": [], "error": "ディレクトリがありません"}
    videos = sorted(
        str(p) for p in base.iterdir()
        if p.suffix.lower() in VIDEO_EXTS and p.is_file()
    )
    return {"dir": str(base), "videos": videos}


# ── フレーム供給 ─────────────────────────────────────────────────────────────

@app.get("/api/projects/{pid}/frame/{index}")
def api_get_frame(pid: str, index: int, preview: int = 0, q: int = 82):
    try:
        # 毎フレーム呼ばれるためキャッシュ版でJSONパース（数十ms）を回避する
        proj = prj.load_project_cached(pid)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))

    try:
        src = video_manager.get(proj["video"]["path"])
    except ValueError as e:
        raise HTTPException(404, str(e))

    index = max(0, min(index, proj["video"]["frame_count"] - 1))
    frame = src.get_frame(index)
    if frame is None:
        raise HTTPException(404, f"フレーム {index} を読めません")

    if preview:
        frame = apply_blurs(frame, proj["persons"], proj["keyframes"], index)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise HTTPException(500, "JPEGエンコード失敗")
    return Response(content=buf.tobytes(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


# ── ランドマーク表示（検証用オーバーレイ） ──────────────────────────────────

@app.get("/api/projects/{pid}/landmarks/{index}")
def api_landmarks(pid: str, index: int):
    """指定フレームで顔検出を実測し、エディタが認識するランドマークを返す。

    プロジェクトには楕円しか永続化されないため、表示のたびにその場で SCRFD 検出
    ＋3DDFA_V2(68点) を回す（オンデマンド）。横顔の認識状態を目視確認するための
    デバッグ用オーバーレイに使う。
    """
    try:
        proj = prj.load_project_cached(pid)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    try:
        src = video_manager.get(proj["video"]["path"])
    except ValueError as e:
        raise HTTPException(404, str(e))
    index = max(0, min(index, proj["video"]["frame_count"] - 1))
    frame = src.get_frame(index)
    if frame is None:
        raise HTTPException(404, f"フレーム {index} を読めません")
    return {"frame": index, "faces": detect.landmarks_for_frame(frame)}


# ── 自動検出ジョブ ───────────────────────────────────────────────────────────

class DetectBody(BaseModel):
    start: int = 0
    end: int | None = None
    region: str = "face"


@app.post("/api/projects/{pid}/detect")
def api_detect(pid: str, body: DetectBody):
    prj.load_project(pid)  # 存在チェック
    res = detect.start_detect(pid, body.start, body.end, body.region)
    if "error" in res:
        raise HTTPException(409, res["error"])
    return res


@app.get("/api/projects/{pid}/detect/status")
def api_detect_status(pid: str):
    return detect.detect_status(pid) or {"state": "none"}


@app.post("/api/projects/{pid}/detect/cancel")
def api_detect_cancel(pid: str):
    return {"ok": detect.cancel_detect(pid)}


# ── 動画内クリックによる対象人物の登録 ──────────────────────────────────────

class PickFaceBody(BaseModel):
    frame: int
    x: float
    y: float


@app.post("/api/projects/{pid}/pick_face")
def api_pick_face(pid: str, body: PickFaceBody):
    """
    指定フレームの (x, y) にある顔を切り出してリファレンス画像として保存する。
    クリック位置を含む顔がなければ、十分近い顔を採用する。
    """
    import uuid

    proj = prj.load_project(pid)
    src = video_manager.get(proj["video"]["path"])
    frame = src.get_frame(body.frame)
    if frame is None:
        raise HTTPException(404, f"フレーム {body.frame} を読めません")

    faces = detect.get_face_app(False).get(frame)
    best = None
    # 1) クリック位置を含む顔（複数なら最小のもの＝最前面の可能性が高い）
    containing = [f for f in faces
                  if f.bbox[0] <= body.x <= f.bbox[2]
                  and f.bbox[1] <= body.y <= f.bbox[3]]
    if containing:
        best = min(containing, key=lambda f: (f.bbox[2] - f.bbox[0])
                   * (f.bbox[3] - f.bbox[1]))
    else:
        # 2) 顔中心がクリック位置に十分近いもの
        def dist(f):
            cx = (f.bbox[0] + f.bbox[2]) / 2
            cy = (f.bbox[1] + f.bbox[3]) / 2
            return ((cx - body.x) ** 2 + (cy - body.y) ** 2) ** 0.5
        near = [(dist(f), f) for f in faces]
        near = [x for x in near
                if x[0] < max(x[1].bbox[2] - x[1].bbox[0], 40) * 1.5]
        if near:
            best = min(near, key=lambda x: x[0])[1]
    if best is None:
        raise HTTPException(
            404, "クリック位置に顔を検出できませんでした。"
                 "顔がはっきり映っているフレームで、顔の中心をクリックしてください。")

    x1, y1, x2, y2 = [int(v) for v in best.bbox]
    mx, my = int((x2 - x1) * 0.35), int((y2 - y1) * 0.35)
    h, w = frame.shape[:2]
    crop = frame[max(0, y1 - my):min(h, y2 + my),
                 max(0, x1 - mx):min(w, x2 + mx)]

    refs_dir = prj.PROJECTS_DIR / f"{pid}.refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    out = refs_dir / f"ref_{uuid.uuid4().hex[:8]}.jpg"
    cv2.imwrite(str(out), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return {"path": str(out)}


@app.get("/api/thumb")
def api_thumb(path: str, size: int = 96):
    """リファレンス画像のサムネイルを返す（ローカルツール前提）。"""
    p = Path(path).expanduser()
    if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        raise HTTPException(400, "画像ファイルではありません")
    img = cv2.imread(str(p))
    if img is None:
        raise HTTPException(404, f"画像を読めません: {path}")
    h, w = img.shape[:2]
    s = size / max(h, w)
    if s < 1:
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


# ── SAM 2 高精度トラッキングジョブ ──────────────────────────────────────────

class TrackBody(BaseModel):
    mode: str = "all"                     # all=全員 / reference=リファレンス照合
    ref_images: list[str] = []
    use_autoclick: bool = True
    region: str = "face"                  # face=顔全体 / mouth=口元のみ


@app.post("/api/projects/{pid}/track")
def api_track(pid: str, body: TrackBody):
    prj.load_project(pid)
    if body.mode == "reference" and not body.ref_images:
        raise HTTPException(400, "リファレンスモードには画像パスが必要です")
    res = tracker.start_track(pid, body.mode, body.ref_images,
                              body.use_autoclick, body.region)
    if "error" in res:
        raise HTTPException(409, res["error"])
    return res


@app.get("/api/projects/{pid}/track/status")
def api_track_status(pid: str):
    return tracker.track_status(pid) or {"state": "none"}


@app.post("/api/projects/{pid}/track/cancel")
def api_track_cancel(pid: str):
    return {"ok": tracker.cancel_track(pid)}


# ── 失敗検知の特徴量測定（keyframe不変・レトロ用ブートストラップ） ──────────────

@app.post("/api/projects/{pid}/measure_features")
def api_measure_features(pid: str):
    prj.load_project(pid)
    res = tracker.start_measure_features(pid)
    if "error" in res:
        raise HTTPException(409, res["error"])
    return res


# ── 再シード（手動修正を引き継いで以降を再追跡） ────────────────────────────

class ReseedBody(BaseModel):
    person_id: str
    frame: int
    end: int | None = None    # 区間限定の再追跡（このフレームまで）


@app.post("/api/projects/{pid}/reseed")
def api_reseed(pid: str, body: ReseedBody):
    prj.load_project(pid)
    res = tracker.start_reseed(pid, body.person_id, body.frame, body.end)
    if "error" in res:
        raise HTTPException(409, res["error"])
    return res


@app.post("/api/projects/{pid}/step")
def api_step(pid: str, body: ReseedBody):
    """1フレームだけ SAM 2 伝播（Sキー用・同期）。"""
    prj.load_project(pid)
    res = tracker.step_propagate(pid, body.person_id, body.frame)
    if "error" in res:
        raise HTTPException(409, res["error"])
    return res


# ── 最終QC（ブラー漏れ検品 + 自動修正） ─────────────────────────────────────

class QcBody(BaseModel):
    every_sec: float = 1.0
    use_api: bool = True


@app.post("/api/projects/{pid}/qc")
def api_qc(pid: str, body: QcBody):
    prj.load_project(pid)
    res = qc.start_qc(pid, body.every_sec, body.use_api)
    if "error" in res:
        raise HTTPException(409, res["error"])
    return res


# ── バッチ処理（複数プロジェクトの連続追跡） ────────────────────────────────

class BatchBody(BaseModel):
    pids: list[str]
    qc: bool = True


@app.post("/api/batch")
def api_batch_start(body: BatchBody):
    items = []
    for pid in body.pids:
        try:
            prj.load_project(pid)
        except FileNotFoundError:
            raise HTTPException(404, f"プロジェクトが見つかりません: {pid}")
        items.append({"pid": pid})
    res = batch.start_batch(items, body.qc)
    if "error" in res:
        raise HTTPException(409, res["error"])
    return res


@app.get("/api/batch/status")
def api_batch_status():
    return batch.batch_status() or {"state": "none"}


@app.post("/api/batch/cancel")
def api_batch_cancel():
    return {"ok": batch.cancel_batch()}


# ── 実行環境情報 / API使用量 ─────────────────────────────────────────────────

@app.get("/api/config")
def api_config():
    return {
        "autoclick_available": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "sam2_available": tracker.SAM2_CKPT.exists(),
    }


@app.get("/api/usage")
def api_usage():
    return autoclick.get_usage()


# ── 書き出しジョブ ───────────────────────────────────────────────────────────

class ExportBody(BaseModel):
    out_path: str | None = None


@app.post("/api/projects/{pid}/export")
def api_export(pid: str, body: ExportBody):
    prj.load_project(pid)
    res = export.start_export(pid, body.out_path)
    if "error" in res:
        raise HTTPException(409, res["error"])
    return res


@app.get("/api/projects/{pid}/export/status")
def api_export_status(pid: str):
    return export.export_status(pid) or {"state": "none"}


@app.post("/api/projects/{pid}/export/cancel")
def api_export_cancel(pid: str):
    return {"ok": export.cancel_export(pid)}


@app.get("/api/projects/{pid}/ae_export.json")
def api_ae_export(
    pid: str,
    tol_px: float = ae_export.DEFAULT_TOL_PX,
    tol_ang: float = ae_export.DEFAULT_TOL_ANG,
):
    """After Effects 取り込み用 JSON（mouth_mask_importer2.jsx が読む v017 形式）を
    ダウンロードさせる。

    数万フレームを 1F=1マスクKF で AE に流すとハング→クラッシュするため、
    直線補間で許容誤差内の中間KFを間引く。tol_px/tol_ang をクエリで調整可能
    （例 ?tol_px=1.0&tol_ang=0.5）。tol_px=0&tol_ang=0 で間引き無効（全KF）。"""
    try:
        proj = prj.load_project(pid)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    data = ae_export.build_ae_json(proj, tol_px=tol_px, tol_ang=tol_ang)
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    fname = f"{proj.get('name', pid)}_ae.json"
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# ── フロントエンド ───────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
