/* movieedit2 顔ブラーエディター フロントエンド */
"use strict";

const $ = (id) => document.getElementById(id);

// ── 状態 ─────────────────────────────────────────────────────────
let proj = null;          // プロジェクト全体（セーブファイルと同一構造）
let curFrame = 0;
let selectedPid = null;
let previewMode = false;
let showMask = true;      // マスク（赤）楕円の表示
let showDetect = true;    // 追跡中の検出範囲（オレンジ）の表示
let showLandmarks = false; // 認識中の顔ランドマーク（68/106/5点）の表示
let landmarksData = null;  // 現在表示中フレームのランドマーク {frame, faces}
let landmarksSeq = 0;      // ランドマーク取得の競合ガード
let playing = false;
let playTimer = null;
let addingPerson = false;
let dirty = false;
let saveTimer = null;
let loadSeq = 0;          // フレーム読み込みの競合ガード
let frameLoading = false;
let frameImg = new Image();

const HOLD_FRAMES = 12;   // render.py と同じ値（補間の見た目を一致させる）
const PARK_MIN_GAP = 30;      // render.py と同じ（長い無検出区間で端なら画面外へ）
const PARK_EDGE_FACTOR = 1.5; // render.py と同じ（端キーフレーム判定）

// ── API ──────────────────────────────────────────────────────────
async function api(path, opts = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return res.json();
}

// ── キーフレーム補間（render.py の interp_ellipse と同一ロジック） ──
function lerpAngle(a, b, t) {
  const diff = ((b - a + 180) % 360 + 360) % 360 - 180;
  return a + diff * t;
}

// 楕円中心が最寄りの画面端に近い（フレームアウトしかけ）か（render.py と同一）
function nearEdge(kf) {
  const vw = proj.video.width, vh = proj.video.height;
  const rmax = Math.max(kf.rx, kf.ry);
  const d = Math.min(kf.cx, vw - kf.cx, kf.cy, vh - kf.cy);
  return d < rmax * PARK_EDGE_FACTOR;
}

// 直近の楕円を最寄り画面端の外へ退避（render.py の _parked_ellipse と同一）
function parkedEllipse(ref) {
  const vw = proj.video.width, vh = proj.video.height;
  const rmax = Math.max(ref.rx, ref.ry);
  const margin = rmax * 1.3 + 24;
  const dists = { left: ref.cx, right: vw - ref.cx, top: ref.cy, bottom: vh - ref.cy };
  const edge = Object.keys(dists).reduce((a, b) => (dists[a] <= dists[b] ? a : b));
  let cx = ref.cx, cy = ref.cy;
  if (edge === "left") cx = -margin;
  else if (edge === "right") cx = vw + margin;
  else if (edge === "top") cy = -margin;
  else cy = vh + margin;
  return { ...ref, cx, cy, visible: true, parked: true };
}

function interpEllipse(kfs, frame) {
  if (!kfs) return null;
  const exact = kfs[String(frame)];
  if (exact !== undefined) {
    // 不在マーク: 編集用の枠は画面外パーキング（掴んで戻せる）
    if (exact.absent) return proj ? parkedEllipse(exact) : null;
    return exact.visible === false ? null : exact;
  }

  const frames = Object.keys(kfs).map(Number).sort((a, b) => a - b);
  let prevF = null, nextF = null;
  for (const f of frames) {
    if (f < frame) prevF = f;
    else if (f > frame) { nextF = f; break; }
  }
  const pk = prevF !== null ? kfs[String(prevF)] : null;
  const nk = nextF !== null ? kfs[String(nextF)] : null;
  const pOk = pk && pk.visible !== false;
  const nOk = nk && nk.visible !== false;

  if (pOk && nOk) {
    const gap = nextF - prevF;
    // 長い無検出区間 かつ 前後ともフレーム端 → 画面外へ退避（両端は HOLD ぶん保持）
    if (proj && gap >= PARK_MIN_GAP
        && frame - prevF > HOLD_FRAMES && nextF - frame > HOLD_FRAMES
        && nearEdge(pk) && nearEdge(nk)) {
      const ref = (frame - prevF) <= (nextF - frame) ? pk : nk;
      return parkedEllipse(ref);
    }
    const t = (frame - prevF) / (nextF - prevF);
    return {
      cx: pk.cx + (nk.cx - pk.cx) * t,
      cy: pk.cy + (nk.cy - pk.cy) * t,
      rx: pk.rx + (nk.rx - pk.rx) * t,
      ry: pk.ry + (nk.ry - pk.ry) * t,
      angle: lerpAngle(pk.angle || 0, nk.angle || 0, t),
      visible: true,
    };
  }
  if (pOk && frame - prevF <= HOLD_FRAMES) return { ...pk, visible: true };
  if (nOk && nextF - frame <= HOLD_FRAMES) return { ...nk, visible: true };
  // 検出できない区間: 画面外パーキング（render.py と同一ロジック）
  const ref = pOk ? pk : (nOk ? nk : null);
  if (ref && proj) return parkedEllipse(ref);
  return null;
}

// ── ブラー濃度の補間（render.py の strength_at と同一ロジック） ──
function interpScalar(kfs, frame) {
  if (!kfs || !Object.keys(kfs).length) return null;
  const exact = kfs[String(frame)];
  if (exact !== undefined) return Number(exact);
  const frames = Object.keys(kfs).map(Number).sort((a, b) => a - b);
  let prevF = null, nextF = null;
  for (const f of frames) {
    if (f < frame) prevF = f;
    else { nextF = f; break; }
  }
  if (prevF !== null && nextF !== null) {
    const t = (frame - prevF) / (nextF - prevF);
    const a = Number(kfs[String(prevF)]), b = Number(kfs[String(nextF)]);
    return a + (b - a) * t;
  }
  return Number(kfs[String(prevF !== null ? prevF : nextF)]);
}

function strengthAt(person, frame) {
  const v = interpScalar(person.strength_kfs, frame);
  return v !== null ? v : (person.blur?.strength ?? 30);
}

// 楕円ローカル座標系との相互変換（回転対応）
function toLocal(pos, ell) {
  const a = -(ell.angle || 0) * Math.PI / 180;
  const dx = pos.x - ell.cx, dy = pos.y - ell.cy;
  return {
    x: dx * Math.cos(a) - dy * Math.sin(a),
    y: dx * Math.sin(a) + dy * Math.cos(a),
  };
}

function fromLocal(local, ell) {
  const a = (ell.angle || 0) * Math.PI / 180;
  return {
    x: ell.cx + local.x * Math.cos(a) - local.y * Math.sin(a),
    y: ell.cy + local.x * Math.sin(a) + local.y * Math.cos(a),
  };
}

// ── 保存（オートセーブ） ─────────────────────────────────────────
// 次の保存に付ける変更履歴ラベル。保存時に「上書き直前の状態」がこの名前で
// 履歴に退避され、後から任意の地点に戻せる（undo）。
let pendingLabel = null;

function markDirty(label) {
  dirty = true;
  pendingLabel = label || pendingLabel || "編集";
  invalidateIssues();
  $("save-status").textContent = "● 未保存の変更";
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => saveProject(), 1200);
}

async function saveProject(labelArg) {
  if (!proj || !dirty) return;
  const label = labelArg || pendingLabel;
  pendingLabel = null;
  proj.ui = proj.ui || {};
  proj.ui.frame = curFrame;
  try {
    const q = label ? `?label=${encodeURIComponent(label)}` : "";
    await api(`/api/projects/${proj.id}${q}`, {
      method: "PUT",
      body: JSON.stringify(proj),
    });
    dirty = false;
    const t = new Date().toLocaleTimeString();
    $("save-status").textContent = `保存済み ${t}`;
  } catch (e) {
    $("save-status").textContent = `保存失敗: ${e.message}`;
  }
}

// サーバー側でセーブファイルを上書きするジョブ（追跡・検出・再追跡）の直前に
// 現在の状態を変更履歴へ退避する。失敗しても編集自体は止めない。
async function pushSnapshot(label) {
  if (!proj) return;
  try {
    await api(`/api/projects/${proj.id}/snapshot?label=${encodeURIComponent(label)}`,
              { method: "POST" });
  } catch (e) { /* 履歴退避の失敗は無視 */ }
}

// ── 変更履歴（undo）UI ───────────────────────────────────────────
async function openHistory() {
  if (!proj) return;
  const list = $("history-list");
  list.innerHTML = '<p class="muted">読み込み中...</p>';
  $("history-modal").classList.remove("hidden");
  try {
    const res = await api(`/api/projects/${proj.id}/history`);
    renderHistory(res.entries || [], res.max || 50);
  } catch (e) {
    list.innerHTML = `<p class="muted">履歴の取得に失敗しました: ${e.message}</p>`;
  }
}

function renderHistory(entries, max) {
  const list = $("history-list");
  if (!entries.length) {
    list.innerHTML =
      `<p class="muted">まだ履歴がありません。編集すると、操作ごとに直前の`
      + `状態がここに記録されます（最大${max}件）。</p>`;
    return;
  }
  list.innerHTML = "";
  for (const e of entries) {
    const row = document.createElement("div");
    row.className = "history-row";
    const ts = e.ts ? e.ts.replace("T", " ").slice(5) : "";
    const meta = document.createElement("div");
    meta.className = "history-meta";
    const lab = document.createElement("span");
    lab.className = "history-label";
    lab.textContent = e.label || "(無題)";
    const sub = document.createElement("span");
    sub.className = "muted history-ts";
    sub.textContent = ts;
    meta.appendChild(lab); meta.appendChild(sub);
    const btn = document.createElement("button");
    btn.textContent = "この直前に戻す";
    btn.addEventListener("click", () => restoreTo(e.seq, e.label || "(無題)"));
    row.appendChild(meta); row.appendChild(btn);
    list.appendChild(row);
  }
}

async function restoreTo(seq, label) {
  if (!confirm(`「${label}」の直前の状態に戻します。\n`
      + `いまの状態は「復元前の状態」として履歴に残るので、やり直せます。\n`
      + `戻しますか？`)) return;
  clearTimeout(saveTimer);      // 復元後に古い自動保存が上書きしないよう止める
  clearTimeout(uiSaveTimer);
  dirty = false;
  pendingLabel = null;
  try {
    const res = await api(`/api/projects/${proj.id}/history/${seq}/restore`,
                          { method: "POST" });
    $("history-modal").classList.add("hidden");
    await openProject(res.project.id);   // 復元後の状態でエディタを読み直す
    $("detect-status").textContent = `「${label}」の直前に復元しました`;
  } catch (e) {
    alert(`復元に失敗しました: ${e.message}`);
  }
}

// ── フレーム表示 ─────────────────────────────────────────────────
// 編集でプレビューの見た目が変わるたびに editVersion を上げてキャッシュを無効化する
let editVersion = 0;
const imgCache = new Map();   // key -> {img, ready}
const IMG_CACHE_MAX = 40;

function frameUrl(n) {
  return `/api/projects/${proj.id}/frame/${n}?preview=${previewMode ? 1 : 0}&v=${editVersion}`;
}

function frameKey(n) {
  return `${n}|${previewMode ? 1 : 0}|${editVersion}`;
}

// フレーム画像の取得を開始し（キャッシュ済みなら即座に）、エントリを返す
function fetchFrame(n) {
  const key = frameKey(n);
  let e = imgCache.get(key);
  if (e) return e;
  const img = new Image();
  e = { img, ready: false };
  img.addEventListener("load", () => { e.ready = true; }, { once: true });
  img.addEventListener("error", () => { imgCache.delete(key); }, { once: true });
  img.src = frameUrl(n);
  imgCache.set(key, e);
  while (imgCache.size > IMG_CACHE_MAX) {
    imgCache.delete(imgCache.keys().next().value);
  }
  return e;
}

// 現在フレームの前後を先読みして、フレーム送り・戻りを即時表示にする
function prefetchAround(n) {
  for (const d of [1, -1, 2, -2]) {
    const m = n + d;
    if (m >= 0 && m < proj.video.frame_count) fetchFrame(m);
  }
}

function loadFrame(n, cb) {
  const seq = ++loadSeq;
  const e = fetchFrame(n);
  const show = () => {
    if (seq !== loadSeq) return;
    frameLoading = false;
    frameImg = e.img;
    draw();
    if (cb) cb();
    prefetchAround(n);
  };
  if (e.ready) {
    show();
  } else {
    frameLoading = true;
    e.img.addEventListener("load", show, { once: true });
    e.img.addEventListener("error", () => {
      if (seq === loadSeq) frameLoading = false;
    }, { once: true });
  }
}

function setFrame(n, cb) {
  if (!proj) return;
  curFrame = Math.max(0, Math.min(n, proj.video.frame_count - 1));
  $("seek").value = curFrame;
  updateFrameInfo();
  if (selectedPid) updateStrengthUI();
  loadFrame(curFrame, cb);
  fetchLandmarks(curFrame);   // オンなら現フレームの認識点を取り直す
  tlOverlay();   // タイムラインの再生ヘッドだけ更新（軽量）
  markDirtyUiOnly();
}

// 現フレームのランドマークをサーバーで実測して取得する（表示オン時のみ）。
// 再生中は毎フレームの検出が重いのでスキップ（停止・コマ送りで使う検証機能）。
async function fetchLandmarks(n) {
  if (!showLandmarks || !proj || playing) { landmarksData = null; return; }
  const seq = ++landmarksSeq;
  try {
    const d = await api(`/api/projects/${proj.id}/landmarks/${n}`);
    if (seq !== landmarksSeq) return;   // 取得中に別フレームへ移動していたら破棄
    landmarksData = d;
    draw();
  } catch (e) {
    if (seq === landmarksSeq) { landmarksData = null; draw(); }
  }
}

// UI状態（現在フレーム）の変更は頻繁なので、編集とは別に静かに保存を予約する
let uiSaveTimer = null;
function markDirtyUiOnly() {
  clearTimeout(uiSaveTimer);
  uiSaveTimer = setTimeout(() => { dirty = true; saveProject(); }, 4000);
}

function refreshFrame() {
  editVersion++;       // 編集後は全キャッシュを無効化して取り直す
  imgCache.clear();
  loadFrame(curFrame);
}

function updateFrameInfo() {
  const fps = proj.video.fps;
  const sec = curFrame / fps;
  const mm = String(Math.floor(sec / 60)).padStart(2, "0");
  const ss = (sec % 60).toFixed(2).padStart(5, "0");
  $("frame-info").textContent =
    `/ ${proj.video.frame_count - 1}  (${mm}:${ss})`;
  const fj = $("frame-jump");
  if (document.activeElement !== fj) fj.value = curFrame;
}

// ── キャンバス描画 ───────────────────────────────────────────────
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

// ビュー変換（ズーム / パン）。座標変換: 画面 = ビデオ座標 * viewScale() + view.x/y
let view = { zoom: 1, x: 0, y: 0 };

function canvasScale() {
  return canvas.width / proj.video.width;
}

function viewScale() {
  return canvasScale() * view.zoom;
}

function resetView() {
  view = { zoom: 1, x: 0, y: 0 };
  draw();
}

function draw() {
  if (!proj) return;
  const vw = proj.video.width, vh = proj.video.height;
  const wrap = $("canvas-wrap");
  const scale = Math.min(wrap.clientWidth / vw, wrap.clientHeight / vh, 1.5);
  canvas.width = Math.round(vw * scale);
  canvas.height = Math.round(vh * scale);

  const s = viewScale();
  const ox = view.x, oy = view.y;

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (frameImg.complete && frameImg.naturalWidth > 0) {
    ctx.drawImage(frameImg, ox, oy, vw * s, vh * s);
  }

  for (const p of proj.persons) {
    if (!showMask) break;   // マスク（赤）枠を非表示
    const ell = interpEllipse(proj.keyframes[p.id], curFrame);
    const sel = p.id === selectedPid;
    if (!ell) {
      if (sel) drawNoEllipseHint(p);
      continue;
    }
    const rot = (ell.angle || 0) * Math.PI / 180;
    ctx.save();
    ctx.strokeStyle = ell.parked ? "#8b91a0" : p.color;
    ctx.lineWidth = sel ? 2.5 : 1.5;
    ctx.setLineDash(ell.parked ? [4, 4] : (p.enabled === false ? [6, 4] : []));
    ctx.beginPath();
    ctx.ellipse(ell.cx * s + ox, ell.cy * s + oy, ell.rx * s, ell.ry * s,
                rot, 0, Math.PI * 2);
    ctx.stroke();

    // ラベル
    ctx.setLineDash([]);
    ctx.fillStyle = p.color;
    ctx.font = "12px sans-serif";
    const rmax = Math.max(ell.rx, ell.ry);
    ctx.fillText(p.label + (ell.parked ? "（未検出: 画面外に待機中）" : ""),
                 (ell.cx - rmax) * s + ox, (ell.cy - rmax) * s + oy - 6);

    if (sel) {
      // リサイズハンドル（回転済みの上下左右）
      ctx.fillStyle = "#ffffff";
      for (const [hx, hy] of handlePositions(ell)) {
        ctx.fillRect(hx * s + ox - 4, hy * s + oy - 4, 8, 8);
      }
      // 回転ハンドル（楕円の上方に ◎ + 接続線）
      const rp = rotHandlePos(ell);
      const top = fromLocal({ x: 0, y: -ell.ry }, ell);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(top.x * s + ox, top.y * s + oy);
      ctx.lineTo(rp.x * s + ox, rp.y * s + oy);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(rp.x * s + ox, rp.y * s + oy, 6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(rp.x * s + ox, rp.y * s + oy, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
    }
    ctx.restore();
  }

  // 検出枠（オレンジ）: 各フレームの生の顔検出領域を常時表示。
  // 追跡実行中は下のライブ表示を優先し、二重描画を避ける
  if (showDetect && !liveTrack && proj.detections) {
    ctx.save();
    ctx.strokeStyle = "#ff9f1c";
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 5]);
    for (const p of proj.persons) {
      const e = interpEllipse(proj.detections[p.id], curFrame);
      if (!e) continue;
      ctx.beginPath();
      ctx.ellipse(e.cx * s + ox, e.cy * s + oy, e.rx * s, e.ry * s,
                  (e.angle || 0) * Math.PI / 180, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  }

  // 追跡ジョブのライブ表示（現在検出中の領域をオレンジ破線で表示）
  if (showDetect && liveTrack && liveTrack.ellipses
      && Math.abs(liveTrack.frame - curFrame) <= 2) {
    ctx.save();
    ctx.strokeStyle = "#ff9f1c";
    ctx.lineWidth = 2;
    ctx.setLineDash([8, 5]);
    for (const e of liveTrack.ellipses) {
      ctx.beginPath();
      ctx.ellipse(e.cx * s + ox, e.cy * s + oy, e.rx * s, e.ry * s,
                  (e.angle || 0) * Math.PI / 180, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.fillStyle = "#ff9f1c";
    ctx.font = "bold 13px sans-serif";
    ctx.fillText("🔍 検出中 frame " + liveTrack.frame, 12, canvas.height - 12);
    ctx.restore();
  }

  // 認識中のランドマーク（検証用）: 68点=シアン / 106点=マゼンタ / 5点(kps)=黄。
  // source の段を明るく強調し、bboxとどの段が採用中かをラベル表示する。
  if (showLandmarks && landmarksData && landmarksData.frame === curFrame) {
    drawLandmarks(s, ox, oy);
  }
}

function drawLandmarks(s, ox, oy) {
  const dot = (pts, color, r) => {
    ctx.fillStyle = color;
    for (const [px, py] of pts) {
      ctx.beginPath();
      ctx.arc(px * s + ox, py * s + oy, r, 0, Math.PI * 2);
      ctx.fill();
    }
  };
  ctx.save();
  for (const f of landmarksData.faces) {
    // 検出bbox（SCRFDが顔を捉えている証拠。横顔でも出る）
    const [x1, y1, x2, y2] = f.bbox;
    ctx.strokeStyle = "#4ade80";
    ctx.lineWidth = 1.5;
    ctx.setLineDash([]);
    ctx.strokeRect(x1 * s + ox, y1 * s + oy, (x2 - x1) * s, (y2 - y1) * s);

    // 採用されない段は暗め、採用中(source)の段は明るく
    const dim = f.source !== "lmk106";
    if (f.lmk106) dot(f.lmk106, dim ? "#7a2f6b" : "#ff5cc8", 1.3);
    if (f.lmk68) dot(f.lmk68, f.source === "lmk68" ? "#00e5ff" : "#0a6b78", 2.0);
    if (f.kps) dot(f.kps, f.source === "kps" ? "#ffe600" : "#8a8300", 3.2);

    // ラベル: 採用中の段と検出スコア
    const label = { lmk68: "68点(3DDFA)", lmk106: "106点",
                    kps: "5点(kps)" }[f.source] || f.source;
    ctx.fillStyle = "#4ade80";
    ctx.font = "11px sans-serif";
    ctx.fillText(`${label}  score ${f.det_score.toFixed(2)}`,
                 x1 * s + ox, y1 * s + oy - 4);
  }
  ctx.restore();
}

function rotHandlePos(ell) {
  const off = 26 / viewScale();  // 画面上で常に26pxの距離
  return fromLocal({ x: 0, y: -ell.ry - off }, ell);
}

function drawNoEllipseHint(p) {
  ctx.save();
  ctx.fillStyle = p.color;
  ctx.font = "13px sans-serif";
  ctx.fillText(`${p.label}: このフレームは非表示（キーフレーム追加で表示）`, 12, 24);
  ctx.restore();
}

function handlePositions(ell) {
  // 回転を反映したハンドル位置（右・左・上・下の順）
  return [
    fromLocal({ x: ell.rx, y: 0 }, ell),
    fromLocal({ x: -ell.rx, y: 0 }, ell),
    fromLocal({ x: 0, y: -ell.ry }, ell),
    fromLocal({ x: 0, y: ell.ry }, ell),
  ].map((p) => [p.x, p.y]);
}

// ── キャンバス操作（ドラッグ移動・リサイズ・人物追加） ───────────
let dragMode = null;   // "move" | "r" | "l" | "t" | "b"
let dragEll = null;    // ドラッグ中の楕円（ビデオ座標）
let dragStart = null;

function toVideoCoords(ev) {
  const rect = canvas.getBoundingClientRect();
  const s = viewScale();
  return {
    x: (ev.clientX - rect.left - view.x) / s,
    y: (ev.clientY - rect.top - view.y) / s,
  };
}

function hitTest(pos) {
  // 選択中人物のハンドル優先
  if (selectedPid) {
    const ell = interpEllipse(proj.keyframes[selectedPid], curFrame);
    if (ell) {
      const s = viewScale();
      const tol = 8 / s;
      // 回転ハンドル
      const rp = rotHandlePos(ell);
      if (Math.hypot(pos.x - rp.x, pos.y - rp.y) < tol) {
        return { pid: selectedPid, mode: "rotate", ell };
      }
      const hs = handlePositions(ell);
      const modes = ["r", "l", "t", "b"];
      for (let i = 0; i < 4; i++) {
        if (Math.abs(pos.x - hs[i][0]) < tol && Math.abs(pos.y - hs[i][1]) < tol) {
          return { pid: selectedPid, mode: modes[i], ell };
        }
      }
    }
  }
  // 楕円内クリック（後に描いたもの＝リスト後方を優先。回転を考慮）
  for (let i = proj.persons.length - 1; i >= 0; i--) {
    const p = proj.persons[i];
    const ell = interpEllipse(proj.keyframes[p.id], curFrame);
    if (!ell) continue;
    const lc = toLocal(pos, ell);
    const dx = lc.x / ell.rx, dy = lc.y / ell.ry;
    if (dx * dx + dy * dy <= 1) return { pid: p.id, mode: "move", ell };
  }
  return null;
}

// ── ズーム / パン ────────────────────────────────────────────────
let panning = false;
let panStart = null;

canvas.addEventListener("wheel", (ev) => {
  if (!proj) return;
  ev.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
  const old = view.zoom;
  view.zoom = Math.min(10, Math.max(0.2,
    old * (ev.deltaY < 0 ? 1.15 : 1 / 1.15)));
  // カーソル位置を不動点にしてズーム
  const r = view.zoom / old;
  view.x = mx - (mx - view.x) * r;
  view.y = my - (my - view.y) * r;
  draw();
}, { passive: false });

canvas.addEventListener("mousedown", (ev) => {
  if (!proj) return;

  // ホイール（中）ボタンドラッグでパン
  if (ev.button === 1) {
    ev.preventDefault();
    panning = true;
    panStart = { x: ev.clientX - view.x, y: ev.clientY - view.y };
    return;
  }
  if (ev.button !== 0) return;

  const pos = toVideoCoords(ev);

  if (pickingTarget) {
    pickTargetAt(pos);
    setPickingMode(false);
    return;
  }
  if (addingPerson) {
    addPersonAt(pos);
    setAddingMode(false);
    return;
  }

  const hit = hitTest(pos);
  if (hit) {
    if (selectedPid !== hit.pid) selectPerson(hit.pid);
    dragMode = hit.mode;
    dragEll = { cx: hit.ell.cx, cy: hit.ell.cy, rx: hit.ell.rx,
                ry: hit.ell.ry, angle: hit.ell.angle || 0 };
    dragStart = pos;
  } else {
    selectPerson(null);
  }
});

window.addEventListener("mousemove", (ev) => {
  if (panning) {
    view.x = ev.clientX - panStart.x;
    view.y = ev.clientY - panStart.y;
    draw();
    return;
  }
  if (!dragMode || !proj) return;
  const pos = toVideoCoords(ev);
  const dx = pos.x - dragStart.x;
  const dy = pos.y - dragStart.y;
  const e = { ...dragEll };
  if (dragMode === "move") {
    e.cx += dx; e.cy += dy;
  } else if (dragMode === "rotate") {
    // ハンドルは楕円の「上」方向 → マウス方向 + 90度 が新しい角度
    e.angle = Math.atan2(pos.y - dragEll.cy, pos.x - dragEll.cx)
              * 180 / Math.PI + 90;
  } else {
    // リサイズは回転済みローカル軸に沿って行う
    const lc = toLocal(pos, dragEll);
    if (dragMode === "r" || dragMode === "l") e.rx = Math.max(4, Math.abs(lc.x));
    else e.ry = Math.max(4, Math.abs(lc.y));
  }

  setManualKeyframe(selectedPid, curFrame, e, /*quiet=*/true);
  draw();
});

window.addEventListener("mouseup", () => {
  panning = false;
  if (dragMode) {
    dragMode = null;
    markDirty();
    drawTimeline();
    if (previewMode) refreshFrame();
  }
});

function setManualKeyframe(pid, frame, ell, quiet = false) {
  if (!proj.keyframes[pid]) proj.keyframes[pid] = {};
  proj.keyframes[pid][String(frame)] = {
    cx: Math.round(ell.cx * 10) / 10,
    cy: Math.round(ell.cy * 10) / 10,
    rx: Math.round(ell.rx * 10) / 10,
    ry: Math.round(ell.ry * 10) / 10,
    angle: Math.round((ell.angle || 0) * 10) / 10,
    visible: ell.visible !== false,
    src: "manual",
  };
  if (!quiet) { markDirty(); draw(); drawTimeline(); }
}

// ── 人物管理 ─────────────────────────────────────────────────────
const COLORS = ["#ff5555", "#50b0ff", "#5fd068", "#f5a623",
                "#c678dd", "#2ec4b6", "#ff7ab2", "#d4c05a"];

function addPersonAt(pos) {
  const idx = proj.persons.length;
  const p = {
    id: `p${idx + 1}_${Math.random().toString(16).slice(2, 8)}`,
    label: `人物${idx + 1}`,
    color: COLORS[idx % COLORS.length],
    enabled: true,
    blur: { type: "gaussian", strength: 30, feather: 0.25 },
  };
  proj.persons.push(p);
  const r = proj.video.width * 0.05;
  setManualKeyframe(p.id, curFrame, { cx: pos.x, cy: pos.y, rx: r, ry: r * 1.3 });
  selectPerson(p.id);
  renderPersonList();
}

function selectPerson(pid) {
  selectedPid = pid;
  renderPersonList();
  renderBlurPanel();
  draw();
}

function renderPersonList() {
  const wrap = $("person-list");
  wrap.innerHTML = "";
  if (!proj.persons.length) {
    wrap.innerHTML = '<p class="muted">人物がいません。「自動検出」か「＋追加」で作成してください。</p>';
    return;
  }
  for (const p of proj.persons) {
    const div = document.createElement("div");
    div.className = "person-item" + (p.id === selectedPid ? " selected" : "");

    const chk = document.createElement("input");
    chk.type = "checkbox";
    chk.checked = p.enabled !== false;
    chk.title = "ブラー有効/無効";
    chk.addEventListener("click", (ev) => {
      ev.stopPropagation();
      p.enabled = chk.checked;
      markDirty(); draw();
      if (previewMode) refreshFrame();
    });

    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = p.color;

    const name = document.createElement("span");
    name.className = "name";
    const kfCount = Object.keys(proj.keyframes[p.id] || {}).length;
    name.textContent = `${p.label} (${kfCount}kf)`;
    name.title = "ダブルクリックで名前変更";
    name.addEventListener("dblclick", () => {
      const v = prompt("名前", p.label);
      if (v) { p.label = v; markDirty(); renderPersonList(); draw(); }
    });

    const del = document.createElement("button");
    del.className = "del";
    del.textContent = "🗑";
    del.title = "人物を削除";
    del.addEventListener("click", (ev) => {
      ev.stopPropagation();
      if (!confirm(`${p.label} を削除しますか？（キーフレームも消えます）`)) return;
      proj.persons = proj.persons.filter((x) => x.id !== p.id);
      delete proj.keyframes[p.id];
      if (selectedPid === p.id) selectedPid = null;
      markDirty(); renderPersonList(); renderBlurPanel(); draw(); drawTimeline();
      if (previewMode) refreshFrame();
    });

    div.append(chk, dot, name, del);
    div.addEventListener("click", () => selectPerson(p.id));
    wrap.appendChild(div);
  }
}

// ── ブラー設定パネル ─────────────────────────────────────────────
function renderBlurPanel() {
  const panel = $("blur-panel");
  const p = proj.persons.find((x) => x.id === selectedPid);
  if (!p) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");
  $("blur-target").textContent = p.label;
  $("blur-type").value = p.blur.type || "gaussian";
  updateStrengthUI(p);
  $("blur-feather").value = Math.round((p.blur.feather ?? 0.25) * 100);
  $("blur-feather-val").textContent = $("blur-feather").value + "%";
}

// 濃さスライダーとKF表示を現在フレームの値に同期する
function updateStrengthUI(p) {
  p = p || proj.persons.find((x) => x.id === selectedPid);
  if (!p) return;
  const eff = Math.round(strengthAt(p, curFrame));
  $("blur-strength").value = eff;
  $("blur-strength-val").textContent = eff;
  const kfs = p.strength_kfs || {};
  const n = Object.keys(kfs).length;
  if (n) {
    const here = kfs[String(curFrame)] !== undefined ? "◆" : "";
    $("strength-kf-info").textContent = `｜KFモード ${n}個 ${here}`;
  } else {
    $("strength-kf-info").textContent = "";
  }
}

function bindBlurControls() {
  const get = () => proj.persons.find((x) => x.id === selectedPid);
  $("blur-type").addEventListener("change", () => {
    const p = get(); if (!p) return;
    p.blur.type = $("blur-type").value;
    markDirty(); if (previewMode) refreshFrame();
  });
  $("blur-strength").addEventListener("input", () => {
    const p = get(); if (!p) return;
    const v = Number($("blur-strength").value);
    if (p.strength_kfs && Object.keys(p.strength_kfs).length) {
      // KFモード中: スライダー操作 = 現在フレームにキーを打つ（AEのストップウォッチON相当）
      p.strength_kfs[String(curFrame)] = v;
      drawTimeline();
    } else {
      p.blur.strength = v;
    }
    $("blur-strength-val").textContent = v;
    updateStrengthUI(p);
    markDirty();
  });
  $("blur-strength").addEventListener("change", () => { if (previewMode) refreshFrame(); });

  $("btn-skf-add").addEventListener("click", () => {
    const p = get(); if (!p) return;
    if (!p.strength_kfs) p.strength_kfs = {};
    p.strength_kfs[String(curFrame)] = Number($("blur-strength").value);
    markDirty(); updateStrengthUI(p); drawTimeline();
    if (previewMode) refreshFrame();
  });
  $("btn-skf-delete").addEventListener("click", () => {
    const p = get(); if (!p || !p.strength_kfs) return;
    delete p.strength_kfs[String(curFrame)];
    if (!Object.keys(p.strength_kfs).length) delete p.strength_kfs;
    markDirty(); updateStrengthUI(p); drawTimeline();
    if (previewMode) refreshFrame();
  });
  $("btn-skf-clear").addEventListener("click", () => {
    const p = get(); if (!p || !p.strength_kfs) return;
    if (!confirm(`${p.label} の濃さキーフレームを全て削除して固定値に戻しますか？`)) return;
    delete p.strength_kfs;
    markDirty(); updateStrengthUI(p); drawTimeline();
    if (previewMode) refreshFrame();
  });
  $("blur-feather").addEventListener("input", () => {
    const p = get(); if (!p) return;
    p.blur.feather = Number($("blur-feather").value) / 100;
    $("blur-feather-val").textContent = $("blur-feather").value + "%";
    markDirty();
  });
  $("blur-feather").addEventListener("change", () => { if (previewMode) refreshFrame(); });

  $("btn-kf-add").addEventListener("click", () => {
    if (!selectedPid) return;
    const ell = interpEllipse(proj.keyframes[selectedPid], curFrame)
      || { cx: proj.video.width / 2, cy: proj.video.height / 2,
           rx: proj.video.width * 0.05, ry: proj.video.width * 0.065 };
    setManualKeyframe(selectedPid, curFrame, ell);
    if (previewMode) refreshFrame();
  });

  $("btn-kf-prop").addEventListener("click", () => {
    if (!selectedPid) return;
    const kfs = proj.keyframes[selectedPid] || {};
    const cur = kfs[String(curFrame)];
    if (!cur || cur.src !== "manual") {
      alert("このフレームに手動キーフレームがありません。\n先に楕円を修正してから伝播してください。");
      return;
    }
    // 修正前の位置 = 自分を除いた残りのキーフレームからの補間値
    const rest = { ...kfs };
    delete rest[String(curFrame)];
    const base = interpEllipse(rest, curFrame);
    if (!base) {
      alert("比較できる自動キーフレームが前後にないため、伝播できません。");
      return;
    }
    const dcx = cur.cx - base.cx;
    const dcy = cur.cy - base.cy;
    const sx = cur.rx / Math.max(1e-6, base.rx);
    const sy = cur.ry / Math.max(1e-6, base.ry);
    const da = (cur.angle || 0) - (base.angle || 0);

    // 次の手動キーフレームの手前まで、修正量を適用
    const frames = Object.keys(kfs).map(Number)
      .filter((f) => f > curFrame).sort((a, b) => a - b);
    let count = 0;
    for (const f of frames) {
      const kf = kfs[String(f)];
      if (kf.src === "manual") break;
      kf.cx = Math.round((kf.cx + dcx) * 10) / 10;
      kf.cy = Math.round((kf.cy + dcy) * 10) / 10;
      kf.rx = Math.round(kf.rx * sx * 10) / 10;
      kf.ry = Math.round(kf.ry * sy * 10) / 10;
      kf.angle = Math.round(((kf.angle || 0) + da) * 10) / 10;
      count++;
    }
    markDirty(); draw(); drawTimeline();
    if (previewMode) refreshFrame();
    $("detect-status").textContent =
      count > 0 ? `修正を後続 ${count} フレームに伝播しました`
                : "伝播先の自動キーフレームがありません";
  });

  $("btn-kf-reseed").addEventListener("click", async () => {
    if (!selectedPid) return;
    const kfs = proj.keyframes[selectedPid] || {};
    const cur = kfs[String(curFrame)];
    if (!cur || cur.src !== "manual") {
      alert("このフレームに手動キーフレームがありません。\n先に楕円を修正してから再追跡してください。");
      return;
    }
    dirty = true;
    await saveProject();
    try {
      $("btn-track").disabled = true;
      $("btn-detect").disabled = true;
      await api(`/api/projects/${proj.id}/reseed`, {
        method: "POST",
        body: JSON.stringify({ person_id: selectedPid, frame: curFrame }),
      });
      $("btn-detect-cancel").dataset.job = "track";
      pollTrack();
    } catch (e) {
      $("detect-status").textContent = e.message;
      $("btn-track").disabled = false;
      $("btn-detect").disabled = false;
    }
  });

  $("btn-kf-toggle").addEventListener("click", () => {
    if (!selectedPid) return;
    const kfs = proj.keyframes[selectedPid] || {};
    const cur = kfs[String(curFrame)];
    const ell = interpEllipse(kfs, curFrame);
    if (cur) {
      cur.visible = cur.visible === false;
      cur.src = "manual";
    } else if (ell) {
      setManualKeyframe(selectedPid, curFrame, { ...ell, visible: false });
      proj.keyframes[selectedPid][String(curFrame)].visible = false;
    } else {
      // 非表示区間 → このフレームから表示させる
      $("btn-kf-add").click();
      return;
    }
    markDirty(); draw(); drawTimeline();
    if (previewMode) refreshFrame();
  });

  $("btn-kf-delete").addEventListener("click", () => {
    if (!selectedPid) return;
    const kfs = proj.keyframes[selectedPid];
    if (kfs && kfs[String(curFrame)]) {
      delete kfs[String(curFrame)];
      markDirty(); draw(); drawTimeline(); renderPersonList();
      if (previewMode) refreshFrame();
    }
  });
}

// ── タイムライン ─────────────────────────────────────────────────
const tlCanvas = $("tl-tracks");
const tlBase = document.createElement("canvas");  // 静的部分のキャッシュ
let tlStart = 0, tlEnd = 0;   // 表示範囲（フレーム）。tlEnd<=tlStart で全体表示
let rangeIn = null, rangeOut = null;   // I/O キーの区間選択

const ISSUE_COLORS = { suspect: "#c678dd", giant: "#ff5555",
                       bridged: "#f5d34a", gap: "#8a93a8",
                       review: "#4aa3f5" };
const ISSUE_LABELS = { suspect: "要注意(未補間)", giant: "サイズ異常",
                       bridged: "遮蔽補間", gap: "KF欠落",
                       review: "要確認(失敗の疑い)" };

function tlRange() {
  const total = proj.video.frame_count;
  return tlEnd > tlStart ? [tlStart, tlEnd] : [0, total];
}
function tlFrameToX(f) {
  const [a, b] = tlRange();
  return ((f - a) / (b - a)) * tlCanvas.width;
}
function tlXToFrame(x) {
  const [a, b] = tlRange();
  return a + (x / tlCanvas.width) * (b - a);
}

// ── 要注意区間の検出（プロジェクトデータから毎回導出・キャッシュ付き） ──
let issuesCache = null, issuesCacheProj = null;
function invalidateIssues() { issuesCache = null; }

function computeIssues() {
  if (issuesCache && issuesCacheProj === proj) return issuesCache;
  const issues = [];
  for (const p of proj.persons) {
    const kfs = proj.keyframes[p.id] || {};
    const frames = Object.keys(kfs).map(Number).sort((x, y) => x - y);
    if (frames.length < 10) continue;
    const areas = [];
    for (const f of frames) {
      const k = kfs[f];
      if (k.visible !== false && !k.parked) areas.push(k.rx * k.ry);
    }
    if (!areas.length) continue;
    const med = areas.slice().sort((x, y) => x - y)[areas.length >> 1];
    let cur = null;
    const push = () => { if (cur) { issues.push(cur); cur = null; } };
    for (let i = 0; i < frames.length; i++) {
      const f = frames[i], k = kfs[f];
      let type = null;
      if (k.suspect) type = "suspect";
      else if (k.visible !== false && k.rx * k.ry > med * 4) type = "giant";
      else if (k.bridged) type = "bridged";
      // 失敗検知の要確認フラグ（tracker の MOUTH_REVIEW_*）。棄却はしていない
      // ＝マスクは AI の実測のまま。人に「ここを見て」と伝えるだけ
      else if (k.review) type = "review";
      if (type && cur && cur.type === type && f - cur.end <= 2) cur.end = f;
      else { push(); if (type) cur = { pid: p.id, type, start: f, end: f }; }
      if (i + 1 < frames.length && frames[i + 1] - f > 2) {
        push();
        issues.push({ pid: p.id, type: "gap", start: f + 1,
                      end: frames[i + 1] - 1 });
      }
    }
    push();
  }
  issues.sort((x, y) => x.start - y.start);
  issuesCache = issues;
  issuesCacheProj = proj;
  return issues;
}

function issueKey(it) { return `${it.pid}:${it.type}:${it.start}-${it.end}`; }
function isReviewed(it) {
  return (proj.reviewed_issues || []).includes(issueKey(it));
}

function jumpIssue(dir) {
  if (!proj) return;
  const list = computeIssues().filter((it) => !isReviewed(it));
  if (!list.length) {
    $("detect-status").textContent = "要注意箇所はありません";
    return;
  }
  let target = null;
  if (dir > 0) target = list.find((it) => it.start > curFrame) || list[0];
  else {
    for (const it of list) if (it.start < curFrame) target = it;
    if (!target) target = list[list.length - 1];
  }
  setFrame(target.start);
  $("detect-status").textContent =
    `要注意: ${ISSUE_LABELS[target.type]} f${target.start}〜${target.end}` +
    "（R キーで確認済みにできます）";
}

function reviewCurrentIssue() {
  if (!proj) return;
  const it = computeIssues().find(
    (x) => x.start - 2 <= curFrame && curFrame <= x.end + 2 && !isReviewed(x));
  if (!it) return;
  proj.reviewed_issues = proj.reviewed_issues || [];
  proj.reviewed_issues.push(issueKey(it));
  markDirty();
  drawTimeline();
  $("detect-status").textContent =
    `確認済みにしました: ${ISSUE_LABELS[it.type]} f${it.start}〜${it.end}`;
}

// ── タイムライン描画 ─────────────────────────────────────────────
function drawTimeline() { renderTimelineBase(); tlOverlay(); }

function renderTimelineBase() {
  if (!proj) return;
  const persons = proj.persons;
  const rowH = 8, gap = 2;
  const marks = (proj.qc && proj.qc.marks) || [];
  const markRow = marks.length ? 10 : 0;
  tlCanvas.width = tlCanvas.clientWidth || tlCanvas.parentElement.clientWidth;
  tlCanvas.height = (persons.length ? persons.length * (rowH + gap) : 0) + markRow;
  tlBase.width = tlCanvas.width;
  tlBase.height = tlCanvas.height;
  const c = tlBase.getContext("2d");
  c.clearRect(0, 0, tlBase.width, tlBase.height);
  const [va, vb] = tlRange();
  const fw = Math.max(1, tlCanvas.width / (vb - va));

  // QC 要確認マーク（赤▼、最上段）
  if (marks.length) {
    c.fillStyle = "#ff5555";
    for (const m of marks) {
      const x = tlFrameToX(m.frame);
      if (x < -4 || x > tlCanvas.width + 4) continue;
      c.beginPath();
      c.moveTo(x - 4, 0); c.lineTo(x + 4, 0); c.lineTo(x, 8);
      c.closePath(); c.fill();
    }
  }
  c.save();
  c.translate(0, markRow);

  const pIdx = {};
  persons.forEach((p, i) => {
    pIdx[p.id] = i;
    const y = i * (rowH + gap);
    c.fillStyle = "#333845";
    c.fillRect(0, y, tlCanvas.width, rowH);
    const kfs = proj.keyframes[p.id] || {};
    c.fillStyle = p.color;
    for (const fs of Object.keys(kfs)) {
      const f = Number(fs);
      if (f < va - 1 || f > vb + 1 || kfs[fs].visible === false) continue;
      c.fillRect(tlFrameToX(f), y, fw, rowH);
    }
    // 手動キーフレームは白で強調
    c.fillStyle = "#ffffff";
    for (const fs of Object.keys(kfs)) {
      const f = Number(fs);
      if (f < va - 1 || f > vb + 1 || kfs[fs].src !== "manual") continue;
      c.fillRect(tlFrameToX(f), y, Math.max(2, fw), rowH);
    }
    // 濃さキーフレームは黄色のマーカー（下半分）
    if (p.strength_kfs) {
      c.fillStyle = "#f5d34a";
      for (const fs of Object.keys(p.strength_kfs)) {
        const f = Number(fs);
        if (f < va - 1 || f > vb + 1) continue;
        c.fillRect(tlFrameToX(f) - 1, y + rowH - 4, 3, 4);
      }
    }
  });

  // 要注意区間のバンド（確認済みは薄く）
  for (const it of computeIssues()) {
    const i = pIdx[it.pid];
    if (i === undefined) continue;
    const y = i * (rowH + gap);
    const x0 = tlFrameToX(it.start), x1 = tlFrameToX(it.end + 1);
    if (x1 < 0 || x0 > tlCanvas.width) continue;
    c.globalAlpha = isReviewed(it) ? 0.15 : 0.45;
    c.fillStyle = ISSUE_COLORS[it.type];
    c.fillRect(x0, y, Math.max(2, x1 - x0), rowH);
    c.globalAlpha = isReviewed(it) ? 0.3 : 1.0;
    c.fillRect(x0, y, Math.max(2, x1 - x0), 2);
    c.globalAlpha = 1.0;
  }
  c.restore();

  // I/O 区間ハイライト（全高）
  if (rangeIn != null && rangeOut != null) {
    const x0 = tlFrameToX(rangeIn), x1 = tlFrameToX(rangeOut + 1);
    c.fillStyle = "rgba(80,160,255,0.18)";
    c.fillRect(x0, 0, Math.max(2, x1 - x0), tlBase.height);
    c.fillStyle = "rgba(80,160,255,0.9)";
    c.fillRect(x0, 0, 1.5, tlBase.height);
    c.fillRect(x1 - 1.5, 0, 1.5, tlBase.height);
  }
}

function tlOverlay() {
  // 人物が居ない間は高さ0のキャンバスになる（drawImage が例外を投げる）
  if (!proj || !tlBase.width || !tlBase.height) return;
  const c = tlCanvas.getContext("2d");
  c.clearRect(0, 0, tlCanvas.width, tlCanvas.height);
  c.drawImage(tlBase, 0, 0);
  const x = tlFrameToX(curFrame);   // 再生ヘッド
  if (x >= 0 && x <= tlCanvas.width) {
    c.fillStyle = "rgba(255,255,255,0.75)";
    c.fillRect(x, 0, 1.5, tlCanvas.height);
  }
}

// ── タイムライン操作: クリック=シーク / ドラッグ=パン / ホイール=ズーム ──
let tlDrag = null;

tlCanvas.addEventListener("mousedown", (ev) => {
  if (!proj) return;
  tlDrag = { x: ev.clientX, moved: false, start: tlStart, end: tlEnd };
});

window.addEventListener("mousemove", (ev) => {
  if (!tlDrag || !proj) return;
  const dx = ev.clientX - tlDrag.x;
  if (Math.abs(dx) > 3) tlDrag.moved = true;
  if (!tlDrag.moved || tlDrag.end <= tlDrag.start) return;  // 全体表示はパン不要
  const span = tlDrag.end - tlDrag.start;
  const total = proj.video.frame_count;
  const df = (-dx / tlCanvas.width) * span;
  let na = tlDrag.start + df, nb = tlDrag.end + df;
  if (na < 0) { nb -= na; na = 0; }
  if (nb > total) { na -= nb - total; nb = total; }
  tlStart = Math.max(0, Math.round(na));
  tlEnd = Math.min(total, Math.round(nb));
  drawTimeline();
});

window.addEventListener("mouseup", (ev) => {
  if (!tlDrag) return;
  const drag = tlDrag;
  tlDrag = null;
  if (drag.moved || !proj) return;
  const rect = tlCanvas.getBoundingClientRect();
  setFrame(Math.round(tlXToFrame(ev.clientX - rect.left)));
});

tlCanvas.addEventListener("wheel", (ev) => {
  if (!proj) return;
  ev.preventDefault();
  const rect = tlCanvas.getBoundingClientRect();
  const fx = tlXToFrame(ev.clientX - rect.left);  // カーソル位置を不動点に
  const [a, b] = tlRange();
  const total = proj.video.frame_count;
  const factor = ev.deltaY < 0 ? 1 / 1.3 : 1.3;
  const span = Math.min(total, Math.max(60, (b - a) * factor));
  let na = fx - ((fx - a) * span) / (b - a);
  let nb = na + span;
  if (na < 0) { nb -= na; na = 0; }
  if (nb > total) { na -= nb - total; nb = total; na = Math.max(0, na); }
  if (span >= total) { tlStart = 0; tlEnd = 0; }
  else { tlStart = Math.round(na); tlEnd = Math.round(nb); }
  drawTimeline();
}, { passive: false });

tlCanvas.addEventListener("dblclick", () => {
  tlStart = 0; tlEnd = 0;
  drawTimeline();
});

// ── 区間ツール（I/O キー + 補間 / 区間限定再追跡） ────────────────
function tlLerpAngle(a, b, t) {
  const d = (((b - a + 180) % 360) + 360) % 360 - 180;
  return a + d * t;
}

function normalizeRange() {
  if (rangeIn != null && rangeOut != null && rangeIn > rangeOut) {
    [rangeIn, rangeOut] = [rangeOut, rangeIn];
  }
  updateRangeUI();
  drawTimeline();
}

function updateRangeUI() {
  const has = rangeIn != null && rangeOut != null;
  $("range-info").textContent =
    rangeIn != null || rangeOut != null
      ? `区間 ${rangeIn ?? "?"}〜${rangeOut ?? "?"}` : "";
  $("btn-range-bridge").disabled = !has;
  $("btn-range-blend").disabled = !has;
  $("btn-range-reseed").disabled = !has;
  $("btn-range-absent").disabled = !has;
  $("btn-range-clear").disabled = rangeIn == null && rangeOut == null;
}

function bridgeRange() {
  if (!proj || !selectedPid) { alert("先に人物を選択してください"); return; }
  if (rangeIn == null || rangeOut == null) return;
  const kfs = (proj.keyframes[selectedPid] = proj.keyframes[selectedPid] || {});
  const findEdge = (from, dir) => {
    for (let off = 0; off <= 24; off++) {
      const k = kfs[String(from + dir * off)];
      if (k && k.visible !== false && !k.parked) return from + dir * off;
    }
    return null;
  };
  const f0 = findEdge(rangeIn, -1), f1 = findEdge(rangeOut, +1);
  if (f0 == null || f1 == null || f1 - f0 < 2) {
    alert("区間の前後に補間の土台になるキーフレームが見つかりません");
    return;
  }
  // 固定点 = 両端 + 区間内の手動KF。固定点の間をピースワイズ線形補間する
  const fixed = [f0];
  for (let f = f0 + 1; f < f1; f++) {
    const k = kfs[String(f)];
    if (k && k.src === "manual" && k.visible !== false) fixed.push(f);
  }
  fixed.push(f1);
  let n = 0;
  for (let s = 0; s < fixed.length - 1; s++) {
    const a = kfs[String(fixed[s])], b = kfs[String(fixed[s + 1])];
    for (let f = fixed[s] + 1; f < fixed[s + 1]; f++) {
      const t = (f - fixed[s]) / (fixed[s + 1] - fixed[s]);
      kfs[String(f)] = {
        cx: Math.round((a.cx + (b.cx - a.cx) * t) * 10) / 10,
        cy: Math.round((a.cy + (b.cy - a.cy) * t) * 10) / 10,
        rx: Math.round((a.rx + (b.rx - a.rx) * t) * 10) / 10,
        ry: Math.round((a.ry + (b.ry - a.ry) * t) * 10) / 10,
        angle: Math.round(tlLerpAngle(a.angle || 0, b.angle || 0, t) * 10) / 10,
        visible: true, src: "auto", bridged: true,
      };
      n++;
    }
  }
  markDirty(`区間補間 f${f0}〜f${f1}`); draw(); drawTimeline();
  if (previewMode) refreshFrame();
  $("detect-status").textContent =
    `区間 f${f0}〜f${f1} を補間で埋めました（${n}フレーム、手動KFは保持）`;
}

// 区間ブレンド修正: 先頭/末尾の手動修正量（ズレ）を線形に配分して、
// 区間内の自動KFに乗せる。「伝播」が一定ズレを丸コピーするのに対し、
// これは始点のズレ→終点のズレへ滑らかに変化させる（ズレが一定でない区間向け）。
// ブリッジと違い自動追跡の細かい動きは残したまま、ドリフトぶんだけ補正する。
function _autoBaseAt(kfs, frame) {
  // その frame の手動KFを除いた補間値＝補正前の自動位置の推定
  const rest = { ...kfs };
  delete rest[String(frame)];
  return interpEllipse(rest, frame);
}

function _deltaVs(cur, base) {
  const da = (((cur.angle || 0) - (base.angle || 0) + 180) % 360 + 360) % 360 - 180;
  return {
    dcx: cur.cx - base.cx,
    dcy: cur.cy - base.cy,
    sx: cur.rx / Math.max(1e-6, base.rx),
    sy: cur.ry / Math.max(1e-6, base.ry),
    da,
  };
}

function blendRange() {
  if (!proj || !selectedPid) { alert("先に人物を選択してください"); return; }
  if (rangeIn == null || rangeOut == null) return;
  const kfs = proj.keyframes[selectedPid] || {};
  const [a, b] = rangeIn <= rangeOut ? [rangeIn, rangeOut] : [rangeOut, rangeIn];
  const ka = kfs[String(a)], kb = kfs[String(b)];
  if (!ka || ka.src !== "manual" || ka.visible === false ||
      !kb || kb.src !== "manual" || kb.visible === false) {
    alert("区間の先頭と末尾の両方に、手動で修正した表示中キーフレームが必要です。\n" +
          "先頭フレームと末尾フレームをそれぞれ直してから実行してください。");
    return;
  }
  if (b - a < 2) { alert("区間が短すぎます（間に自動キーフレームがありません）"); return; }
  const baseA = _autoBaseAt(kfs, a), baseB = _autoBaseAt(kfs, b);
  if (!baseA || !baseB) {
    alert("両端の補正前位置を推定できません（前後に自動キーフレームが必要です）。");
    return;
  }
  const dA = _deltaVs(ka, baseA), dB = _deltaVs(kb, baseB);
  let count = 0;
  for (let f = a + 1; f < b; f++) {
    const kf = kfs[String(f)];
    if (!kf) continue;
    if (kf.src === "manual") continue;   // 途中の手動修正は保持
    if (kf.visible === false) continue;  // 不在フレームには乗せない
    const t = (f - a) / (b - a);
    const dcx = dA.dcx + (dB.dcx - dA.dcx) * t;
    const dcy = dA.dcy + (dB.dcy - dA.dcy) * t;
    const sx = dA.sx + (dB.sx - dA.sx) * t;
    const sy = dA.sy + (dB.sy - dA.sy) * t;
    const da = dA.da + (dB.da - dA.da) * t;
    kf.cx = Math.round((kf.cx + dcx) * 10) / 10;
    kf.cy = Math.round((kf.cy + dcy) * 10) / 10;
    kf.rx = Math.round(kf.rx * sx * 10) / 10;
    kf.ry = Math.round(kf.ry * sy * 10) / 10;
    kf.angle = Math.round(((kf.angle || 0) + da) * 10) / 10;
    kf.blended = true;
    count++;
  }
  markDirty(`区間ブレンド f${a}〜f${b}`); draw(); drawTimeline();
  if (previewMode) refreshFrame();
  $("detect-status").textContent =
    `区間 f${a}〜f${b} を両端のズレでブレンド修正しました（${count}フレーム、手動KFは保持）`;
}

async function rangeReseed() {
  if (!proj || !selectedPid) { alert("先に人物を選択してください"); return; }
  if (rangeIn == null || rangeOut == null) return;
  const kfs = proj.keyframes[selectedPid] || {};
  const seed = kfs[String(rangeIn)];
  if (!seed || seed.visible === false) {
    alert("区間の先頭フレームにキーフレームが必要です（再追跡のシードになります）");
    return;
  }
  dirty = true;
  await saveProject();
  await pushSnapshot(`区間再追跡 f${rangeIn}〜${rangeOut}`);
  try {
    $("btn-track").disabled = true;
    $("btn-detect").disabled = true;
    await api(`/api/projects/${proj.id}/reseed`, {
      method: "POST",
      body: JSON.stringify({ person_id: selectedPid, frame: rangeIn,
                             end: rangeOut }),
    });
    $("btn-detect-cancel").dataset.job = "track";
    pollTrack();
  } catch (e) {
    $("detect-status").textContent = e.message;
    $("btn-track").disabled = false;
    $("btn-detect").disabled = false;
  }
}

$("btn-range-bridge").addEventListener("click", bridgeRange);
$("btn-range-blend").addEventListener("click", blendRange);
$("btn-range-reseed").addEventListener("click", rangeReseed);
$("btn-range-clear").addEventListener("click", () => {
  rangeIn = rangeOut = null;
  updateRangeUI();
  drawTimeline();
});

// フレーム番号を打ち込んで区間を設定（I/Oキーの代わり）。「始-終」形式。
function setRangeFromInput() {
  if (!proj) return;
  const raw = ($("range-set").value || "").trim();
  const m = raw.match(/^(\d+)\s*[-〜~,]\s*(\d+)$/);
  if (!m) { alert("「開始-終了」の形式で入力してください（例 120-180）"); return; }
  const max = (proj.video.frame_count || 1) - 1;
  const clamp = (v) => Math.max(0, Math.min(v, max));
  const a = clamp(parseInt(m[1], 10)), b = clamp(parseInt(m[2], 10));
  rangeIn = Math.min(a, b);
  rangeOut = Math.max(a, b);
  normalizeRange();
  setFrame(rangeIn);   // 先頭へシークして確認しやすく
}
$("btn-range-set").addEventListener("click", setRangeFromInput);
$("range-set").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") { ev.preventDefault(); setRangeFromInput(); }
});

// ── 不在マーク（誤認識/飛び回り区間をマスクごと画面外へ） ──────────
// absent:true を立てると interpEllipse がパーキングを返し、プレビュー/書き出しは
// ブラーを掛けない。元のジオメトリは残すので「復帰」で戻せる（可逆）。
function _lastKnownEllipse(kfs, frame) {
  for (let f = frame; f >= Math.max(0, frame - 480); f--) {
    const k = kfs[String(f)];
    if (k && k.visible !== false && !k.absent) return k;
  }
  const max = proj.video.frame_count - 1;
  for (let f = frame; f <= Math.min(max, frame + 480); f++) {
    const k = kfs[String(f)];
    if (k && k.visible !== false && !k.absent) return k;
  }
  return null;
}

// frames を不在に（on=true）／復帰（on=false）。トグルはこれを使う。
function setAbsent(pid, a, b, on) {
  const kfs = (proj.keyframes[pid] = proj.keyframes[pid] || {});
  let ref = null;
  if (on) {
    ref = _lastKnownEllipse(kfs, a);
    if (!ref) {
      // 土台となる楕円が無い＝サイズ不明。画面中央そばの小楕円で退避させる
      const r = proj.video.width * 0.05;
      ref = { cx: proj.video.width / 2, cy: proj.video.height / 2,
              rx: r, ry: r * 1.3, angle: 0 };
    }
  }
  let count = 0;
  for (let f = a; f <= b; f++) {
    const key = String(f);
    const kf = kfs[key];
    if (on) {
      if (kf && !kf.absent) {
        kf._prevVisible = kf.visible !== false;  // 復帰用に元の可視を控える
        kf.visible = false;
        kf.absent = true;
      } else if (!kf) {
        kfs[key] = { cx: ref.cx, cy: ref.cy, rx: ref.rx, ry: ref.ry,
                     angle: ref.angle || 0, visible: false, absent: true,
                     src: "manual", synthetic: true };
      } else { continue; }  // 既に absent
      count++;
    } else {
      if (!kf || !kf.absent) continue;
      if (kf.synthetic) { delete kfs[key]; }      // 合成マーカーは丸ごと削除
      else {
        delete kf.absent;
        kf.visible = kf._prevVisible !== false;   // 元の可視へ戻す
        delete kf._prevVisible;
      }
      count++;
    }
  }
  markDirty(); draw(); drawTimeline();
  if (previewMode) refreshFrame();
  return count;
}

function toggleAbsentFrame() {
  if (!proj || !selectedPid) { alert("先に人物を選択してください"); return; }
  const kf = (proj.keyframes[selectedPid] || {})[String(curFrame)];
  const on = !(kf && kf.absent);
  const n = setAbsent(selectedPid, curFrame, curFrame, on);
  $("detect-status").textContent = on
    ? `f${curFrame} を不在にしました（マスクを画面外へ）`
    : `f${curFrame} を復帰しました`;
}

function toggleAbsentRange() {
  if (!proj || !selectedPid) { alert("先に人物を選択してください"); return; }
  if (rangeIn == null || rangeOut == null) {
    alert("先に区間を指定してください（I/O キー、または「始-終」入力→区間設定）");
    return;
  }
  const [a, b] = rangeIn <= rangeOut ? [rangeIn, rangeOut] : [rangeOut, rangeIn];
  const startKf = (proj.keyframes[selectedPid] || {})[String(a)];
  const on = !(startKf && startKf.absent);   // 先頭が不在なら区間を復帰、でなければ不在化
  const n = setAbsent(selectedPid, a, b, on);
  $("detect-status").textContent = on
    ? `区間 f${a}〜f${b} を不在にしました（${n}フレーム、マスクを画面外へ）`
    : `区間 f${a}〜f${b} を復帰しました（${n}フレーム）`;
}

$("btn-kf-absent").addEventListener("click", toggleAbsentFrame);
$("btn-range-absent").addEventListener("click", toggleAbsentRange);

// ── 再生 ─────────────────────────────────────────────────────────
function setPlaying(v) {
  playing = v;
  $("btn-play").textContent = playing ? "⏸" : "▶";
  clearInterval(playTimer);
  if (playing) {
    const interval = Math.max(1000 / proj.video.fps, 33);
    playTimer = setInterval(() => {
      if (frameLoading) return;  // 読み込みが追いつかない時はスキップしない
      if (curFrame >= proj.video.frame_count - 1) { setPlaying(false); return; }
      setFrame(curFrame + 1);
    }, interval);
  }
}

// ── 高精度追跡（SAM 2）設定 ──────────────────────────────────────
let serverConfig = { autoclick_available: false, sam2_available: false };

async function loadConfig() {
  try {
    serverConfig = await api("/api/config");
  } catch (e) { /* デフォルトのまま */ }
  $("autoclick-note").textContent = serverConfig.autoclick_available
    ? "Claude API 利用可能"
    : "ANTHROPIC_API_KEY 未設定のため自動クリックはスキップされます";
}

function getTargets() {
  return {
    mode: $("track-mode").value,
    ref_images: $("ref-images").value.split("\n")
      .map((s) => s.trim()).filter(Boolean),
    use_autoclick: $("chk-autoclick").checked,
    region: $("detect-region").value,   // バッチ実行時に使う
  };
}

function applyTargetsToUi() {
  const t = (proj && proj.targets) || {};
  $("track-mode").value = t.mode || "all";
  $("ref-images").value = (t.ref_images || []).join("\n");
  $("chk-autoclick").checked = t.use_autoclick !== false;
  if (t.region) $("detect-region").value = t.region;
  $("ref-images-ctl").classList.toggle("hidden", $("track-mode").value !== "reference");
  renderRefThumbs();
}

function renderRefThumbs() {
  const wrap = $("ref-thumbs");
  wrap.innerHTML = "";
  const paths = $("ref-images").value.split("\n")
    .map((s) => s.trim()).filter(Boolean);
  for (const p of paths) {
    const img = document.createElement("img");
    img.src = `/api/thumb?path=${encodeURIComponent(p)}`;
    img.title = `${p}\n（クリックで削除）`;
    img.addEventListener("click", () => {
      if (!confirm("このリファレンスを削除しますか？")) return;
      $("ref-images").value = paths.filter((x) => x !== p).join("\n");
      if (proj) { proj.targets = getTargets(); markDirty(); }
      renderRefThumbs();
    });
    img.addEventListener("error", () => { img.style.opacity = "0.3"; });
    wrap.appendChild(img);
  }
}

// ── 動画内クリックで対象人物を登録 ───────────────────────────────
let pickingTarget = false;

function setPickingMode(v) {
  pickingTarget = v;
  canvas.classList.toggle("adding", v || addingPerson);
  $("btn-pick-target").textContent =
    v ? "対象の顔をクリックしてください..." : "🖱 動画から対象人物を選択";
}

async function pickTargetAt(pos) {
  try {
    const res = await api(`/api/projects/${proj.id}/pick_face`, {
      method: "POST",
      body: JSON.stringify({ frame: curFrame, x: pos.x, y: pos.y }),
    });
    const ta = $("ref-images");
    ta.value = (ta.value.trim() ? ta.value.trim() + "\n" : "") + res.path;
    $("track-mode").value = "reference";
    $("ref-images-ctl").classList.remove("hidden");
    proj.targets = getTargets();
    markDirty();
    renderRefThumbs();
    $("detect-status").textContent =
      "対象人物を登録しました（追跡モード: リファレンス照合）";
  } catch (e) {
    alert(e.message);
  }
}

function bindTrackControls() {
  $("track-mode").addEventListener("change", () => {
    $("ref-images-ctl").classList.toggle("hidden", $("track-mode").value !== "reference");
    if (proj) { proj.targets = getTargets(); markDirty(); }
  });
  $("ref-images").addEventListener("change", () => {
    if (proj) { proj.targets = getTargets(); markDirty(); }
    renderRefThumbs();
  });
  $("btn-pick-target").addEventListener("click", () => {
    setPickingMode(!pickingTarget);
  });
  $("chk-autoclick").addEventListener("change", () => {
    if (proj) { proj.targets = getTargets(); markDirty(); }
  });
  $("detect-region").addEventListener("change", () => {
    if (proj) { proj.targets = getTargets(); markDirty(); }
  });

  $("btn-track").addEventListener("click", async () => {
    const t = getTargets();
    if (t.mode === "reference" && !t.ref_images.length) {
      alert("リファレンス照合には、対象人物の顔画像パスを1枚以上指定してください。");
      return;
    }
    if (proj.persons.length &&
        !confirm("高精度追跡を実行すると自動検出の人物は置き換えられます（手動編集済みの人物は残ります）。実行しますか？")) {
      return;
    }
    proj.targets = t;
    dirty = true;
    await saveProject();
    await pushSnapshot("高精度追跡の実行");
    try {
      $("btn-track").disabled = true;
      $("btn-detect").disabled = true;
      await api(`/api/projects/${proj.id}/track`, {
        method: "POST",
        body: JSON.stringify({ ...t, region: $("detect-region").value }),
      });
      autoQcAfterTrack = true;   // 追跡完了後に漏れQCを自動実行
      pollTrack();
    } catch (e) {
      $("detect-status").textContent = e.message;
      $("btn-track").disabled = false;
      $("btn-detect").disabled = false;
      autoQcAfterTrack = false;
    }
  });
}

let liveTrack = null;
let autoQcAfterTrack = false;   // 高精度追跡の完了後に漏れQCを自動実行するフラグ

// 漏れQCを実行する（確認ダイアログは呼び出し側の責務。自動実行時は確認なし）
async function runQc() {
  if (!proj || !proj.persons.length) return false;
  const useApi = serverConfig.autoclick_available;
  dirty = true;
  await saveProject();
  await pushSnapshot("漏れQCの実行");
  try {
    $("btn-qc").disabled = true;
    $("btn-track").disabled = true;
    $("btn-detect").disabled = true;
    await api(`/api/projects/${proj.id}/qc`, {
      method: "POST",
      body: JSON.stringify({ every_sec: 1.0, use_api: useApi }),
    });
    $("btn-detect-cancel").dataset.job = "track";
    pollTrack();
    return true;
  } catch (e) {
    $("detect-status").textContent = e.message;
    $("btn-qc").disabled = false;
    $("btn-track").disabled = false;
    $("btn-detect").disabled = false;
    return false;
  }
}

async function pollTrack() {
  try {
    const st = await api(`/api/projects/${proj.id}/track/status`);
    if (st.state === "running") {
      $("detect-status").textContent = `${st.message} ${(st.progress * 100).toFixed(0)}%`;
      $("btn-detect-cancel").classList.remove("hidden");
      $("btn-detect-cancel").dataset.job = "track";
      $("btn-track").disabled = true;
      $("btn-detect").disabled = true;
      $("btn-qc").disabled = true;
      // ライブ表示: 追跡ジョブが今見ているフレームを追いかける
      if (st.live && $("chk-live").checked && !playing) {
        liveTrack = st.live;
        if (!dragMode) setFrame(st.live.frame);
        else draw();
      }
      setTimeout(pollTrack, 900);
    } else if (st.state !== "none") {
      // 追跡完了時のみ漏れQCを自動実行（QC自身の完了では再実行しない）
      const runQcNow = st.state === "done" && autoQcAfterTrack;
      autoQcAfterTrack = false;
      $("detect-status").textContent = st.message || "";
      $("btn-track").disabled = false;
      $("btn-detect").disabled = false;
      $("btn-qc").disabled = false;
      $("btn-detect-cancel").classList.add("hidden");
      liveTrack = null;
      if (st.state === "done") {
        proj = await api(`/api/projects/${proj.id}`);
        selectedPid = null;
        renderPersonList(); renderBlurPanel(); draw(); drawTimeline();
        if (previewMode) refreshFrame();
        if (runQcNow) await runQc();   // 確認ダイアログなしで自動実行
      } else {
        draw();
      }
    }
  } catch (e) {
    $("detect-status").textContent = `状態取得失敗: ${e.message}`;
    $("btn-track").disabled = false;
    $("btn-detect").disabled = false;
    $("btn-qc").disabled = false;
    liveTrack = null;
    autoQcAfterTrack = false;
  }
}

// ── Sキー: 1フレームだけ SAM 2 伝播して進む ──────────────────────
let stepBusy = false;

async function stepPropagate() {
  if (stepBusy || !proj) return;
  if (!selectedPid) {
    $("detect-status").textContent = "人物未選択のため伝播なしで移動しました";
    setFrame(curFrame + 1);
    return;
  }
  stepBusy = true;
  $("detect-status").textContent = "⏵ SAM 2 で1フレーム伝播中…";
  try {
    const r = await api(`/api/projects/${proj.id}/step`, {
      method: "POST",
      body: JSON.stringify({ person_id: selectedPid, frame: curFrame }),
    });
    if (!proj.keyframes[selectedPid]) proj.keyframes[selectedPid] = {};
    proj.keyframes[selectedPid][String(r.frame)] = r.kf;
    $("detect-status").textContent = r.skipped === "next_manual"
      ? `⏵ frame ${r.frame} は手動KFのため移動のみ`
        + (r.calibrated ? "（形を学習しました）" : "")
      : `⏵ 伝播: frame ${r.frame} を更新` + (r.calibrated ? "（形を学習）" : "");
    setFrame(r.frame);
    drawTimeline();
  } catch (e) {
    $("detect-status").textContent = `伝播できず移動のみ: ${e.message}`;
    setFrame(curFrame + 1);
  } finally {
    stepBusy = false;
  }
}

// ── Claude API 使用量表示 ────────────────────────────────────────
let usageTimer = null;

async function pollUsage() {
  try {
    const u = await api("/api/usage");
    const el = $("api-usage");
    const b = (proj && u.projects && u.projects[proj.id])
      || { calls: 0, est_cost_usd: 0, input_tokens: 0, output_tokens: 0, last_call: null };
    if (!u.calls && !u.active) { el.textContent = ""; el.classList.remove("active"); return; }
    el.textContent = (u.active ? "🤖 API呼び出し中… " : "🤖 ")
      + `このPJ ${b.calls}回/$${(b.est_cost_usd || 0).toFixed(3)}`
      + ` ｜ 全体 ${u.calls}回/$${(u.est_cost_usd || 0).toFixed(3)}`;
    el.title = `このプロジェクトの使用量（概算）\n`
      + `${b.calls}回 / 入力 ${b.input_tokens} tok / 出力 ${b.output_tokens} tok\n最終: ${b.last_call || "-"}\n`
      + `※プロジェクト別の集計は本日の機能追加以降の呼び出しが対象\n`
      + `── 全体累計 ──\n${u.calls}回 / 約$${(u.est_cost_usd || 0).toFixed(3)}（モデル: ${u.model}）`;
    el.classList.toggle("active", !!u.active);
  } catch (e) { /* 表示のみなので無視 */ }
}

async function updateGlobalUsage() {
  try {
    const u = await api("/api/usage");
    $("usage-global").textContent = u.calls
      ? `🤖 Claude API 全体累計: ${u.calls}回 / 約$${(u.est_cost_usd || 0).toFixed(3)}（モデル: ${u.model}）`
      : "";
  } catch (e) { /* 表示のみ */ }
}

// ── 自動検出 / 書き出し ──────────────────────────────────────────
async function pollDetect() {
  try {
    const st = await api(`/api/projects/${proj.id}/detect/status`);
    if (st.state === "running") {
      $("detect-status").textContent = `${st.message} ${(st.progress * 100).toFixed(0)}%`;
      $("btn-detect-cancel").classList.remove("hidden");
      setTimeout(pollDetect, 800);
    } else {
      $("detect-status").textContent = st.message || "";
      $("btn-detect").disabled = false;
      $("btn-detect-cancel").classList.add("hidden");
      if (st.state === "done") {
        // 検出結果を取り込む（未保存のUI編集は先に保存済み）
        proj = await api(`/api/projects/${proj.id}`);
        selectedPid = null;
        renderPersonList(); renderBlurPanel(); draw(); drawTimeline();
        if (previewMode) refreshFrame();
      }
    }
  } catch (e) {
    $("detect-status").textContent = `状態取得失敗: ${e.message}`;
    $("btn-detect").disabled = false;
  }
}

async function pollExport() {
  try {
    const st = await api(`/api/projects/${proj.id}/export/status`);
    if (st.state === "running") {
      $("export-status").textContent = `書き出し ${(st.progress * 100).toFixed(0)}%`;
      setTimeout(pollExport, 1000);
    } else {
      $("export-status").textContent = st.message || "";
      $("btn-export").disabled = false;
    }
  } catch (e) {
    $("export-status").textContent = `状態取得失敗: ${e.message}`;
    $("btn-export").disabled = false;
  }
}

// ── 画面遷移 ─────────────────────────────────────────────────────
function showStart() {
  setPlaying(false);
  clearInterval(usageTimer);
  liveTrack = null;
  $("screen-editor").classList.add("hidden");
  $("screen-start").classList.remove("hidden");
  loadProjectList();
  updateGlobalUsage();
}

async function openProject(pid) {
  proj = await api(`/api/projects/${pid}`);
  selectedPid = null;
  dirty = false;
  view = { zoom: 1, x: 0, y: 0 };
  $("screen-start").classList.add("hidden");
  $("screen-editor").classList.remove("hidden");
  $("proj-name").textContent = proj.name;
  $("save-status").textContent = "";
  $("detect-status").textContent = "";
  $("export-status").textContent = "";
  $("seek").max = proj.video.frame_count - 1;
  renderPersonList();
  renderBlurPanel();
  applyTargetsToUi();
  // セーブファイルに記録された前回の作業位置から再開
  setFrame(proj.ui?.frame || 0, () => { draw(); drawTimeline(); });
  drawTimeline();
  pollDetect();  // 実行中ジョブがあれば表示を復元
  pollTrack();
  clearInterval(usageTimer);
  usageTimer = setInterval(pollUsage, 3000);
  pollUsage();
}

// ── スタート画面 ─────────────────────────────────────────────────
const batchSelected = new Set();   // バッチ対象の pid

async function loadProjectList() {
  const wrap = $("project-list");
  try {
    const items = await api("/api/projects");
    wrap.innerHTML = items.length ? "" : '<p class="muted">プロジェクトはまだありません。</p>';
    for (const it of items) {
      const div = document.createElement("div");
      div.className = "proj-item" + (it.video_exists ? "" : " missing");
      const sec = Math.round(it.frame_count / (it.fps || 30));
      // バッチ対象チェックボックス
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.className = "p-batch";
      cb.title = "バッチ追跡の対象にする";
      cb.checked = batchSelected.has(it.id);
      cb.disabled = !it.video_exists;
      cb.addEventListener("click", (ev) => {
        ev.stopPropagation();
        if (cb.checked) batchSelected.add(it.id);
        else batchSelected.delete(it.id);
        updateBatchUI();
      });
      div.appendChild(cb);
      // 保存済み追跡設定のサマリー（バッチの仕込み確認用）
      const setup = it.targets_mode
        ? `⚙ ${it.targets_mode === "reference"
              ? `リファレンス${it.targets_refs}枚` : "全員"}・`
          + `${it.targets_region === "mouth" ? "口元" : "顔全体"}`
        : "⚙ 追跡設定なし（開いて設定）";
      const info = document.createElement("span");
      info.className = "p-info";
      info.innerHTML = `
        <span class="p-name">${it.name}</span>
        <span class="p-meta">${it.updated.replace("T", " ")} ・ ${sec}秒 ・ 人物${it.person_count}
          ・ ${setup}
          ${it.video_exists ? "" : "⚠ 動画が見つかりません"}</span>`;
      div.appendChild(info);
      const del = document.createElement("button");
      del.className = "p-del";
      del.textContent = "削除";
      del.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        if (!confirm(`プロジェクト「${it.name}」を削除しますか？`)) return;
        await api(`/api/projects/${it.id}`, { method: "DELETE" });
        batchSelected.delete(it.id);
        loadProjectList();
      });
      div.appendChild(del);
      info.addEventListener("click", () => openProject(it.id));
      wrap.appendChild(div);
    }
    updateBatchUI();
    pollBatch(true);   // 実行中バッチがあれば表示を再開（ブラウザ再訪時）
  } catch (e) {
    wrap.innerHTML = `<p class="muted">読み込み失敗: ${e.message}</p>`;
  }
}

// ── バッチ処理 UI ────────────────────────────────────────────────
const BATCH_STATE_LABEL = {
  pending: "待機", tracking: "追跡中", qc: "QC中",
  done: "✅完了", error: "❌失敗", skipped: "スキップ",
};

function updateBatchUI() {
  $("btn-batch").disabled = batchSelected.size === 0;
  $("btn-batch").textContent =
    batchSelected.size ? `選択した${batchSelected.size}件をバッチ追跡` : "バッチ追跡";
}

let batchPollTimer = null;

async function startBatch() {
  if (!batchSelected.size) return;
  if (!confirm(
      `${batchSelected.size}件のプロジェクトを順番に高精度追跡します。\n` +
      `各プロジェクトの保存済み設定（対象人物・範囲）を使います。\n` +
      `処理中はPCをスリープさせないでください。開始しますか？`)) return;
  try {
    $("btn-batch").disabled = true;
    await api("/api/batch", {
      method: "POST",
      body: JSON.stringify({ pids: [...batchSelected],
                             qc: $("chk-batch-qc").checked }),
    });
    pollBatch();
  } catch (e) {
    $("batch-status").textContent = e.message;
    updateBatchUI();
  }
}

async function pollBatch(once = false) {
  clearTimeout(batchPollTimer);
  let st;
  try {
    st = await api("/api/batch/status");
  } catch (e) {
    return;
  }
  const box = $("batch-status");
  if (st.state === "none") { box.innerHTML = ""; return; }
  const rows = st.items.map((it) => {
    const pct = it.progress != null ? Math.round(it.progress * 100) : null;
    return `<div class="batch-row ${it.state}">
       <span>${BATCH_STATE_LABEL[it.state] || it.state}${pct != null ? ` ${pct}%` : ""}</span>
       <span class="b-name">${it.name}</span>
       ${pct != null
         ? `<span class="b-prog"><i style="width:${pct}%"></i></span>`
         : ""}
       <span class="b-msg">${it.job_message || it.message || ""}</span>
     </div>`;
  }).join("");
  const head = st.state === "running"
    ? `バッチ実行中（開始 ${st.started.replace("T", " ")}）
       <button id="btn-batch-cancel">中止</button>`
    : `バッチ${st.state === "done" ? "完了" : "中止"}（${(st.finished || "").replace("T", " ")}）`;
  box.innerHTML = `<div class="batch-head">${head}</div>${rows}`;
  const cancel = $("btn-batch-cancel");
  if (cancel) cancel.addEventListener("click", async () => {
    if (confirm("バッチを中止しますか？（実行中の追跡もキャンセルされます）")) {
      await api("/api/batch/cancel", { method: "POST", body: "{}" });
    }
  });
  if (st.state === "running") {
    batchPollTimer = setTimeout(() => pollBatch(), 3000);
  } else {
    updateBatchUI();
  }
}

async function loadVideoList() {
  const dir = $("video-dir").value.trim();
  const res = await api(`/api/videos${dir ? `?dir=${encodeURIComponent(dir)}` : ""}`);
  $("video-dir").value = res.dir;
  const wrap = $("video-list");
  wrap.innerHTML = "";
  if (res.error) {
    wrap.innerHTML = `<p class="muted">${res.error}: ${res.dir}</p>`;
    return;
  }
  if (!res.videos.length) {
    wrap.innerHTML = '<p class="muted">動画ファイルがありません。</p>';
    return;
  }
  for (const v of res.videos) {
    const div = document.createElement("div");
    div.className = "video-item";
    div.textContent = "🎥 " + v.split("/").pop();
    div.title = v;
    div.addEventListener("click", () => createProject(v));
    wrap.appendChild(div);
  }
}

async function createProject(videoPath) {
  try {
    const p = await api("/api/projects", {
      method: "POST",
      body: JSON.stringify({ video_path: videoPath }),
    });
    openProject(p.id);
  } catch (e) {
    alert("プロジェクト作成失敗: " + e.message);
  }
}

// ── 追加モード ───────────────────────────────────────────────────
function setAddingMode(v) {
  addingPerson = v;
  canvas.classList.toggle("adding", v);
  $("btn-add-person").textContent = v ? "クリックで配置..." : "＋追加";
}

// ── イベント束ね ─────────────────────────────────────────────────
function bindEvents() {
  $("btn-scan-dir").addEventListener("click", loadVideoList);
  $("btn-create-direct").addEventListener("click", () => {
    const p = $("video-path-direct").value.trim();
    if (p) createProject(p);
  });

  $("btn-back").addEventListener("click", async () => {
    dirty = true;
    await saveProject();
    showStart();
  });

  $("btn-save").addEventListener("click", () => { dirty = true; saveProject(); });
  $("btn-history").addEventListener("click", openHistory);
  $("btn-history-close").addEventListener("click",
    () => $("history-modal").classList.add("hidden"));
  $("history-modal").addEventListener("click", (ev) => {
    if (ev.target === $("history-modal")) $("history-modal").classList.add("hidden");
  });

  $("btn-add-person").addEventListener("click", () => setAddingMode(!addingPerson));

  $("chk-preview").addEventListener("change", () => {
    previewMode = $("chk-preview").checked;
    refreshFrame();
  });

  $("chk-show-mask").addEventListener("change", () => {
    showMask = $("chk-show-mask").checked;
    draw();
  });

  $("chk-show-detect").addEventListener("change", () => {
    showDetect = $("chk-show-detect").checked;
    draw();
  });

  $("chk-show-landmarks").addEventListener("change", () => {
    showLandmarks = $("chk-show-landmarks").checked;
    if (showLandmarks) {
      fetchLandmarks(curFrame);   // オンにした瞬間に現フレームを取得
    } else {
      landmarksData = null;
      draw();
    }
  });

  $("btn-detect").addEventListener("click", async () => {
    if (proj.persons.length &&
        !confirm("再検出すると自動検出の人物は置き換えられます（手動編集済みの人物は残ります）。実行しますか？")) {
      return;
    }
    dirty = true;
    await saveProject();  // 検出はサーバー側でセーブファイルを読むため先に保存
    await pushSnapshot("高速検出の実行");
    try {
      $("btn-detect").disabled = true;
      await api(`/api/projects/${proj.id}/detect`, {
        method: "POST",
        body: JSON.stringify({ region: $("detect-region").value }),
      });
      $("btn-detect-cancel").dataset.job = "detect";
      pollDetect();
    } catch (e) {
      $("detect-status").textContent = e.message;
      $("btn-detect").disabled = false;
    }
  });

  $("btn-detect-cancel").addEventListener("click", () => {
    const job = $("btn-detect-cancel").dataset.job || "detect";
    api(`/api/projects/${proj.id}/${job}/cancel`, { method: "POST" });
  });

  $("btn-qc").addEventListener("click", async () => {
    if (!proj.persons.length) {
      alert("検品対象の人物がいません。先に検出/追跡を実行してください。");
      return;
    }
    const useApi = serverConfig.autoclick_available;
    if (!confirm("ブラー漏れの検品と自動修正を実行します。\n"
        + (useApi ? "未カバー区間は Claude API でも検品します（従量課金）。"
                  : "APIキー未設定のためローカル検品のみ行います。")
        + "\n実行しますか？")) {
      return;
    }
    await runQc();
  });

  $("btn-export").addEventListener("click", async () => {
    dirty = true;
    await saveProject();
    try {
      $("btn-export").disabled = true;
      await api(`/api/projects/${proj.id}/export`, {
        method: "POST", body: JSON.stringify({}),
      });
      pollExport();
    } catch (e) {
      $("export-status").textContent = e.message;
      $("btn-export").disabled = false;
    }
  });

  $("btn-ae-json").addEventListener("click", async () => {
    if (!proj) return;
    if (!proj.persons.length) {
      alert("書き出す人物がいません。先に検出/追跡を実行してください。");
      return;
    }
    // 最新のキーフレームを保存してからダウンロード（サーバは保存済みを読む）
    dirty = true;
    await saveProject();
    const a = document.createElement("a");
    a.href = `/api/projects/${proj.id}/ae_export.json`;
    a.download = `${proj.name || proj.id}_ae.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    $("export-status").textContent = "AE用JSONを書き出しました";
  });

  $("btn-play").addEventListener("click", () => setPlaying(!playing));
  $("btn-prev").addEventListener("click", () => setFrame(curFrame - 1));
  $("btn-next").addEventListener("click", () => setFrame(curFrame + 1));
  $("seek").addEventListener("input", () => setFrame(Number($("seek").value)));

  window.addEventListener("keydown", (ev) => {
    if ($("screen-editor").classList.contains("hidden")) return;
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
    if (ev.code === "Space") { ev.preventDefault(); setPlaying(!playing); }
    else if (ev.code === "ArrowLeft") setFrame(curFrame - (ev.shiftKey ? 10 : 1));
    else if (ev.code === "ArrowRight") setFrame(curFrame + (ev.shiftKey ? 10 : 1));
    else if (ev.code === "KeyA") setFrame(curFrame - 1);
    else if (ev.code === "KeyS") { ev.preventDefault(); stepPropagate(); }
    else if (ev.code === "Digit0" || ev.code === "Numpad0") resetView();
    else if (ev.code === "KeyN") jumpIssue(1);
    else if (ev.code === "KeyP") jumpIssue(-1);
    else if (ev.code === "KeyR") reviewCurrentIssue();
    else if (ev.code === "KeyI") { rangeIn = curFrame; normalizeRange(); }
    else if (ev.code === "KeyO") { rangeOut = curFrame; normalizeRange(); }
    else if (ev.code === "Escape") {
      rangeIn = rangeOut = null;
      updateRangeUI();
      drawTimeline();
    }
  });

  $("frame-jump").addEventListener("keydown", (ev) => {
    if (ev.code === "Enter" || ev.code === "NumpadEnter") {
      const n = parseInt($("frame-jump").value, 10);
      if (!isNaN(n)) setFrame(n);
      $("frame-jump").blur();
    }
  });

  window.addEventListener("resize", () => { if (proj) { draw(); drawTimeline(); } });

  // タブを閉じる前に保存を試みる
  window.addEventListener("beforeunload", () => {
    if (proj && dirty) {
      proj.ui = proj.ui || {};
      proj.ui.frame = curFrame;
      navigator.sendBeacon(
        `/api/projects/${proj.id}`,
        new Blob([JSON.stringify(proj)], { type: "application/json" })
      );
    }
  });
}

// ── 起動 ─────────────────────────────────────────────────────────
$("btn-batch").addEventListener("click", startBatch);
bindEvents();
bindBlurControls();
bindTrackControls();
loadConfig();
loadProjectList();
loadVideoList();
updateGlobalUsage();
