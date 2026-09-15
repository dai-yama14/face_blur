# Docker セットアップ手順（実作業用チェックリスト）

各デスクトップ（RTX 5080 / 5060）で movieedit2 を Docker で動かすための手順。
**上から順に実行すれば終わる**ように書いてある。設計の全体像は `DEPLOY.md` を参照。

> 前提の確認（2026-07-11 時点の実測）
> - 現行機: RTX 5080 / NVIDIA ドライバ **610.47** / VRAM 16GB
> - NAS へ置く一式は約 **420MB**（`models/` 309MB + `app/tddfa/weights/` 84MB + コード 26MB）
> - ⚠️ **このプロジェクトは git リポジトリではない**。`git clone` は使えないので rsync でコピーする

---

## 0. 事前の掃除（任意・NAS転送を 74MB 減らせる）

`app/tddfa/weights/resnet22.pth`（74MB）は**検証の結果不採用**になったモデル。
既定では読まれないので削除してよい（再検証したくなったら再DL可）。

```bash
rm app/tddfa/weights/resnet22.pth
```

残すなら NAS にもコピーが必要。どちらでも動く。

---

## 1. NAS（中古ノート）側

### 1-1. OS を入れる
`Ubuntu Server 24.04 LTS` を推奨（`DEPLOY.md` の比較表を参照）。

インストール後、**フタを閉じてもスリープしない**ようにする:

```bash
sudo sed -i 's/^#\?HandleLidSwitch=.*/HandleLidSwitch=ignore/' /etc/systemd/logind.conf
sudo systemctl restart systemd-logind
```

### 1-2. IP を固定する
ルータの DHCP 予約、または静的 IP。**この IP を後で `.env` の `NAS_HOST` に書く。**

```bash
ip -4 addr show | grep inet    # 現在の IP を確認
```

### 1-3. Samba 共有を作る

```bash
sudo apt update && sudo apt install -y samba rsync
sudo mkdir -p /srv/movieedit2

# 共有ユーザー（パスワードは後で .env の NAS_PASS に使う）
sudo useradd -M -s /usr/sbin/nologin movieedit
sudo smbpasswd -a movieedit

sudo tee -a /etc/samba/smb.conf >/dev/null <<'EOF'

[movieedit2]
   path = /srv/movieedit2
   browseable = yes
   read only = no
   valid users = movieedit
   force user = movieedit
   create mask = 0664
   directory mask = 0775
EOF
sudo systemctl restart smbd
```

### 1-4. movieedit2 一式を NAS へ転送する

**現行の作業機（WSL）から実行**する。git ではなく rsync を使う:

```bash
# 作業機で。<NAS_IP> は 1-2 で決めた IP、<nasuser> は NAS の SSH ユーザー
rsync -avz --progress \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude 'output/' \
  ~/projects/movieedit2/ <nasuser>@<NAS_IP>:/tmp/movieedit2/

# NAS 側で所定の場所へ移して所有者を合わせる
ssh <nasuser>@<NAS_IP> '
  sudo rsync -a /tmp/movieedit2/ /srv/movieedit2/ &&
  sudo chown -R movieedit:movieedit /srv/movieedit2 &&
  rm -rf /tmp/movieedit2
'
```

**転送されているか必ず確認する**（欠けると起動時に落ちる）:

```bash
ssh <nasuser>@<NAS_IP> 'ls -la /srv/movieedit2/models/ /srv/movieedit2/app/tddfa/weights/'
```

- `models/sam2.1_hiera_base_plus.pt`（309MB）… SAM2 の重み。**必須**
- `app/tddfa/weights/mb1_120x120.pth`（13MB）… 3DDFA の重み。**必須**

> `input/` の動画（数GB）は必要なぶんだけ後から NAS に置けばよい。
> 上の rsync には含まれるので、大きすぎるなら `--exclude 'input/'` を足す。

### 1-5. `.env` を NAS 側に置く（APIキー）

アプリは `/app/.env`（＝NAS 上の `.env`）を読む（`run_editor.py` の `_load_env`）。
Claude 自動クリックを使うならここにキーが要る。

```bash
ssh <nasuser>@<NAS_IP> 'sudo -u movieedit tee /srv/movieedit2/.env' <<'EOF'
ANTHROPIC_API_KEY=sk-ant-...
EOF
```

使わないなら空でよい（自動クリックが無効になるだけで、追跡・書き出しは動く）。

---

## 2. 各デスクトップ（A / B 共通）

### 2-1. NVIDIA ドライバを最新化
RTX 50xx（Blackwell）は新しいドライバが必要。現行機は **610.47** で動作確認済み。

### 2-2. Docker Desktop を入れる
- インストール後、設定 → General →「Use WSL 2 based engine」を **ON**
- WSL 内に Python や CUDA を手で入れる必要は**ない**（全部イメージに入る）

### 2-3. GPU がコンテナから見えるか確認（★ここを飛ばさない）

```powershell
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

→ **RTX 5080 / 5060 が表示されること**。ここで失敗するなら先に進んでも必ず落ちる。
表示されない場合はドライバか Docker Desktop の GPU 連携を疑う。

### 2-4. compose 一式を各デスクトップに置く

イメージのビルドに必要なのは `Dockerfile` / `docker-compose.yml` / `requirements.txt` /
`.env.example` の4つだけ（`.dockerignore` により、コードやモデルはイメージに入らない）。

NAS の共有をエクスプローラで開いてこの4ファイルをコピーするのが手っ取り早い。

### 2-5. `.env` を作る（★ NAS 側の .env とは別物）

**紛らわしいので注意**: `.env` は2箇所にあり、役割が違う。

| 場所 | 読む人 | 中身 |
|---|---|---|
| **デスクトップの compose と同じ階層** | docker compose | `NAS_HOST` / `NAS_SHARE` / `NAS_USER` / `NAS_PASS` |
| **NAS 上の `/srv/movieedit2/.env`** | アプリ本体（`/app/.env`） | `ANTHROPIC_API_KEY` |

デスクトップ側:

```powershell
copy .env.example .env
```

`.env` を編集して、1-2 / 1-3 で決めた値を入れる:

```
NAS_HOST=192.168.1.10     # NAS の固定IP
NAS_SHARE=movieedit2      # smb.conf の [movieedit2] と一致させる
NAS_USER=movieedit
NAS_PASS=（smbpasswd で設定したパスワード）
```

### 2-6. ビルドして起動

```powershell
docker compose build      # 初回のみ。数分〜十数分（torch/CUDA を落とすので重い）
docker compose up -d
docker compose logs -f    # 起動ログを確認。Ctrl+C で抜ける
```

→ ブラウザで **http://localhost:8765**

---

## 3. 起動後の動作確認（ここまでやって「完了」）

1. **GPU を使っているか**
   ```powershell
   docker exec movieedit2 nvidia-smi
   ```
   → GPU が見えること。

2. **モデルが読めているか**
   `docker compose logs` に以下が出ること:
   - `[landmarks3d] 3DDFA config: mb1_120x120.yml (arch=mobilenet)`
   - insightface の `find model: .../det_10g.onnx`（初回はダウンロードが走る＝**要ネット**）

3. **実際に1本流す**
   エディタで短いクリップを開き、「🎯 高精度追跡（SAM 2）」→ 書き出しまで通す。
   ここまで通って初めてセットアップ完了。

---

## 4. 更新のしかた

| 変えたもの | 手順 |
|---|---|
| **アプリのコード**（`app/` `static/`） | NAS の `/srv/movieedit2` を更新 → 各デスクトップで `docker compose restart` |
| **依存パッケージ**（`requirements.txt` / Dockerfile） | 各デスクトップで `docker compose build && docker compose up -d` |
| **`.env` の APIキー** | NAS 側の `.env` を直す → `docker compose restart` |

日常のコード修正は **NAS を1回直して各台 restart** だけ。個別コピー不要。

---

## 5. つまずきやすい点

- **GPU が見えない** → 2-3 の確認コマンドで切り分ける。ここが通らないうちは compose を触らない。
- **CIFS マウントに失敗する**（`docker compose up` が volume エラー）
  → `NAS_HOST` / `NAS_SHARE` / 認証情報を確認。NAS で `sudo systemctl status smbd`。
- **`models/` が空** → SAM2 の重みが NAS に無い。1-4 の確認コマンドで見る。
- **自動クリックが効かない** → NAS 側 `/srv/movieedit2/.env` に `ANTHROPIC_API_KEY` があるか。
  （`requirements.txt` に `anthropic` を追加済み。古いイメージを使っている場合は `build` し直す）
- **書き出しが遅い** → apt の ffmpeg は `h264_nvenc` を含まないことがあり、CPU の libx264 に
  自動フォールバックする（`export.py` の `_has_nvenc()`）。出力は正しい。速度だけの問題。
- **同一プロジェクトの同時編集は不可**（ファイルベース・ロックなし）。台ごとに別プロジェクトで運用する。
- ⚠️ **エディタに認証は無い**。ポート開放でインターネットに晒さないこと。
  外から使うなら Tailscale（`DEPLOY.md` 4章）。

---

## 未検証事項（正直な注記）

この手順は現行環境（WSL2 + RTX 5080）の実測と既存の `Dockerfile` / `docker-compose.yml` を
もとに書いたもので、**NAS 実機と Docker Desktop での通し実行はまだ行っていない**。
特に以下は実機で初めて分かる可能性がある:

- `deadsnakes` PPA の `python3.13` が Ubuntu 24.04 ベースイメージで入るか
  （入らない場合は Dockerfile を `python3.12` に読み替える。コードは互換）
- `docker-compose.yml` の `gpus: all` が使えるか
  （古い compose では通らない。その場合はファイル内コメントの `deploy:` 記法に置換）
- CIFS ボリュームの権限（書き込みできるか）

つまずいたら、その箇所を教えてください。
