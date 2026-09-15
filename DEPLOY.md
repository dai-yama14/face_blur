# movieedit2 分散デプロイ手順（NAS＋各デスクトップGPU）

> **実作業のチェックリストは [`DOCKER_SETUP.md`](DOCKER_SETUP.md) を参照。**
> このファイルは設計の全体像・OS選定・運用方針をまとめたもの。

## 全体像

```
中古ノート = NAS（GPU不要）           デスクトップA(RTX5080) / B(RTX5060)
  Samba 共有: movieedit2 一式          Docker Desktop で movieedit2 コンテナ
    ├ コード（git）                      docker compose up -d
    ├ models/ (SAM2 .pt)          ← CIFSマウント →  /app（コード＋データ）
    ├ app/tddfa/（3DDFAモデル）          --gpus all → 各自のGPUで検出/追跡/書出し
    └ projects/ input/ output/          http://localhost:8765 で編集
```

- **計算は各デスクトップの GPU**（コンテナは自機のGPUを使う）
- **コード・データは NAS に一元化** → 更新は NAS を1回直して各台 `restart`
- WSL2 の手構築・Python/CUDA/モデル導入は不要（全部イメージ＋NASに集約）

---

## 1. NAS（中古ノート）のOS — おすすめ

| OS | 向き | 備考 |
|---|---|---|
| **Ubuntu Server 24.04 LTS**（おすすめ・既定） | CLIに慣れている（＝現状Ubuntu利用中） | 情報量最多・Samba/Docker/Tailscale全部そのまま。GUI無しで軽い |
| **OpenMediaVault**（Debianベース） | 共有・ユーザをブラウザで管理したい | NAS専用ディストロ。古PC再生の定番。Web UIで SMB/NFS/権限を管理、Dockerプラグインあり |
| **Debian 12** | とにかく軽く・枯れた安定 | Ubuntuよりわずかに軽量。CLI |
| ~~TrueNAS SCALE~~ | 非推奨（今回は） | ZFS/大RAM前提で古い非力ノートには重い |

**結論**: Ubuntu で問題なし。CLIでOKなら **Ubuntu Server 24.04 LTS**、ブラウザでNAS管理したいなら **OpenMediaVault**。
（古いノートなので、フタを閉じてもスリープしない設定＝ `logind.conf` の `HandleLidSwitch=ignore` を忘れずに）

### Samba 共有の用意（Ubuntu/Debian の場合）

```bash
sudo apt update && sudo apt install -y samba rsync
sudo mkdir -p /srv/movieedit2
# movieedit2 一式を配置（コード＋models＋app/tddfa資産＋データ）
#   ⚠ このプロジェクトは git リポジトリではないので git clone は使えない。
#   作業機から rsync で送る（手順は DOCKER_SETUP.md 1-4）
# SAM2チェックポイント models/sam2.1_hiera_base_plus.pt が入っていること

# 共有ユーザー
sudo useradd -M -s /usr/sbin/nologin movieedit
sudo smbpasswd -a movieedit          # ← .env の NAS_PASS に合わせる
sudo chown -R movieedit:movieedit /srv/movieedit2

# /etc/samba/smb.conf の末尾に追記
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

- NAS の IP は**固定**にしておく（ルータのDHCP予約 or 静的設定）。`.env` の `NAS_HOST` に使う。
- （任意）insightface のモデルを事前配置してオフライン初回起動可に:
  現行機の `~/.insightface/models/`（buffalo_l 等）を NAS 経由で各コンテナの
  `insightface_cache` ボリュームへ入れておくと初回DL不要。未配置でも初回に自動DL（要ネット）。

---

## 2. 各デスクトップ（A/B 共通）

### 2-1. 前提（1回だけ）
1. **NVIDIA ドライバを最新化**（RTX 50xx はなるべく新しいドライバに）
2. **Docker Desktop をインストール**（WSL2バックエンドは自動。Ubuntuディストロの手構築は不要）
3. Docker Desktop 設定 → General「Use WSL 2 based engine」ON、Resources → GPU 連携が有効

### 2-2. GPU がコンテナから見えるか確認
```powershell
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```
→ RTX 5080 / 5060 が表示されれば OK。

### 2-3. 起動
```powershell
# このリポジトリ（compose 一式）を各デスクトップに置く（軽いので git clone でOK）
cp .env.example .env      # NAS_HOST/SHARE/USER/PASS を自機用に記入
docker compose build      # 初回のみ（依存イメージを作る。数分〜十数分）
docker compose up -d
```
→ ブラウザで **http://localhost:8765**。追跡/書き出しは自機のGPUで走る。

---

## 3. 更新フロー（← ボトルネック解消の要）

| 変えたもの | 手順 |
|---|---|
| **エディタのコード**（app/ や static/） | NAS の `/srv/movieedit2` を更新（`git pull` 等）→ 各デスクトップで `docker compose restart` だけ |
| **依存パッケージ**（requirements / torch） | Dockerfile/requirements を更新 → 各デスクトップで `docker compose build && docker compose up -d`（または1台でビルドしイメージを配布） |

- 日常のコード修正は **NASで1回 + 各台restart**。各マシンへ個別コピー不要・環境ドリフトなし。
- `render.py` を変えたら restart で反映（従来「サーバ再起動が必要」と同じ）。`static/*` はブラウザ再読込。

---

## 4. リモートアクセス（別ネットワークから・任意）

- **Tailscale** を NAS・A・B・手元端末に入れる → どこからでも `http://<デスクトップのTailscale名>:8765`。
  ポート開放不要・暗号化・NAT越え。GPUデスクトップは起動しておく（Tailscaleの Wake-on-LAN 併用可）。
- ⚠️ エディタは**認証なし**。素のポート開放でインターネット公開はしないこと（VPN or 認証付きトンネル必須）。

---

## 5. 注意点

- **書き出しの NVENC**: apt の ffmpeg は h264_nvenc を含まない場合がある。その時はコードが
  自動で libx264（CPU）にフォールバックして正しく出力する（`export.py` の `_has_nvenc()`）。
  GPUエンコードを効かせたいなら nvenc 入りの ffmpeg（BtbN 静的ビルド等）をイメージに差し替える。
  ※ 重い検出/追跡は GPU で動くので、フォールバックしても実用上の影響は書き出し速度のみ。
- **同一プロジェクトの同時編集は不可**（ファイルベース・ロックなし）。別プロジェクト運用ならOK。
- **CIFSの権限**: 書き込みは `force user`/`uid=0` を合わせている。うまく書けない時は共有側の権限を確認。
- **初回のみネット**: insightface のモデル自動DL。事前配置すればオフライン可。
