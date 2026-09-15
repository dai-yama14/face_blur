# NAS セットアップ手順（Ubuntu Server / 中古ノート）

movieedit2 を各デスクトップ(RTX5080/5060)から共有するための SMB(Samba) NAS を作る。
上から順にコピペで実行。値は docker-compose の `.env` と揃えること。

| 項目 | 値（例。自分の環境に合わせる） |
|---|---|
| NAS 固定IP | `192.168.1.10` |
| 共有名 | `movieedit2` |
| 共有パス | `/srv/movieedit2` |
| 共有ユーザー | `movieedit` |

---

## 0. 基本更新

```bash
sudo apt update && sudo apt -y upgrade
sudo timedatectl set-timezone Asia/Tokyo
```

---

## 1. 固定IP（重要：`.env` の NAS_HOST に使う）

DHCPだとIPが変わって各デスクトップから繋がらなくなるので固定する。

```bash
ip -4 a          # インターフェース名を確認（例: enp3s0 / eth0）と現在のIP
ip r | grep default   # ゲートウェイ確認（例: 192.168.1.1）
```

`/etc/netplan/` の yaml を編集（ファイル名は環境依存。`ls /etc/netplan/`）:

```bash
sudo nano /etc/netplan/50-cloud-init.yaml
```

内容（インターフェース名・IP・GWは自分の値に）:

```yaml
network:
  version: 2
  ethernets:
    enp3s0:                     # ← ip a で見た名前
      dhcp4: no
      addresses: [192.168.1.10/24]
      routes:
        - to: default
          via: 192.168.1.1      # ← ゲートウェイ
      nameservers:
        addresses: [192.168.1.1, 8.8.8.8]
```

```bash
sudo netplan apply
ip -4 a          # 192.168.1.10 になっていればOK
```

※ ルータ側でこのIPを DHCP予約しておくとより確実。

---

## 2. フタを閉じてもスリープしない（ノート必須）

```bash
sudo nano /etc/systemd/logind.conf
```

以下を設定（行頭の # を外して値を変更）:

```
HandleLidSwitch=ignore
HandleLidSwitchExternalPower=ignore
HandleLidSwitchDocked=ignore
```

```bash
sudo systemctl restart systemd-logind
```

---

## 3. Samba と git を導入

```bash
sudo apt -y install samba git rsync
```

---

## 4. movieedit2 一式を配置

共有フォルダを作り、**コード＋モデル＋データ**を置く。

```bash
sudo mkdir -p /srv/movieedit2
```

### 方法A（推奨）：現用機(5080)から丸ごとコピー
モデルや app/tddfa 資産も確実に来るので楽。5080機側で実行:

```bash
# 5080機（WSL）で。大きい output/ を除いて転送する例:
rsync -av --exclude 'output/' --exclude '__pycache__/' \
  /home/yamda/projects/movieedit2/  movieedit@192.168.1.10:/srv/movieedit2/
```

### 方法B：git clone ＋ モデルだけ別途コピー
```bash
sudo git clone <リポジトリ> /srv/movieedit2
# ↓ git に入っていない大物を現用機からコピー（必須）
#   models/sam2.1_hiera_base_plus.pt  … SAM2チェックポイント(約323MB)
#   app/tddfa/weights, app/tddfa/configs の pkl/pth 一式
```

### 配置後の必須チェック
```bash
ls -lh /srv/movieedit2/models/sam2.1_hiera_base_plus.pt   # 約323MBがあること
ls /srv/movieedit2/app/tddfa/weights /srv/movieedit2/app/tddfa/configs
ls /srv/movieedit2/run_editor.py                          # コード本体
```

データ用の空フォルダも用意（無ければ）:
```bash
sudo mkdir -p /srv/movieedit2/{projects,input,output}
```

---

## 5. 共有ユーザーと権限

```bash
# ログイン不可のサービス用ユーザー
sudo useradd -M -s /usr/sbin/nologin movieedit

# Samba のパスワード設定（← .env の NAS_PASS に一致させる）
sudo smbpasswd -a movieedit
sudo smbpasswd -e movieedit

# 所有権を共有ユーザーへ
sudo chown -R movieedit:movieedit /srv/movieedit2
```

---

## 6. Samba 共有設定

```bash
sudo tee -a /etc/samba/smb.conf >/dev/null <<'EOF'

[movieedit2]
   path = /srv/movieedit2
   browseable = yes
   read only = no
   valid users = movieedit
   force user = movieedit
   force group = movieedit
   create mask = 0664
   directory mask = 0775
EOF

# 設定の文法チェック
testparm

# 反映
sudo systemctl restart smbd nmbd
sudo systemctl enable smbd nmbd
```

---

## 7. ファイアウォール（ufw を使っている場合）

```bash
sudo ufw allow OpenSSH
sudo ufw allow samba
sudo ufw --force enable
sudo ufw status
```

---

## 8. 動作確認

### NAS 自身から
```bash
smbclient -L localhost -U movieedit          # 共有一覧に movieedit2 が出る
```

### 各デスクトップ（Windows）から
エクスプローラのアドレス欄に:
```
\\192.168.1.10\movieedit2
```
→ movieedit / パスワードで開ければ成功。ファイルの作成・削除ができるか確認。

これが通れば、各デスクトップの `.env` を:
```
NAS_HOST=192.168.1.10
NAS_SHARE=movieedit2
NAS_USER=movieedit
NAS_PASS=（設定したパスワード）
```
にして `docker compose up -d` → コンテナが `/app` にこの共有をマウントして起動する。

---

## 9.（任意）insightface モデルを事前配置してオフライン初回起動

初回検出時のモデル自動DL（要ネット）を避けたい場合、現用機の
`~/.insightface/models/`（buffalo_l など）を NAS に置いておき、
各デスクトップの初回だけコンテナ内 `/root/.insightface` へ展開しておくと良い（任意）。

---

## 10.（任意・後日）リモートアクセス（Tailscale）

別ネットワークから使うとき用。今すぐは不要。

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
```

各デスクトップと手元端末にも Tailscale を入れれば、外出先から
`http://<デスクトップのTailscale名>:8765` で編集できる（ポート開放不要・暗号化）。
※ エディタは認証なしなので、素のポート開放でのネット公開は禁止。

---

## トラブル時のチェック

- 繋がらない → `sudo systemctl status smbd` / NASのIP(`ip a`) / `ufw status` / 同一LANか
- 書けない → `/srv/movieedit2` の所有者が movieedit か、`force user` が効いているか
- IPが変わった → 手順1の固定IP or ルータのDHCP予約を確認
