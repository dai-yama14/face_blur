# app/tddfa — 3DDFA_V2（ベンダリング）

横顔・大ポーズ対応の 3D 顔ランドマーク（iBUG 68点）用に、
[3DDFA_V2](https://github.com/cleardusk/3DDFA_V2)（MIT License）から
**推論に必要な最小サブセット**のみを取り込んだもの。

## 取り込んだもの
- `TDDFA.py` — 推論本体（torch パス）
- `models/` — MobileNet 等のバックボーン
- `bfm/` — BFM（3D 顔モデル）ローダ
- `utils/` — io / functions / tddfa_util のみ
- `configs/` — bfm_noneck_v3.pkl, tri.pkl, param_mean_std, mb1_120x120.yml, indices/ncc
- `weights/mb1_120x120.pth` — 学習済み重み

## 取り込んでいないもの
FaceBoxes（検出器 — InsightFace を使うため不要）、Sim3DR（レンダリング）、
デモ・学習スクリプト。

## 使い方
本体からは `app/landmarks3d.py` 経由で呼ぶ（直接 import しない）。
InsightFace の bbox を入力に 68点3D を返す。モデル欠落・失敗時は None を返し、
呼び出し側が 106点式へフォールバックする。

## 実行時依存
torch / torchvision / pyyaml（いずれも既存環境に導入済み）。

## パス解決の注意
`TDDFA.py` は cfg の相対パスを cwd 基準で解決するため、`landmarks3d._init()`
で checkpoint_fp / bfm_fp / param_mean_std_fp を絶対パス化している
（chdir を使わずスレッド安全にするため）。`bfm/bfm.py` は自ファイル基準で
`../configs/tri.pkl` を読むので、この `configs/` との相対配置を保つこと。
