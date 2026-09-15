# ============================================================
# movieedit2 - 顔ブラーエディター 実行イメージ
# 各デスクトップ(RTX 5080 / 5060)のローカルGPUで動かす。
# コード・モデル・データはイメージに焼かず、NAS からバインドマウントする
# （＝更新は NAS 側を1回直して各台 restart するだけ）。
# ============================================================

# CUDA 12.8 + cuDNN ランタイム。
#  - RTX 50xx(Blackwell, sm_120) は CUDA 12.8 系で対応
#  - onnxruntime-gpu 1.24 が cuDNN9/CUDA12 を要求するため cudnn 版ベースを使う
FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Python 3.13（現行環境に一致）＋ ffmpeg(書き出し) ＋ OpenCV ランタイム依存
#  ※ deadsnakes で 3.13 が入らない場合は python3.12 系に読み替え可（コード互換）
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common curl ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 python3.13-dev python3.13-venv \
        ffmpeg libgl1 libglib2.0-0 \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13 \
    && ln -sf /usr/bin/python3.13 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# PyTorch は CUDA 12.8 専用 index から（Blackwell 対応ビルド = torch 2.11 + cu128）
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
        torch==2.11.0 torchvision==0.26.0

# 残りの依存（PyPI）
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

# アプリ本体は /app に NAS からマウントされる（COPY しない）
WORKDIR /app
EXPOSE 8765

# 既定は 0.0.0.0 待受（コンテナ外＝ホストの localhost:8765 から到達可能に）
CMD ["python", "run_editor.py", "--host", "0.0.0.0", "--port", "8765"]
