"""
run_editor.py - 顔ブラーエディター起動スクリプト

Usage:
  python run_editor.py [--port 8765] [--host 127.0.0.1]

起動後、ブラウザで http://localhost:8765 を開く。
"""
import argparse
import os
from pathlib import Path

import uvicorn


def load_env():
    """プロジェクト直下の .env を読み込む（ANTHROPIC_API_KEY など）。"""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key, value)


def main():
    load_env()
    parser = argparse.ArgumentParser(description="顔ブラーエディターを起動")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--dev", action="store_true",
        help="開発モード: app/ 配下の .py を編集すると自動でサーバー再起動"
             "（実行中のジョブは中断されるので、追跡中は使わないこと）")
    args = parser.parse_args()

    print(f"\n  顔ブラーエディター起動: http://localhost:{args.port}\n")
    uvicorn.run("app.main:app", host=args.host, port=args.port,
                log_level="warning", reload=args.dev)


if __name__ == "__main__":
    main()
