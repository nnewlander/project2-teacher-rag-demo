from __future__ import annotations

"""
一键启动 API（Windows 友好）。

动机：
- 有些环境在绑定 8000 端口时会遇到 WinError 10013（端口被占用/被拦截）
- 该脚本会按候选端口依次尝试启动
"""

import argparse
import sys

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FastAPI server (tries multiple ports).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="首选端口（占用则自动尝试下一个）")
    parser.add_argument("--reload", action="store_true", help="开发模式自动重载")
    args = parser.parse_args()

    # 候选端口：优先用户指定，其次常用备用
    candidates = [args.port, 8001, 8010, 8080]
    tried = []
    last_err: Exception | None = None

    for p in candidates:
        tried.append(p)
        try:
            uvicorn.run(
                "app.main:app",
                host=args.host,
                port=p,
                reload=args.reload,
                log_level="info",
            )
            return
        except OSError as e:
            last_err = e
            # 继续尝试下一个端口
            continue

    print(f"Failed to bind ports tried: {tried}", file=sys.stderr)
    if last_err:
        raise last_err


if __name__ == "__main__":
    main()

