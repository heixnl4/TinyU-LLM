"""
TinyU-LLM Web 后端启动脚本
用法:
    python start_server.py          # 默认启动在 0.0.0.0:8000
    python start_server.py --port 8080
"""
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="启动 TinyU-LLM Web 后端服务")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--reload", action="store_true", help="开启热重载（开发模式）")
    args = parser.parse_args()

    print(f"正在启动 TinyU-LLM Web 后端...")
    print(f"地址: http://{args.host}:{args.port}")
    print(f"API 文档: http://{args.host}:{args.port}/docs")
    print("-" * 50)

    uvicorn.run(
        "web.api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
