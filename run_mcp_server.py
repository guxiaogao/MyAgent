# coding: utf-8
# 快捷启动 OpenManus MCP 服务器的入口；引入本模块的同时也顺带解决了其他导入问题。
from app.mcp.server import MCPServer, parse_args


if __name__ == "__main__":
    args = parse_args()

    # 创建并运行服务器（保持原有流程）
    server = MCPServer()
    server.run(transport=args.transport)
