from contextlib import AsyncExitStack
from typing import Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import ImageContent, ListToolsResult, TextContent

from app.logger import logger
from app.tool.base import BaseTool, ToolResult
from app.tool.tool_collection import ToolCollection


class MCPClientTool(BaseTool):
    """客户端侧的工具代理，将调用转发给 MCP 服务器上的对应工具。"""

    session: Optional[ClientSession] = None
    server_id: str = ""  # 所属 MCP 服务器的标识
    original_name: str = ""

    async def execute(self, **kwargs) -> ToolResult:
        """通过远程调用 MCP 服务器来执行该工具。"""
        if not self.session:
            return ToolResult(error="未连接到 MCP 服务器")

        try:
            logger.info(f"正在执行工具：{self.original_name}")
            # MCP 桥接核心：以工具在服务器端的原始名称转发调用
            result = await self.session.call_tool(self.original_name, kwargs)
            # 从返回内容中分别抽取文本项与图片项
            content_str = ", ".join(
                item.text for item in result.content if isinstance(item, TextContent)
            )
            image = next(
                (
                    item.data
                    for item in result.content
                    if isinstance(item, ImageContent)
                ),
                None,
            )
            return ToolResult(
                output=content_str
                or ("已返回图片。" if image else "无输出返回。"),
                base64_image=image,
            )
        except Exception as e:
            return ToolResult(error=f"工具执行出错：{str(e)}")


class MCPClients(ToolCollection):
    """
    连接多个 MCP 服务器的工具集合，通过 Model Context Protocol 管理可用的工具。
    """

    sessions: Dict[str, ClientSession] = {}
    exit_stacks: Dict[str, AsyncExitStack] = {}
    server_instructions: Dict[str, str] = {}
    description: str = "用于与服务器交互的 MCP 客户端工具"

    def __init__(self):
        super().__init__()  # 以空工具列表初始化
        self.name = "mcp"  # 保持原有名称以向后兼容
        self.sessions = {}
        self.exit_stacks = {}
        self.server_instructions = {}

    async def connect_sse(self, server_url: str, server_id: str = "") -> None:
        """通过 SSE 传输协议连接 MCP 服务器。"""
        if not server_url:
            raise ValueError("Server URL is required.")

        server_id = server_id or server_url

        # 新建连接前先断开旧连接，确保状态干净
        if server_id in self.sessions:
            await self.disconnect(server_id)

        exit_stack = AsyncExitStack()
        self.exit_stacks[server_id] = exit_stack

        streams_context = sse_client(url=server_url)
        streams = await exit_stack.enter_async_context(streams_context)
        session = await exit_stack.enter_async_context(ClientSession(*streams))
        self.sessions[server_id] = session

        await self._initialize_and_list_tools(server_id)

    async def connect_stdio(
        self,
        command: str,
        args: List[str],
        server_id: str = "",
        tool_name_prefix: bool = True,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        """通过 stdio 传输协议连接 MCP 服务器。"""
        if not command:
            raise ValueError("Server command is required.")

        server_id = server_id or command

        # 新建连接前先断开旧连接，确保状态干净
        if server_id in self.sessions:
            await self.disconnect(server_id)

        exit_stack = AsyncExitStack()
        self.exit_stacks[server_id] = exit_stack

        server_params = StdioServerParameters(command=command, args=args, env=env)
        stdio_transport = await exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        session = await exit_stack.enter_async_context(ClientSession(read, write))
        self.sessions[server_id] = session

        await self._initialize_and_list_tools(server_id, tool_name_prefix)

    async def _initialize_and_list_tools(
        self, server_id: str, tool_name_prefix: bool = True
    ) -> None:
        """初始化会话，并把服务器上的所有工具注册进工具映射。"""
        session = self.sessions.get(server_id)
        if not session:
            raise RuntimeError(f"Session not initialized for server {server_id}")

        initialization = await session.initialize()
        if initialization.instructions:
            self.server_instructions[server_id] = initialization.instructions
        response = await session.list_tools()

        # 为服务器的每个工具创建对应的客户端代理对象
        for tool in response.tools:
            original_name = tool.name
            # 按需为工具名添加服务器前缀，避免多服务器之间重名冲突
            tool_name = (
                f"mcp_{server_id}_{original_name}"
                if tool_name_prefix
                else original_name
            )
            tool_name = self._sanitize_tool_name(tool_name)

            server_tool = MCPClientTool(
                name=tool_name,
                description=tool.description,
                parameters=tool.inputSchema,
                session=session,
                server_id=server_id,
                original_name=original_name,
            )
            self.tool_map[tool_name] = server_tool

        # 同步工具元组，使其与 tool_map 保持一致
        self.tools = tuple(self.tool_map.values())
        logger.info(
            f"已连接服务器 {server_id}，可用工具：{[tool.name for tool in response.tools]}"
        )

    def _sanitize_tool_name(self, name: str) -> str:
        """规范化工具名称，使其满足 MCPClientTool 的命名要求。"""
        import re

        # 将非法字符替换为下划线
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", name)

        # 合并连续的下划线
        sanitized = re.sub(r"_+", "_", sanitized)

        # 去除首尾的下划线
        sanitized = sanitized.strip("_")

        # 超长时截断到 64 个字符
        if len(sanitized) > 64:
            sanitized = sanitized[:64]

        return sanitized

    async def list_tools(self) -> ListToolsResult:
        """列出所有可用的工具。"""
        tools_result = ListToolsResult(tools=[])
        for session in self.sessions.values():
            response = await session.list_tools()
            tools_result.tools += response.tools
        return tools_result

    async def disconnect(self, server_id: str = "") -> None:
        """断开指定的 MCP 服务器；未提供 server_id 时断开所有服务器。"""
        if server_id:
            if server_id in self.sessions:
                try:
                    exit_stack = self.exit_stacks.get(server_id)

                    # 关闭退出栈，由其统一完成会话与传输的清理
                    if exit_stack:
                        try:
                            await exit_stack.aclose()
                        except RuntimeError as e:
                            if "cancel scope" in str(e).lower():
                                logger.warning(
                                    f"从服务器 {server_id} 断开时出现 cancel scope 错误，将继续清理：{e}"
                                )
                            else:
                                raise

                    # 清理内部引用
                    self.sessions.pop(server_id, None)
                    self.exit_stacks.pop(server_id, None)
                    self.server_instructions.pop(server_id, None)

                    # 移除属于该服务器的所有工具代理
                    self.tool_map = {
                        k: v
                        for k, v in self.tool_map.items()
                        if v.server_id != server_id
                    }
                    self.tools = tuple(self.tool_map.values())
                    logger.info(f"已断开 MCP 服务器 {server_id}")
                except Exception as e:
                    logger.error(f"断开服务器 {server_id} 时出错：{e}")
        else:
            # 按确定顺序逐个断开所有服务器
            for sid in sorted(list(self.sessions.keys())):
                await self.disconnect(sid)
            self.tool_map = {}
            self.tools = tuple()
            logger.info("已断开所有 MCP 服务器")
