from typing import Any, Dict, List, Optional, Tuple

from pydantic import Field

from app.agent.toolcall import ToolCallAgent
from app.logger import logger
from app.prompt.mcp import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import AgentState, Message
from app.tool.mcp import MCPClients


class MCPAgent(ToolCallAgent):
    """用于与 MCP（Model Context Protocol）服务器交互的智能体。

    该智能体通过 SSE 或 stdio 传输方式连接 MCP 服务器，
    并将服务器提供的工具接入智能体的工具接口。
    """

    name: str = "mcp_agent"
    description: str = "An agent that connects to an MCP server and uses its tools."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    # 初始化 MCP 工具集合
    mcp_clients: MCPClients = Field(default_factory=MCPClients)
    available_tools: MCPClients = None  # 将在 initialize() 中设置

    max_steps: int = 20
    connection_type: str = "stdio"  # 连接类型："stdio" 或 "sse"

    # 记录各工具的 schema，用于检测工具变更
    tool_schemas: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    _refresh_tools_interval: int = 5  # 每隔 N 步刷新一次工具列表

    # 触发终止的特殊工具名称
    special_tool_names: List[str] = Field(default_factory=lambda: ["terminate"])

    async def initialize(
        self,
        connection_type: Optional[str] = None,
        server_url: Optional[str] = None,
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        server_id: str = "",
        tool_name_prefix: bool = True,
    ) -> None:
        """初始化 MCP 连接。

        Args:
            connection_type: 连接类型（"stdio" 或 "sse"）
            server_url: MCP 服务器 URL（用于 SSE 连接）
            command: 要运行的命令（用于 stdio 连接）
            args: 命令参数（用于 stdio 连接）
        """
        if connection_type:
            self.connection_type = connection_type

        # 根据连接类型连接 MCP 服务器
        if self.connection_type == "sse":
            if not server_url:
                raise ValueError("Server URL is required for SSE connection")
            await self.mcp_clients.connect_sse(
                server_url=server_url, server_id=server_id
            )
        elif self.connection_type == "stdio":
            if not command:
                raise ValueError("Command is required for stdio connection")
            await self.mcp_clients.connect_stdio(
                command=command,
                args=args or [],
                server_id=server_id,
                tool_name_prefix=tool_name_prefix,
            )
        else:
            raise ValueError(f"Unsupported connection type: {self.connection_type}")

        # 将可用工具集合指向 MCP 客户端实例
        self.available_tools = self.mcp_clients

        # 保存初始工具 schema
        await self._refresh_tools()

        # 在系统消息中说明可用工具
        tool_names = list(self.mcp_clients.tool_map.keys())
        tools_info = ", ".join(tool_names)
        resolved_server_id = server_id or command or server_url or ""
        instructions = self.mcp_clients.server_instructions.get(resolved_server_id, "")

        # 写入系统提示词与可用工具信息
        self.memory.add_message(
            Message.system_message(
                f"{self.system_prompt}\n\nAvailable MCP tools: {tools_info}"
                + (
                    f"\n\nMCP server instructions:\n{instructions}"
                    if instructions
                    else ""
                )
            )
        )

    async def _refresh_tools(self) -> Tuple[List[str], List[str]]:
        """从 MCP 服务器刷新可用工具列表。

        Returns:
            (新增工具, 移除工具) 组成的元组
        """
        if not self.mcp_clients.sessions:
            return [], []

        # 直接从服务器获取当前工具 schema
        response = await self.mcp_clients.list_tools()
        current_tools = {tool.name: tool.inputSchema for tool in response.tools}

        # 通过集合差集计算新增与移除的工具
        current_names = set(current_tools.keys())
        previous_names = set(self.tool_schemas.keys())

        added_tools = list(current_names - previous_names)
        removed_tools = list(previous_names - current_names)

        # 检查既有工具的 schema 是否发生变化
        changed_tools = []
        for name in current_names.intersection(previous_names):
            if current_tools[name] != self.tool_schemas.get(name):
                changed_tools.append(name)

        # 更新本地缓存的工具 schema
        self.tool_schemas = current_tools

        # 记录变更日志并将变更写入记忆
        if added_tools:
            logger.info(f"新增 MCP 工具：{added_tools}")
            self.memory.add_message(
                Message.system_message(f"New tools available: {', '.join(added_tools)}")
            )
        if removed_tools:
            logger.info(f"移除了 MCP 工具：{removed_tools}")
            self.memory.add_message(
                Message.system_message(
                    f"Tools no longer available: {', '.join(removed_tools)}"
                )
            )
        if changed_tools:
            logger.info(f"发生变化的 MCP 工具：{changed_tools}")

        return added_tools, removed_tools

    async def think(self) -> bool:
        """处理当前状态并决策下一步动作。"""
        # 检查 MCP 会话与工具是否可用
        if not self.mcp_clients.sessions or not self.mcp_clients.tool_map:
            logger.info("MCP 服务已不可用，结束交互")
            self.state = AgentState.FINISHED
            return False

        # 周期性刷新工具列表
        if self.current_step % self._refresh_tools_interval == 0:
            await self._refresh_tools()
            # 工具全部消失说明服务已关闭
            if not self.mcp_clients.tool_map:
                logger.info("MCP 服务已关闭，结束交互")
                self.state = AgentState.FINISHED
                return False

        # 交给父类的 think 方法处理
        return await super().think()

    def _should_finish_execution(self, name: str, **kwargs) -> bool:
        """判断工具执行后是否应结束智能体"""
        # 工具名为 terminate 时终止
        return name.lower() == "terminate"

    async def cleanup(self) -> None:
        """结束后清理 MCP 连接。"""
        if self.mcp_clients.sessions:
            await self.mcp_clients.disconnect()
            logger.info("MCP 连接已关闭")

    async def run(self, request: Optional[str] = None) -> str:
        """运行智能体，结束后执行清理。"""
        try:
            result = await super().run(request)
            return result
        finally:
            # 即使发生异常也确保执行清理
            await self.cleanup()
