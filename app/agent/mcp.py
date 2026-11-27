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

    # MCP 客户端集合，负责管理与 MCP 服务器的连接及工具调用
    mcp_clients: MCPClients = Field(default_factory=MCPClients)
    # 可用工具集合，初始化后指向 mcp_clients，使基类的工具调用逻辑直接复用 MCP 工具
    available_tools: MCPClients = None

    max_steps: int = 20
    # 连接类型："stdio"（本地子进程）或 "sse"（远程 Streamable HTTP 服务，
    # 取名 sse 为历史遗留，底层实际使用 MCP 2025-03-26 规范的 Streamable HTTP）
    connection_type: str = "stdio"

    # 本地缓存的各工具 inputSchema 快照，键为工具名；
    # 每次刷新时与服务器最新 schema 对比，用于检测工具的新增/移除/变更
    tool_schemas: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    # 每隔 N 步刷新一次工具列表，以感知 MCP 服务器工具的动态变化
    _refresh_tools_interval: int = 5

    # 执行这些工具后智能体立即终止运行
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
            server_id: 服务器标识，用于在多服务器场景下区分会话与指令
            tool_name_prefix: 是否为工具名添加服务器前缀，避免多个服务器的工具重名冲突
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

        # 将可用工具集合指向 MCP 客户端实例，
        # 这样基类 ToolCallAgent 调用 available_tools 时即透明地转发给 MCP 服务器
        self.available_tools = self.mcp_clients

        # 保存初始工具 schema，作为后续检测工具变更的基准
        await self._refresh_tools()

        # 将系统提示词与服务器提供的工具信息写入记忆：
        # server_id 优先级为 显式 server_id > command > server_url，
        # 与 mcp_clients 内部注册会话时使用的标识保持一致
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
        # （既有工具的 schema 变化无法用差集表达，需逐个对比）
        current_names = set(current_tools.keys())
        previous_names = set(self.tool_schemas.keys())

        added_tools = list(current_names - previous_names)
        removed_tools = list(previous_names - current_names)

        # 检查既有工具的 schema 是否发生变化（深比较 JSON schema 字典）
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
        # 检查 MCP 会话与工具是否可用：连接中断或工具列表为空时直接结束
        if not self.mcp_clients.sessions or not self.mcp_clients.tool_map:
            logger.info("MCP 服务已不可用，结束交互")
            self.state = AgentState.FINISHED
            return False

        # 周期性刷新工具列表，以便在运行过程中感知服务器的动态变化
        # （current_step 从 0 开始，因此第一步也会执行一次刷新）
        if self.current_step % self._refresh_tools_interval == 0:
            await self._refresh_tools()
            # 工具全部消失说明服务已关闭
            if not self.mcp_clients.tool_map:
                logger.info("MCP 服务已关闭，结束交互")
                self.state = AgentState.FINISHED
                return False

        # 交给父类的 think 方法处理（LLM 推理 + 工具选择）
        return await super().think()

    def _should_finish_execution(self, name: str, **kwargs) -> bool:
        """判断工具执行后是否应结束智能体。

        这是父类的钩子方法：工具名不区分大小写，只要为 terminate
        就在本次工具执行完毕后终止智能体运行。
        """
        return name.lower() == "terminate"

    async def cleanup(self) -> None:
        """结束后清理 MCP 连接。"""
        # 仅在存在活跃会话时断开，避免重复清理
        if self.mcp_clients.sessions:
            await self.mcp_clients.disconnect()
            logger.info("MCP 连接已关闭")

    async def run(self, request: Optional[str] = None) -> str:
        """运行智能体，结束后执行清理。

        使用 try/finally 保证：无论正常结束还是中途抛出异常，
        MCP 连接都会被关闭，避免残留子进程或挂起的 SSE 连接。
        """
        try:
            result = await super().run(request)
            return result
        finally:
            await self.cleanup()
