from typing import Dict, List, Optional

from pydantic import Field, model_validator

from app.agent.browser import BrowserContextHelper
from app.agent.toolcall import ToolCallAgent
from app.config import config
from app.daytona.sandbox import create_sandbox, delete_sandbox
from app.daytona.tool_base import SandboxToolsBase
from app.logger import logger
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.tool.ask_human import AskHuman
from app.tool.mcp import MCPClients, MCPClientTool
from app.tool.sandbox.sb_browser_tool import SandboxBrowserTool
from app.tool.sandbox.sb_files_tool import SandboxFilesTool
from app.tool.sandbox.sb_shell_tool import SandboxShellTool
from app.tool.sandbox.sb_vision_tool import SandboxVisionTool


class SandboxManus(ToolCallAgent):
    """支持沙箱工具与 MCP 工具的通用智能体。"""

    name: str = "SandboxManus"
    description: str = "A versatile agent that can solve various tasks using multiple sandbox-tools including MCP-based tools"

    system_prompt: str = SYSTEM_PROMPT.format(directory=config.workspace_root)
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 10000
    max_steps: int = 20

    # 用于访问远程工具的 MCP 客户端
    mcp_clients: MCPClients = Field(default_factory=MCPClients)

    # 向工具集合中添加通用工具
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            # PythonExecute(),
            # BrowserUseTool(),
            # StrReplaceEditor(),
            AskHuman(),
            Terminate(),
        )
    )

    special_tool_names: list[str] = Field(default_factory=lambda: [Terminate().name])
    browser_context_helper: Optional[BrowserContextHelper] = None

    # 记录已连接的 MCP 服务器
    connected_servers: Dict[str, str] = Field(
        default_factory=dict
    )  # server_id -> url/command 的映射
    _initialized: bool = False
    sandbox_link: Optional[dict[str, dict[str, str]]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def initialize_helper(self) -> "SandboxManus":
        """以同步方式初始化基础组件。"""
        self.browser_context_helper = BrowserContextHelper(self)
        return self

    @classmethod
    async def create(cls, **kwargs) -> "SandboxManus":
        """工厂方法：创建并正确初始化一个 SandboxManus 实例。"""
        instance = cls(**kwargs)
        await instance.initialize_mcp_servers()
        await instance.initialize_sandbox_tools()
        instance._initialized = True
        return instance

    async def initialize_sandbox_tools(
        self,
        password: str = config.daytona.VNC_password,
    ) -> None:
        try:
            # 创建新沙箱
            if password:
                sandbox = create_sandbox(password=password)
                self.sandbox = sandbox
            else:
                raise ValueError("password must be provided")
            vnc_link = sandbox.get_preview_link(6080)
            website_link = sandbox.get_preview_link(8080)
            vnc_url = vnc_link.url if hasattr(vnc_link, "url") else str(vnc_link)
            website_url = (
                website_link.url if hasattr(website_link, "url") else str(website_link)
            )

            # 从创建的沙箱对象中获取实际的 sandbox_id
            actual_sandbox_id = sandbox.id if hasattr(sandbox, "id") else "new_sandbox"
            if not self.sandbox_link:
                self.sandbox_link = {}
            self.sandbox_link[actual_sandbox_id] = {
                "vnc": vnc_url,
                "website": website_url,
            }
            logger.info(f"VNC 链接：{vnc_url}")
            logger.info(f"网站链接：{website_url}")
            SandboxToolsBase._urls_printed = True
            sb_tools = [
                SandboxBrowserTool(sandbox),
                SandboxFilesTool(sandbox),
                SandboxShellTool(sandbox),
                SandboxVisionTool(sandbox),
            ]
            self.available_tools.add_tools(*sb_tools)

        except Exception as e:
            logger.error(f"初始化沙箱工具时出错：{e}")
            raise

    async def initialize_mcp_servers(self) -> None:
        """初始化与已配置 MCP 服务器的连接。"""
        for server_id, server_config in config.mcp_config.servers.items():
            try:
                if server_config.type == "sse":
                    if server_config.url:
                        await self.connect_mcp_server(server_config.url, server_id)
                        logger.info(
                            f"已连接到 MCP 服务器 {server_id}（地址：{server_config.url}）"
                        )
                elif server_config.type == "stdio":
                    if server_config.command:
                        await self.connect_mcp_server(
                            server_config.command,
                            server_id,
                            use_stdio=True,
                            stdio_args=server_config.args,
                        )
                        logger.info(
                            f"已通过命令 {server_config.command} 连接到 MCP 服务器 {server_id}"
                        )
            except Exception as e:
                logger.error(f"连接 MCP 服务器 {server_id} 失败：{e}")

    async def connect_mcp_server(
        self,
        server_url: str,
        server_id: str = "",
        use_stdio: bool = False,
        stdio_args: List[str] = None,
    ) -> None:
        """连接到 MCP 服务器并注册其提供的工具。"""
        if use_stdio:
            await self.mcp_clients.connect_stdio(
                server_url, stdio_args or [], server_id
            )
            self.connected_servers[server_id or server_url] = server_url
        else:
            await self.mcp_clients.connect_sse(server_url, server_id)
            self.connected_servers[server_id or server_url] = server_url

        # 仅将当前服务器新提供的工具加入可用工具集合
        new_tools = [
            tool for tool in self.mcp_clients.tools if tool.server_id == server_id
        ]
        self.available_tools.add_tools(*new_tools)

    async def disconnect_mcp_server(self, server_id: str = "") -> None:
        """断开 MCP 服务器连接并移除其工具。"""
        await self.mcp_clients.disconnect(server_id)
        if server_id:
            self.connected_servers.pop(server_id, None)
        else:
            self.connected_servers.clear()

        # 重建工具集合：剔除已断开服务器的工具，保留本地工具与其余 MCP 工具
        base_tools = [
            tool
            for tool in self.available_tools.tools
            if not isinstance(tool, MCPClientTool)
        ]
        self.available_tools = ToolCollection(*base_tools)
        self.available_tools.add_tools(*self.mcp_clients.tools)

    async def delete_sandbox(self, sandbox_id: str) -> None:
        """根据 ID 删除沙箱。"""
        try:
            await delete_sandbox(sandbox_id)
            logger.info(f"沙箱 {sandbox_id} 删除成功")
            if sandbox_id in self.sandbox_link:
                del self.sandbox_link[sandbox_id]
        except Exception as e:
            logger.error(f"删除沙箱 {sandbox_id} 时出错：{e}")
            raise e

    async def cleanup(self):
        """清理 SandboxManus 智能体的资源。"""
        if self.browser_context_helper:
            await self.browser_context_helper.cleanup_browser()
        # 仅在已完成初始化时才断开 MCP 连接并删除沙箱
        if self._initialized:
            await self.disconnect_mcp_server()
            await self.delete_sandbox(self.sandbox.id if self.sandbox else "unknown")
            self._initialized = False

    async def think(self) -> bool:
        """结合当前上下文处理状态并决策下一步动作。"""
        if not self._initialized:
            await self.initialize_mcp_servers()
            self._initialized = True

        # 保存原始提示词；检查最近几条消息中是否调用了沙箱浏览器工具
        original_prompt = self.next_step_prompt
        recent_messages = self.memory.messages[-3:] if self.memory.messages else []
        browser_in_use = any(
            tc.function.name == SandboxBrowserTool().name
            for msg in recent_messages
            if msg.tool_calls
            for tc in msg.tool_calls
        )

        # 浏览器正在使用时，临时注入实时浏览器上下文提示词
        if browser_in_use:
            self.next_step_prompt = (
                await self.browser_context_helper.format_next_step_prompt()
            )

        result = await super().think()

        # 恢复原始提示词，避免浏览器上下文提示词影响后续步骤
        self.next_step_prompt = original_prompt

        return result
