import json
from typing import TYPE_CHECKING, Optional

from app.agent.mcp import MCPAgent
from app.logger import logger
from app.prompt.browser import NEXT_STEP_PROMPT
from app.schema import Message
from app.tool.terminate import Terminate


# 仅在类型检查时导入，避免循环导入
if TYPE_CHECKING:
    from app.agent.base import BaseAgent  # 记忆类型定义所在模块


_SANDBOX_BROWSER_TOOL_NAME = "sandbox_browser"


class BrowserContextHelper:
    def __init__(self, agent: "BaseAgent"):
        self.agent = agent
        self._current_base64_image: Optional[str] = None

    async def get_browser_state(self) -> Optional[dict]:
        """获取浏览器当前状态，失败时返回 None。"""
        browser_tool = self.agent.available_tools.get_tool(_SANDBOX_BROWSER_TOOL_NAME)
        if not browser_tool or not hasattr(browser_tool, "get_current_state"):
            logger.warning("未找到 SandboxBrowserTool 或其不支持获取当前状态")
            return None
        try:
            result = await browser_tool.get_current_state()
            if result.error:
                logger.debug(f"获取浏览器状态出错：{result.error}")
                return None
            if hasattr(result, "base64_image") and result.base64_image:
                self._current_base64_image = result.base64_image
            else:
                self._current_base64_image = None
            return json.loads(result.output)
        except Exception as e:
            logger.debug(f"获取浏览器状态失败：{str(e)}")
            return None

    async def format_next_step_prompt(self) -> str:
        """获取浏览器状态并据此格式化下一步提示词。"""
        browser_state = await self.get_browser_state()
        url_info, tabs_info, content_above_info, content_below_info = "", "", "", ""
        results_info = ""  # 结果信息暂为空，如有需要可从 agent 获取

        if browser_state and not browser_state.get("error"):
            url_info = f"\n   URL: {browser_state.get('url', 'N/A')}\n   Title: {browser_state.get('title', 'N/A')}"
            tabs = browser_state.get("tabs", [])
            if tabs:
                tabs_info = f"\n   {len(tabs)} tab(s) available"
            pixels_above = browser_state.get("pixels_above", 0)
            pixels_below = browser_state.get("pixels_below", 0)
            if pixels_above > 0:
                content_above_info = f" ({pixels_above} pixels)"
            if pixels_below > 0:
                content_below_info = f" ({pixels_below} pixels)"

            if self._current_base64_image:
                image_message = Message.user_message(
                    content="Current browser screenshot:",
                    base64_image=self._current_base64_image,
                )
                self.agent.memory.add_message(image_message)
                self._current_base64_image = None  # 写入后清空缓存，避免同一截图被重复添加

        return NEXT_STEP_PROMPT.format(
            url_placeholder=url_info,
            tabs_placeholder=tabs_info,
            content_above_placeholder=content_above_info,
            content_below_placeholder=content_below_info,
            results_placeholder=results_info,
        )

    async def cleanup_browser(self):
        """清理浏览器工具资源。"""
        browser_tool = self.agent.available_tools.get_tool(_SANDBOX_BROWSER_TOOL_NAME)
        if browser_tool and hasattr(browser_tool, "cleanup"):
            await browser_tool.cleanup()


class BrowserAgent(MCPAgent):
    """基于 Browser Use CLI 3.0 MCP 服务器的纯浏览器智能体。"""

    name: str = "browser"
    description: str = "A browser agent that can control a browser to accomplish tasks"
    system_prompt: str = """\
Use Browser Use CLI 3.0 through `browser_exec`. Pass the Python body from CLI
examples as the `code` argument. Use `browser_screenshot` when visual context
is needed. The browser-harness session persists across calls. Use `terminate`
when the task is complete.
"""

    async def initialize(self) -> None:
        await super().initialize(
            connection_type="stdio",
            command="uvx",
            args=["browser-use", "--cli-mcp"],
            server_id="browser_use",
            tool_name_prefix=False,
        )
        self.available_tools.add_tool(Terminate())

    @classmethod
    async def create(cls, **kwargs) -> "BrowserAgent":
        instance = cls(**kwargs)
        await instance.initialize()
        return instance

    async def run(self, request: Optional[str] = None) -> str:
        # 若 MCP 会话尚未建立，先完成初始化再运行
        if not self.mcp_clients.sessions:
            await self.initialize()
        return await super().run(request)
