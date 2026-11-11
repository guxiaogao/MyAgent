import argparse
import asyncio
import logging
from typing import Optional

import httpx
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from dotenv import load_dotenv

from app.tool.browser_use_tool import _BROWSER_DESCRIPTION
from app.tool.str_replace_editor import _STR_REPLACE_EDITOR_DESCRIPTION
from app.tool.terminate import _TERMINATE_DESCRIPTION

from .agent import A2AManus
from .agent_executor import ManusExecutor


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main(host: str = "localhost", port: int = 10000):
    """启动 Manus 智能体服务器。"""
    try:
        capabilities = AgentCapabilities(streaming=False, pushNotifications=True)
        skills = [
            AgentSkill(
                id="Python Execute",
                name="Python 执行工具",
                description="执行 Python 代码字符串。注意：只有 print 输出可见，函数返回值不会被捕获。请使用 print 语句查看结果。",
                tags=["执行 Python 代码"],
                examples=[
                    "执行 Python 代码：'''python \n Print('Hello World') \n '''"
                ],
            ),
            AgentSkill(
                id="Browser use",
                name="浏览器使用工具",
                description=_BROWSER_DESCRIPTION,
                tags=["使用浏览器"],
                examples=["go_to 'https://www.google.com'"],
            ),
            AgentSkill(
                id="Replace String",
                name="字符串替换工具",
                description=_STR_REPLACE_EDITOR_DESCRIPTION,
                tags=["操作文件"],
                examples=["把 'file.txt' 中的 'old' 替换为 'new'"],
            ),
            AgentSkill(
                id="Ask human",
                name="询问人类工具",
                description="使用此工具向人类求助。",
                tags=["向人类求助"],
                examples=["询问人类：'现在几点了？'"],
            ),
            AgentSkill(
                id="terminate",
                name="终止工具",
                description=_TERMINATE_DESCRIPTION,
                tags=["终止任务"],
                examples=["terminate"],
            ),
            # 按需添加更多技能
        ]

        agent_card = AgentCard(
            name="Manus 智能体",
            description="一个通用智能体，能够使用多种工具（包括基于 MCP 的工具）解决各类任务",
            url=f"http://{host}:{port}/",
            version="1.0.0",
            defaultInputModes=A2AManus.SUPPORTED_CONTENT_TYPES,
            defaultOutputModes=A2AManus.SUPPORTED_CONTENT_TYPES,
            capabilities=capabilities,
            skills=skills,
        )

        httpx_client = httpx.AsyncClient()
        request_handler = DefaultRequestHandler(
            agent_executor=ManusExecutor(
                agent_factory=lambda: A2AManus.create(max_steps=3)
            ),
            task_store=InMemoryTaskStore(),
            push_notifier=InMemoryPushNotifier(httpx_client),
        )

        server = A2AStarletteApplication(
            agent_card=agent_card, http_handler=request_handler
        )

        logger.info(f"正在启动服务器：{host}:{port}")
        return server.build()
    except Exception as e:
        logger.error(f"服务器启动过程中发生错误：{e}")
        exit(1)


def run_server(host: Optional[str] = "localhost", port: Optional[int] = 10000):
    try:
        import uvicorn

        app = asyncio.run(main(host, port))
        config = uvicorn.Config(
            app=app, host=host, port=port, loop="asyncio", proxy_headers=True
        )
        uvicorn.Server(config=config).run()
        logger.info(f"服务器已启动：{host}:{port}")
    except Exception as e:
        logger.error(f"启动服务器时发生错误：{e}")


if __name__ == "__main__":
    # 解析 host 与 port 命令行参数，均带有默认值
    parser = argparse.ArgumentParser(description="启动 Manus 智能体服务")
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="服务器主机地址，默认为 localhost",
    )
    parser.add_argument(
        "--port", type=int, default=10000, help="服务器端口，默认为 10000"
    )
    args = parser.parse_args()
    # 使用指定或默认的 host 与 port 启动服务器
    run_server(args.host, args.port)
