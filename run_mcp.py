#!/usr/bin/env python  # shebang 行：类 Unix 系统下允许直接以 ./run_mcp.py 方式执行本脚本
import argparse  # 导入命令行参数解析模块，用于解析连接方式、服务器地址等参数
import asyncio  # 导入异步IO框架，MCP 连接与智能体运行都是异步操作
import sys  # 导入系统模块，用于获取当前 Python 解释器路径与非零退出码

from app.agent.mcp import MCPAgent  # 导入 MCP 智能体：负责连接 MCP 服务器并调用其暴露的工具
from app.config import config  # 导入全局配置对象，读取项目根路径与 MCP 服务器引用名
from app.logger import logger  # 导入全局日志器，输出连接、运行与错误信息


class MCPRunner:  # MCP 运行器类：封装智能体的初始化、三种运行模式与资源清理
    """具备路径处理与正确配置的 MCP 智能体运行器。"""  # 类文档：带路径处理与配置的 MCP 智能体运行器

    def __init__(self):  # 构造函数：初始化运行器的基础属性
        self.root_path = config.root_path  # 记录项目根路径（stdio 模式启动子进程时用于定位模块）
        self.server_reference = config.mcp_config.server_reference  # 读取配置中的 MCP 服务器模块引用名（如 app.mcp.server）
        self.agent = MCPAgent()  # 实例化 MCP 智能体，后续所有对远程服务的调用都由它完成

    async def initialize(  # 定义异步初始化方法：按指定方式建立与 MCP 服务器的连接
        self,
        connection_type: str,  # 连接类型：stdio（本地子进程标准输入输出）或 sse（HTTP 长连接）
        server_url: str | None = None,  # SSE 模式下的服务器地址；stdio 模式下不使用，默认 None
    ) -> None:
        """使用合适的连接方式初始化 MCP 智能体。"""  # 方法文档：用合适的连接方式初始化 MCP 智能体
        logger.info(f"正在以 {connection_type} 方式初始化 MCPAgent...")  # 记录正在以哪种方式初始化连接

        if connection_type == "stdio":  # 分支一：stdio 模式，把服务器作为本地子进程拉起
            await self.agent.initialize(  # 调用智能体初始化，建立 MCP 会话
                connection_type="stdio",  # 指定连接类型为标准输入输出管道
                command=sys.executable,  # 使用当前 Python 解释器作为子进程启动命令
                args=["-m", self.server_reference],  # 以"python -m 模块名"方式运行配置指定的 MCP 服务器
            )
        else:  # sse  # 分支二：SSE 模式，连接一个已经在运行的远程 MCP 服务
            await self.agent.initialize(connection_type="sse", server_url=server_url)  # 通过 URL 与远程服务建立 SSE 连接

        logger.info(f"已通过 {connection_type} 连接到 MCP 服务器")  # 记录连接成功日志

    async def run_interactive(self) -> None:  # 交互模式：多轮对话循环，直到用户主动退出
        """以交互模式运行智能体。"""  # 方法文档：以交互模式运行智能体
        print("\nMCP 智能体交互模式（输入 'exit' 退出）\n")  # 打印模式说明与退出方式提示
        while True:  # 无限循环，维持多轮会话
            user_input = input("\n请输入你的请求：")  # 阻塞读取用户本轮请求
            if user_input.lower() in ["exit", "quit", "q"]:  # 判断是否输入了退出指令（不区分大小写）
                break  # 跳出循环，结束会话
            response = await self.agent.run(user_input)  # 把请求交给智能体：其通过 MCP 协议调用远端工具后汇总响应
            print(f"\n智能体：{response}")  # 打印智能体本轮响应

    async def run_single_prompt(self, prompt: str) -> None:  # 单次模式：执行一条提示词后即退出
        """用单条提示词运行智能体。"""  # 方法文档：用单条提示词运行智能体
        await self.agent.run(prompt)  # 执行一次任务，完成后返回

    async def run_default(self) -> None:  # 默认模式：单轮输入并执行后结束
        """以默认模式运行智能体。"""  # 方法文档：以默认模式运行智能体
        prompt = input("请输入你的提示词：")  # 读取用户提示词
        if not prompt.strip():  # 校验输入去掉空白后是否为空
            logger.warning("提供的提示词为空。")  # 记录空输入告警日志
            return  # 直接结束，不执行任务

        logger.warning("正在处理你的请求...")  # 输出"开始处理请求"提示日志
        await self.agent.run(prompt)  # 执行任务
        logger.info("请求处理完成。")  # 记录完成日志

    async def cleanup(self) -> None:  # 资源清理方法：断开连接并释放会话
        """清理智能体资源。"""  # 方法文档：清理智能体资源
        await self.agent.cleanup()  # 关闭 MCP 连接、释放底层会话资源
        logger.info("会话已结束")  # 记录会话结束日志


def parse_args() -> argparse.Namespace:  # 定义参数解析函数，返回解析后的参数对象
    """解析命令行参数。"""  # 方法文档：解析命令行参数
    parser = argparse.ArgumentParser(description="运行 MCP 智能体")  # 创建参数解析器，说明用途
    parser.add_argument(  # 注册连接类型参数
        "--connection",  # 参数长名称 --connection
        "-c",  # 参数短名称 -c
        choices=["stdio", "sse"],  # 可选值限定为 stdio 或 sse 两种
        default="stdio",  # 默认使用 stdio 连接方式
        help="连接类型：stdio 或 sse",  # 帮助说明：连接类型
    )
    parser.add_argument(  # 注册服务器地址参数
        "--server-url",  # 参数长名称 --server-url
        default="http://127.0.0.1:8000/sse",  # 默认指向本机 8000 端口的 SSE 端点
        help="SSE 连接地址",  # 帮助说明：SSE 连接地址
    )
    parser.add_argument(  # 注册交互模式开关
        "--interactive", "-i", action="store_true", help="以交互模式运行"  # -i 为布尔开关，传入即进入多轮交互模式
    )
    parser.add_argument("--prompt", "-p", help="执行单条提示词后退出")  # 注册单次提示词参数：执行完即退出
    return parser.parse_args()  # 解析并返回实际传入的参数


async def run_mcp() -> None:  # 定义 MCP 运行主协程：组装初始化逻辑与三种运行模式
    """MCP 运行器的主入口。"""  # 方法文档：运行器主入口
    args = parse_args()  # 解析命令行参数
    runner = MCPRunner()  # 创建运行器实例

    try:  # try 块保证任何退出路径都会走 finally 清理资源
        await runner.initialize(args.connection, args.server_url)  # 按参数初始化与 MCP 服务器的连接

        if args.prompt:  # 模式一：命令行指定了提示词
            await runner.run_single_prompt(args.prompt)  # 单次执行该提示词后结束
        elif args.interactive:  # 模式二：传入了 -i 交互开关
            await runner.run_interactive()  # 进入多轮交互循环
        else:  # 模式三：无额外参数
            await runner.run_default()  # 执行默认的单轮请求模式

    except KeyboardInterrupt:  # 捕获用户按 Ctrl+C 中断
        logger.info("程序已被用户中断")  # 记录"用户中断"日志
    except Exception as e:  # 捕获其余所有异常
        logger.error(f"运行 MCPAgent 出错：{str(e)}", exc_info=True)  # 记录错误信息并附带完整异常堆栈
        sys.exit(1)  # 以非零退出码结束进程，便于脚本/上层系统感知失败
    finally:  # 无论成功、异常还是中断都执行
        await runner.cleanup()  # 释放智能体与 MCP 连接资源


if __name__ == "__main__":  # 判断本文件是否作为主程序直接运行
    asyncio.run(run_mcp())  # 启动事件循环并运行 MCP 服务入口协程
