import argparse  # 导入命令行参数解析模块，用于解析 --prompt 等启动参数
import asyncio  # 导入异步IO框架，用于运行协程式的智能体主循环

from app.agent.manus import Manus  # 导入核心智能体 Manus，它是单智能体模式的执行主体
from app.logger import logger  # 导入全局日志器，用于输出运行状态与告警信息


async def main():  # 定义异步主函数，作为整个程序的协程入口
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="带提示词运行 Manus 智能体")  # 创建参数解析器，说明用途为"带提示词运行 Manus 智能体"
    parser.add_argument(  # 注册一个命令行参数
        "--prompt", type=str, required=False, help="智能体的输入提示词"  # --prompt：字符串类型、非必填，作为智能体的任务输入
    )
    args = parser.parse_args()  # 解析实际传入的命令行参数，得到 args 对象

    # 创建并初始化 Manus 智能体
    agent = await Manus.create()  # 通过异步工厂方法创建并初始化智能体（内部加载配置、绑定可用工具）
    try:  # try 块保证无论成功、异常还是中断，最后都能走到 finally 清理资源
        # 若命令行提供了提示词则直接使用，否则交互式询问输入
        prompt = args.prompt if args.prompt else input("请输入你的提示词：")  # 获取提示词：命令行给过就直接用，否则在终端交互式输入
        if not prompt.strip():  # 判断提示词去掉首尾空白后是否为空
            logger.warning("提供的提示词为空。")  # 记录"空提示词"告警日志
            return  # 直接结束，不执行任何任务

        logger.warning("正在处理你的请求...")  # 输出"开始处理请求"的提示日志
        await agent.run(prompt)  # 核心一行：驱动智能体进入"思考→选工具→执行→观察"的循环，直到任务完成
        logger.info("请求处理完成。")  # 记录任务完成日志
    except KeyboardInterrupt:  # 捕获用户按 Ctrl+C 主动中断
        logger.warning("操作已被中断。")  # 记录中断告警日志
    finally:  # 无论上面如何退出都会执行
        # 确保退出前清理智能体资源
        await agent.cleanup()  # 异步释放智能体占用的资源（浏览器会话、MCP 连接等），防止泄漏


if __name__ == "__main__":  # 判断本文件是否作为主程序直接运行（而不是被其他模块导入）
    asyncio.run(main())  # 启动事件循环并运行 main 协程，程序正式开始
