import asyncio  # 导入异步IO框架，用于运行协程与对流程执行施加超时控制
import time  # 导入时间模块，用于统计流程执行总耗时

from app.agent.data_analysis import DataAnalysis  # 导入数据分析智能体，按配置可选加入流程
from app.agent.manus import Manus  # 导入核心通用智能体 Manus，作为流程的主力执行者
from app.config import config  # 导入全局配置对象，读取运行流程（run_flow）相关配置项
from app.flow.flow_factory import FlowFactory, FlowType  # 导入流程工厂与流程类型枚举，用于创建流程调度器
from app.logger import logger  # 导入全局日志器，输出流程运行状态


async def run_flow():  # 定义异步流程主函数：把多个智能体交给流程引擎统一调度
    agents = {  # 构建智能体字典：key 为名称，value 为实例，交给流程引擎在步骤中选用
        "manus": Manus(),  # 注册主智能体 Manus（每个智能体独立持有记忆与工具）
    }
    if config.run_flow_config.use_data_analysis_agent:  # 读取配置，判断是否启用数据分析智能体（对应并行/分工场景）
        agents["data_analysis"] = DataAnalysis()  # 追加注册数据分析智能体到字典
    try:  # 外层 try 捕获交互与流程整体的异常
        prompt = input("请输入你的提示词：")  # 终端交互，获取用户的任务描述

        if prompt.strip().isspace() or not prompt:  # 校验输入：纯空白字符或空串均视为无效
            logger.warning("提供的提示词为空。")  # 记录空输入告警日志
            return  # 直接结束，不创建流程

        flow = FlowFactory.create_flow(  # 核心一行：通过工厂创建流程实例（串行/并行/条件分支由流程类型决定）
            flow_type=FlowType.PLANNING,  # 指定为规划（PLANNING）流程：先由 LLM 规划步骤，再逐步分派执行
            agents=agents,  # 把智能体字典交给流程引擎，规划出的每个步骤可由合适的智能体执行
        )
        logger.warning("正在处理你的请求...")  # 输出"开始处理请求"的提示日志

        try:  # 内层 try 专门捕获执行超时异常
            start_time = time.time()  # 记录流程开始时间戳
            result = await asyncio.wait_for(  # 异步等待流程执行完成，并施加整体超时限制
                flow.execute(prompt),  # 流程引擎驱动各智能体按规划好的步骤推进任务
                timeout=3600,  # 整体执行超时 60 分钟  # 整体超时 3600 秒（60 分钟）
            )
            elapsed_time = time.time() - start_time  # 计算流程总耗时（秒）
            logger.info(f"请求处理完成，耗时 {elapsed_time:.2f} 秒")  # 记录耗时日志
            logger.info(result)  # 输出流程的最终执行结果
        except asyncio.TimeoutError:  # 捕获超时异常
            logger.error("请求处理在 1 小时后超时")  # 记录"超时1小时"错误日志
            logger.info(  # 给用户的超时后续提示
                "操作因超时被终止。请尝试更简单的请求。"  # 提示任务被终止，建议换更简单的请求
            )

    except KeyboardInterrupt:  # 捕获用户按 Ctrl+C 主动中断
        logger.info("操作已被用户取消。")  # 记录"用户取消"日志
    except Exception as e:  # 捕获其余所有未预料异常
        logger.error(f"错误：{str(e)}")  # 记录错误信息日志


if __name__ == "__main__":  # 判断本文件是否作为主程序直接运行
    asyncio.run(run_flow())  # 启动事件循环并运行流程编排主协程
