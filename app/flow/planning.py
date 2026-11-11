import json
import time
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import Field

from app.agent.base import BaseAgent
from app.flow.base import BaseFlow
from app.llm import LLM
from app.logger import logger
from app.schema import AgentState, Message, ToolChoice
from app.tool import PlanningTool


class PlanStepStatus(str, Enum):
    """定义计划步骤可能状态的枚举类"""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

    @classmethod
    def get_all_statuses(cls) -> list[str]:
        """返回所有可能的步骤状态值列表"""
        return [status.value for status in cls]

    @classmethod
    def get_active_statuses(cls) -> list[str]:
        """返回活跃状态（未开始或进行中）的值列表"""
        return [cls.NOT_STARTED.value, cls.IN_PROGRESS.value]

    @classmethod
    def get_status_marks(cls) -> Dict[str, str]:
        """返回状态到标记符号的映射"""
        return {
            cls.COMPLETED.value: "[✓]",
            cls.IN_PROGRESS.value: "[→]",
            cls.BLOCKED.value: "[!]",
            cls.NOT_STARTED.value: "[ ]",
        }


class PlanningFlow(BaseFlow):
    """基于规划工具管理任务规划与执行的流程。"""

    llm: LLM = Field(default_factory=lambda: LLM())
    planning_tool: PlanningTool = Field(default_factory=PlanningTool)
    executor_keys: List[str] = Field(default_factory=list)
    active_plan_id: str = Field(default_factory=lambda: f"plan_{int(time.time())}")
    current_step_index: Optional[int] = None

    def __init__(
        self, agents: Union[BaseAgent, List[BaseAgent], Dict[str, BaseAgent]], **data
    ):
        # 在调用父类初始化前先设置执行者键名（兼容旧参数名 executors）
        if "executors" in data:
            data["executor_keys"] = data.pop("executors")

        # 若提供了 plan_id 则设置计划 ID（兼容旧参数名）
        if "plan_id" in data:
            data["active_plan_id"] = data.pop("plan_id")

        # 未提供规划工具时使用默认实例初始化
        if "planning_tool" not in data:
            planning_tool = PlanningTool()
            data["planning_tool"] = planning_tool

        # 用处理后的数据调用父类初始化
        super().__init__(agents, **data)

        # 未指定执行者时，默认使用所有智能体
        if not self.executor_keys:
            self.executor_keys = list(self.agents.keys())

    def get_executor(self, step_type: Optional[str] = None) -> BaseAgent:
        """
        为当前步骤选择合适的执行者智能体。
        可扩展为按步骤类型/需求选择智能体。
        """
        # 提供了步骤类型且与某个智能体键名匹配时，直接使用该智能体
        if step_type and step_type in self.agents:
            return self.agents[step_type]

        # 否则按顺序取第一个可用的执行者
        for key in self.executor_keys:
            if key in self.agents:
                return self.agents[key]

        # 兜底返回主智能体
        return self.primary_agent

    async def execute(self, input_text: str) -> str:
        """使用智能体执行规划流程。"""
        try:
            if not self.primary_agent:
                raise ValueError("没有可用的主智能体")

            # 有输入时创建初始计划
            if input_text:
                await self._create_initial_plan(input_text)

                # 校验计划是否创建成功
                if self.active_plan_id not in self.planning_tool.plans:
                    logger.error(
                        f"计划创建失败：在规划工具中未找到计划 ID {self.active_plan_id}。"
                    )
                    return f"为以下请求创建计划失败：{input_text}"

            result = ""
            while True:
                # 获取当前要执行的步骤
                self.current_step_index, step_info = await self._get_current_step_info()

                # 没有更多步骤（计划完成）时结束循环
                if self.current_step_index is None:
                    result += await self._finalize_plan()
                    break

                # 用合适的智能体执行当前步骤
                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)
                step_result = await self._execute_step(executor, step_info)
                result += step_result + "\n"

                # 智能体主动终止时结束流程
                if hasattr(executor, "state") and executor.state == AgentState.FINISHED:
                    break

            return result
        except Exception as e:
            logger.error(f"PlanningFlow 执行出错: {str(e)}")
            return f"执行失败：{str(e)}"

    async def _create_initial_plan(self, request: str) -> None:
        """使用流程的 LLM 与规划工具根据请求创建初始计划。"""
        logger.info(f"正在创建初始计划，ID: {self.active_plan_id}")

        system_message_content = (
            "You are a planning assistant. Create a concise, actionable plan with clear steps. "
            "Focus on key milestones rather than detailed sub-steps. "
            "Optimize for clarity and efficiency."
        )
        agents_description = []
        for key in self.executor_keys:
            if key in self.agents:
                agents_description.append(
                    {
                        "name": key.upper(),
                        "description": self.agents[key].description,
                    }
                )
        if len(agents_description) > 1:
            # 多智能体时附加其描述，便于 LLM 在步骤中指定执行者
            system_message_content += (
                f"\nNow we have {agents_description} agents. "
                f"The infomation of them are below: {json.dumps(agents_description)}\n"
                "When creating steps in the planning tool, please specify the agent names using the format '[agent_name]'."
            )

        # 创建用于生成计划的系统消息
        system_message = Message.system_message(system_message_content)

        # 创建包含用户请求的用户消息
        user_message = Message.user_message(
            f"Create a reasonable plan with clear steps to accomplish the task: {request}"
        )

        # 带规划工具调用 LLM
        response = await self.llm.ask_tool(
            messages=[user_message],
            system_msgs=[system_message],
            tools=[self.planning_tool.to_param()],
            tool_choice=ToolChoice.AUTO,
        )

        # 处理返回的工具调用（如有）
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name == "planning":
                    # 解析工具参数
                    args = tool_call.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            logger.error(f"解析工具参数失败: {args}")
                            continue

                    # 确保使用流程统一的 plan_id，并执行规划工具
                    args["plan_id"] = self.active_plan_id

                    # 通过工具集合执行该工具，而非直接调用
                    result = await self.planning_tool.execute(**args)

                    logger.info(f"计划创建结果: {str(result)}")
                    return

        # 走到这里说明 LLM 未成功调用规划工具，创建默认计划
        logger.warning("正在创建默认计划")

        # 使用规划工具创建默认计划
        await self.planning_tool.execute(
            **{
                "command": "create",
                "plan_id": self.active_plan_id,
                "title": f"Plan for: {request[:50]}{'...' if len(request) > 50 else ''}",
                "steps": ["Analyze request", "Execute task", "Verify results"],
            }
        )

    async def _get_current_step_info(self) -> tuple[Optional[int], Optional[dict]]:
        """
        解析当前计划，定位第一个未完成步骤的索引与信息。
        找不到活跃步骤时返回 (None, None)。
        """
        if (
            not self.active_plan_id
            or self.active_plan_id not in self.planning_tool.plans
        ):
            logger.error(f"未找到 ID 为 {self.active_plan_id} 的计划")
            return None, None

        try:
            # 直接从规划工具存储中访问计划数据
            plan_data = self.planning_tool.plans[self.active_plan_id]
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])

            # 查找第一个未完成的步骤
            for i, step in enumerate(steps):
                if i >= len(step_statuses):
                    status = PlanStepStatus.NOT_STARTED.value
                else:
                    status = step_statuses[i]

                if status in PlanStepStatus.get_active_statuses():
                    # 提取步骤的类型/类别信息（如有）
                    step_info = {"text": step}

                    # 尝试从步骤文本中提取类型标记（如 [SEARCH] 或 [CODE]）
                    import re

                    type_match = re.search(r"\[([A-Z_]+)\]", step)
                    if type_match:
                        step_info["type"] = type_match.group(1).lower()

                    # 将当前步骤标记为进行中
                    try:
                        await self.planning_tool.execute(
                            command="mark_step",
                            plan_id=self.active_plan_id,
                            step_index=i,
                            step_status=PlanStepStatus.IN_PROGRESS.value,
                        )
                    except Exception as e:
                        logger.warning(f"标记步骤为进行中时出错: {e}")
                        # 标记失败时直接更新步骤状态兜底
                        if i < len(step_statuses):
                            step_statuses[i] = PlanStepStatus.IN_PROGRESS.value
                        else:
                            while len(step_statuses) < i:
                                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                            step_statuses.append(PlanStepStatus.IN_PROGRESS.value)

                        plan_data["step_statuses"] = step_statuses

                    return i, step_info

            return None, None  # 未找到活跃步骤

        except Exception as e:
            logger.warning(f"查找当前步骤索引时出错: {e}")
            return None, None

    async def _execute_step(self, executor: BaseAgent, step_info: dict) -> str:
        """使用指定智能体的 agent.run() 执行当前步骤。"""
        # 准备包含当前计划状态的智能体上下文
        plan_status = await self._get_plan_text()
        step_text = step_info.get("text", f"Step {self.current_step_index}")

        # 构建执行当前步骤的提示词
        step_prompt = f"""
        CURRENT PLAN STATUS:
        {plan_status}

        YOUR CURRENT TASK:
        You are now working on step {self.current_step_index}: "{step_text}"

        Please only execute this current step using the appropriate tools. When you're done, provide a summary of what you accomplished.
        """

        # 通过 agent.run() 执行步骤
        try:
            step_result = await executor.run(step_prompt)

            # 执行成功后将该步骤标记为已完成
            await self._mark_step_completed()

            return step_result
        except Exception as e:
            logger.error(f"执行步骤 {self.current_step_index} 时出错: {e}")
            return f"执行步骤 {self.current_step_index} 时出错：{str(e)}"

    async def _mark_step_completed(self) -> None:
        """将当前步骤标记为已完成。"""
        if self.current_step_index is None:
            return

        try:
            # 通过规划工具将步骤标记为完成
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_status=PlanStepStatus.COMPLETED.value,
            )
            logger.info(
                f"已将计划 {self.active_plan_id} 中的步骤 {self.current_step_index} 标记为完成"
            )
        except Exception as e:
            logger.warning(f"更新计划状态失败: {e}")
            # 标记失败时直接在规划工具存储中更新状态
            if self.active_plan_id in self.planning_tool.plans:
                plan_data = self.planning_tool.plans[self.active_plan_id]
                step_statuses = plan_data.get("step_statuses", [])

                # 确保 step_statuses 列表足够长
                while len(step_statuses) <= self.current_step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)

                # 更新状态
                step_statuses[self.current_step_index] = PlanStepStatus.COMPLETED.value
                plan_data["step_statuses"] = step_statuses

    async def _get_plan_text(self) -> str:
        """获取当前计划的格式化文本。"""
        try:
            result = await self.planning_tool.execute(
                command="get", plan_id=self.active_plan_id
            )
            return result.output if hasattr(result, "output") else str(result)
        except Exception as e:
            logger.error(f"获取计划时出错: {e}")
            return self._generate_plan_text_from_storage()

    def _generate_plan_text_from_storage(self) -> str:
        """规划工具失败时，直接从存储生成计划文本。"""
        try:
            if self.active_plan_id not in self.planning_tool.plans:
                return f"错误：未找到 ID 为 {self.active_plan_id} 的计划"

            plan_data = self.planning_tool.plans[self.active_plan_id]
            title = plan_data.get("title", "未命名计划")
            steps = plan_data.get("steps", [])
            step_statuses = plan_data.get("step_statuses", [])
            step_notes = plan_data.get("step_notes", [])

            # 确保 step_statuses 与 step_notes 的长度和步骤数一致
            while len(step_statuses) < len(steps):
                step_statuses.append(PlanStepStatus.NOT_STARTED.value)
            while len(step_notes) < len(steps):
                step_notes.append("")

            # 按状态统计步骤数
            status_counts = {status: 0 for status in PlanStepStatus.get_all_statuses()}

            for status in step_statuses:
                if status in status_counts:
                    status_counts[status] += 1

            completed = status_counts[PlanStepStatus.COMPLETED.value]
            total = len(steps)
            progress = (completed / total) * 100 if total > 0 else 0

            plan_text = f"计划：{title}（ID: {self.active_plan_id}）\n"
            plan_text += "=" * len(plan_text) + "\n\n"

            plan_text += (
                f"进度：已完成 {completed}/{total} 步（{progress:.1f}%）\n"
            )
            plan_text += f"状态：{status_counts[PlanStepStatus.COMPLETED.value]} 个已完成，{status_counts[PlanStepStatus.IN_PROGRESS.value]} 个进行中，"
            plan_text += f"{status_counts[PlanStepStatus.BLOCKED.value]} 个被阻塞，{status_counts[PlanStepStatus.NOT_STARTED.value]} 个未开始\n\n"
            plan_text += "步骤：\n"

            status_marks = PlanStepStatus.get_status_marks()

            for i, (step, status, notes) in enumerate(
                zip(steps, step_statuses, step_notes)
            ):
                # 用状态标记标识步骤状态
                status_mark = status_marks.get(
                    status, status_marks[PlanStepStatus.NOT_STARTED.value]
                )

                plan_text += f"{i}. {status_mark} {step}\n"
                if notes:
                    plan_text += f"   备注：{notes}\n"

            return plan_text
        except Exception as e:
            logger.error(f"从存储生成计划文本时出错: {e}")
            return f"错误：无法获取 ID 为 {self.active_plan_id} 的计划"

    async def _finalize_plan(self) -> str:
        """完成计划并直接使用流程的 LLM 生成总结。"""
        plan_text = await self._get_plan_text()

        # 直接使用流程的 LLM 生成总结
        try:
            system_message = Message.system_message(
                "你是一个规划助手，任务是总结已完成的计划。"
            )

            user_message = Message.user_message(
                f"计划已全部完成。以下是最终计划状态：\n\n{plan_text}\n\n请总结已完成的成果以及最终的想法。"
            )

            response = await self.llm.ask(
                messages=[user_message], system_msgs=[system_message]
            )

            return f"计划已完成：\n\n{response}"
        except Exception as e:
            logger.error(f"使用 LLM 汇总计划时出错: {e}")

            # LLM 失败时回退为用智能体生成总结
            try:
                agent = self.primary_agent
                summary_prompt = f"""
                The plan has been completed. Here is the final plan status:

                {plan_text}

                Please provide a summary of what was accomplished and any final thoughts.
                """
                summary = await agent.run(summary_prompt)
                return f"Plan completed:\n\n{summary}"
            except Exception as e2:
                logger.error(f"使用智能体汇总计划时出错: {e2}")
                return "Plan completed. Error generating summary."
