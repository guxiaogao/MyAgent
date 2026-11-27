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
            # 主循环：反复取出下一个活跃步骤交给智能体执行，
            # 直到计划全部完成或智能体主动终止流程
            while True:
                # 获取当前要执行的步骤（第一个未完成步骤的索引及其信息）
                self.current_step_index, step_info = await self._get_current_step_info()

                # 没有更多步骤（计划完成）时结束循环
                if self.current_step_index is None:
                    result += await self._finalize_plan()
                    break

                # 用合适的智能体执行当前步骤：
                # 若步骤文本带有 [TYPE] 标记且与某智能体键名匹配，则路由到对应智能体
                step_type = step_info.get("type") if step_info else None
                executor = self.get_executor(step_type)
                step_result = await self._execute_step(executor, step_info)
                result += step_result + "\n"

                # 智能体主动终止（如任务完成、等待用户输入）时结束整个流程，
                # 不再继续推进计划中的剩余步骤
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
            "你是一个规划助手。请创建简洁、可执行的计划，包含清晰的步骤。"
            "聚焦于关键里程碑，而非过于细化的子步骤。"
            "追求清晰与高效。"
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
                f"\n当前有 {len(agents_description)} 个智能体，信息如下：{json.dumps(agents_description)}\n"
                "在规划工具中创建步骤时，请使用 '[智能体名]' 的格式在步骤文本中指定负责执行的智能体名称。"
            )

        # 创建用于生成计划的系统消息
        system_message = Message.system_message(system_message_content)

        # 创建包含用户请求的用户消息
        user_message = Message.user_message(
            f"请制定一个包含清晰步骤的合理计划，以完成以下任务：{request}"
        )

        # 带规划工具调用 LLM 生成计划文本，规划工具负责管理LLM返回的规划
        response = await self.llm.ask_tool(
            messages=[user_message],
            system_msgs=[system_message],
            tools=[self.planning_tool.to_param()],
            tool_choice=ToolChoice.AUTO,
        )

        # 处理返回的工具调用（如有）：只关注名为 planning 的工具调用
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call.function.name == "planning":
                    # 解析工具参数：不同模型返回的 arguments 可能是
                    # JSON 字符串或已解析的字典，需统一转为字典
                    args = tool_call.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            logger.error(f"解析工具参数失败: {args}")
                            continue

                    # 强制覆盖为流程统一的 plan_id，防止 LLM 自行生成的 plan_id
                    # 与流程内部状态不一致；随后执行规划工具完成计划创建
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
                "title": f"任务计划：{request[:50]}{'...' if len(request) > 50 else ''}",
                "steps": ["分析请求", "执行任务", "验证结果"],
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

            # 顺序查找第一个活跃（未开始/进行中）的步骤
            for i, step in enumerate(steps):
                # step_statuses 可能短于 steps（历史数据不完整），越界时按未开始处理
                if i >= len(step_statuses):
                    status = PlanStepStatus.NOT_STARTED.value
                else:
                    status = step_statuses[i]

                if status in PlanStepStatus.get_active_statuses():
                    # 提取步骤的类型/类别信息（如有）
                    step_info = {"text": step}

                    # 尝试从步骤文本中提取类型标记（如 [SEARCH] 或 [CODE]），
                    # 标记转小写后用于匹配同名智能体键名，实现按步骤类型路由
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
                        # 标记失败时兜底：绕过工具，直接在存储中写入"进行中"状态；
                        # 索引已存在则覆盖，否则先补齐前面缺失的状态再追加
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

        # 构建执行当前步骤的提示词：附带完整计划状态作为上下文，
        # 并明确限定智能体只执行当前这一步，完成后输出本次完成情况的总结
        step_prompt = f"""
        当前计划状态：
        {plan_status}

        你的当前任务：
        你正在执行第 {self.current_step_index} 步："{step_text}"

        请只使用合适的工具执行当前这一步。完成后，请总结你完成了哪些工作。
        """

        # 通过 agent.run() 执行步骤（智能体内部可自主进行多轮工具调用，
        # 是否完成当前步骤由智能体自身的终止机制判断）
        try:
            step_result = await executor.run(step_prompt)

            # 执行成功后将该步骤标记为已完成，供下一轮循环推进到后续步骤
            await self._mark_step_completed()

            return step_result
        except Exception as e:
            logger.error(f"执行步骤 {self.current_step_index} 时出错: {e}")
            # 执行出错时步骤停留在活跃状态，下一轮会被再次取出重试，
            # 因此这里将其标记为阻塞以避免无限循环
            await self._mark_step_blocked(f"执行出错：{str(e)}")
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

                # 确保 step_statuses 列表足够长（不足时补齐"未开始"状态）
                while len(step_statuses) <= self.current_step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)

                # 更新状态
                step_statuses[self.current_step_index] = PlanStepStatus.COMPLETED.value
                plan_data["step_statuses"] = step_statuses

    async def _mark_step_blocked(self, note: str = "") -> None:
        """执行步骤抛出异常时，将当前步骤标记为阻塞并记录原因备注。

        阻塞状态不属于活跃状态，可避免流程下一轮重复执行同一步骤造成死循环。
        """
        if self.current_step_index is None:
            return

        try:
            # 通过规划工具标记阻塞并写入备注
            await self.planning_tool.execute(
                command="mark_step",
                plan_id=self.active_plan_id,
                step_index=self.current_step_index,
                step_status=PlanStepStatus.BLOCKED.value,
                step_notes=note,
            )
            logger.info(
                f"已将计划 {self.active_plan_id} 中的步骤 {self.current_step_index} 标记为阻塞"
            )
        except Exception as e:
            logger.warning(f"标记步骤为阻塞失败: {e}")
            # 标记失败时直接在规划工具存储中更新状态与备注
            if self.active_plan_id in self.planning_tool.plans:
                plan_data = self.planning_tool.plans[self.active_plan_id]
                step_statuses = plan_data.get("step_statuses", [])
                step_notes = plan_data.get("step_notes", [])

                # 确保 step_statuses 与 step_notes 列表足够长
                while len(step_statuses) <= self.current_step_index:
                    step_statuses.append(PlanStepStatus.NOT_STARTED.value)
                while len(step_notes) <= self.current_step_index:
                    step_notes.append("")

                # 更新状态与备注
                step_statuses[self.current_step_index] = PlanStepStatus.BLOCKED.value
                if note:
                    step_notes[self.current_step_index] = note
                plan_data["step_statuses"] = step_statuses
                plan_data["step_notes"] = step_notes

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

            # 按各状态统计步骤数，用于计算整体完成进度百分比
            status_counts = {status: 0 for status in PlanStepStatus.get_all_statuses()}

            for status in step_statuses:
                # 忽略未知的状态值，避免统计时出错
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

        # 直接使用流程的 LLM 生成总结（不占用执行智能体的上下文与可用轮次）
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
                计划已全部完成。以下是最终计划状态：

                {plan_text}

                请总结已完成的成果以及最终的想法。
                """
                summary = await agent.run(summary_prompt)
                return f"Plan completed:\n\n{summary}"
            except Exception as e2:
                logger.error(f"使用智能体汇总计划时出错: {e2}")
                return "Plan completed. Error generating summary."
