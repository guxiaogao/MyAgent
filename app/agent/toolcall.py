import asyncio
import json
from typing import Any, List, Optional, Union
from inspect import iscoroutinefunction

from pydantic import Field

from app.agent.react import ReActAgent
from app.exceptions import TokenLimitExceeded
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import TOOL_CHOICE_TYPE, AgentState, Message, ToolCall, ToolChoice
from app.tool import CreateChatCompletion, Terminate, ToolCollection


# REQUIRED 模式下模型未返回任何工具调用时的错误提示
TOOL_CALL_REQUIRED = "需要调用工具，但模型未提供任何工具调用"


class ToolCallAgent(ReActAgent):
    """处理工具/函数调用的基础智能体类，提供了更强的抽象能力"""

    name: str = "toolcall"
    description: str = "可执行工具调用的智能体"

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO           # `AUTO` 表示由模型自行决定：这轮是调用工具，还是只输出文本
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[ToolCall] = Field(default_factory=list)
    # 暂存最近一次工具调用返回的 base64 图片，随下一条消息写入记忆
    _current_base64_image: Optional[str] = None

    max_steps: int = 30
    # 工具观察结果的截断长度：int 表示截断到指定字符数，None/False 表示不截断
    max_observe: Optional[Union[int, bool]] = None

    async def think(self) -> bool:
        """处理当前状态并结合工具决策下一步动作"""
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        try:
            # 调用 LLM 并附带可用工具列表，获取带工具调用意向的响应
            response = await self.llm.ask_tool(
                messages=self.messages,
                system_msgs=(
                    [Message.system_message(self.system_prompt)]
                    if self.system_prompt
                    else None
                ),
                tools=self.available_tools.to_params(),
                tool_choice=self.tool_choices,
            )
        except ValueError:
            raise
        except Exception as e:
            # 判断异常是否为 Token 超限（TokenLimitExceeded）。
            # 兼容两种到达形态：重试后由 tenacity 包装成 RetryError 时，原始异常挂在
            # e.__cause__ 上；重试条件排除后（当前配置），异常本体直接到达。
            if isinstance(e, TokenLimitExceeded):
                token_limit_error = e
            elif hasattr(e, "__cause__") and isinstance(e.__cause__, TokenLimitExceeded):   # 这个分支用不到，因为token超限不会重试
                token_limit_error = e.__cause__
            else:
                raise
            logger.error(f"🚨 Token 超限错误：{token_limit_error}")
            self.memory.add_message(
                Message.assistant_message(
                    f"已达到最大 Token 上限，无法继续执行：{str(token_limit_error)}"
                )
            )
            self.state = AgentState.FINISHED
            return False

        self.tool_calls = tool_calls = (
            response.tool_calls if response and response.tool_calls else []
        )     # 两个等号，链式赋值，tool_calls是方法内局部变量
        content = response.content if response and response.content else ""

        # 记录模型响应信息
        logger.info(f"✨ {self.name} 的思考：{content}")
        logger.info(
            f"🛠️ {self.name} 选择了 {len(tool_calls) if tool_calls else 0} 个工具待使用"
        )
        if tool_calls:
            logger.info(
                f"🧰 准备调用的工具：{[call.function.name for call in tool_calls]}"
            )
            logger.info(f"🔧 工具参数：{[call.function.arguments for call in tool_calls]}")

        try:
            if response is None:
                raise RuntimeError("没有收到 LLM 的响应")

            # 按不同的 tool_choices 模式分别处理
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(
                        f"🤔 注意，{self.name} 在工具不可用时试图调用工具！"
                    )
                if content:
                    self.memory.add_message(Message.assistant_message(content))
                    return True
                return False

            # 构造并记录 assistant 消息（可能携带工具调用信息）
            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # 稍后在 act() 中处理

            # auto 模式下：无工具调用但有文本内容时，视为需要继续处理
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 {self.name} 的思考过程出现异常：{e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"处理过程中遇到错误：{str(e)}"
                )
            )
            return False

    async def act(self) -> str:
        """执行工具调用并处理其结果"""
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)

            # 无工具调用时直接返回最后一条消息的内容
            return self.messages[-1].content or "没有可执行的内容或命令"

        results = []
        image_messages = []
        # 顺序执行每个工具调用，收集结果
        for command in self.tool_calls:
            # 每次工具调用前重置 base64 图片缓存，避免误用上一次的图片
            self._current_base64_image = None

            result = await self.execute_tool(command)

            if self.max_observe:
                result = result[: self.max_observe]  # 工具返回结果不可控，截断结果，防止超出最大观察长度

            logger.info(
                f"🎯 工具 '{command.function.name}' 执行完成！结果：{result}"   # 日志记录的是截断后的结果
            )

            # 将工具响应写入记忆
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
            )
            self.memory.add_message(tool_msg)
            if self._current_base64_image:
                image_messages.append(
                    Message.user_message(
                        content=f"工具 {command.function.name} 返回的图片：",
                        base64_image=self._current_base64_image,
                    )
                )
            results.append(result)

        # 将工具返回的图片统一写入记忆
        self.memory.add_messages(image_messages)
        return "\n\n".join(results)

    async def execute_tool(self, command: ToolCall) -> str:
        """执行单个工具调用，具备健壮的错误处理"""
        # 校验命令结构完整性：缺少调用体、函数信息或工具名均视为无效
        if not command or not command.function or not command.function.name:
            return "错误：无效的命令格式"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"错误：未知工具 '{name}'"

        try:
            # 解析工具调用参数（JSON 格式）
            args = json.loads(command.function.arguments or "{}")

            # 执行工具
            logger.info(f"🔧 正在调用工具：'{name}'...")
            result = await self.available_tools.execute(name=name, tool_input=args)

            # 检查当前工具是否为特殊工具（如 terminate），判断是否触发状态变更
            await self._handle_special_tool(name=name, result=result)

            # 若工具结果携带 base64 图片，则缓存供后续消息使用
            if hasattr(result, "base64_image") and result.base64_image:
                # 暂存 base64 图片，稍后随消息写入记忆
                self._current_base64_image = result.base64_image

            # 将结果格式化为标准观察输出
            observation = (
                f"工具 `{name}` 的执行结果如下：\n{str(result)}"
                if result
                else f"工具 `{name}` 执行完成，无输出"
            )

            return observation
        except json.JSONDecodeError:
            error_msg = f"解析工具 {name} 的参数失败：JSON 格式无效"
            logger.error(
                f"📝 工具 '{name}' 的参数无效 - JSON 格式错误，参数为：{command.function.arguments}"
            )
            return f"错误：{error_msg}"
        except Exception as e:
            error_msg = f"⚠️ 工具 '{name}' 遇到问题：{str(e)}"
            logger.exception(error_msg)
            return f"错误：{error_msg}"

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """处理特殊工具的执行及其触发的状态变更"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # 将智能体状态置为已完成
            logger.info(f"🏁 特殊工具 '{name}' 已完成任务！")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """判断工具执行后是否应结束智能体"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """检查工具名是否在特殊工具列表中"""
        return name.lower() in [n.lower() for n in self.special_tool_names]

    async def cleanup(self):
        """清理智能体工具占用的资源。"""
        logger.info(f"🧹 正在清理智能体 '{self.name}' 的资源...")
        for tool_name, tool_instance in self.available_tools.tool_map.items():
            if hasattr(tool_instance, "cleanup") and iscoroutinefunction(
                tool_instance.cleanup
            ):
                try:
                    logger.debug(f"🧼 正在清理工具：{tool_name}")
                    await tool_instance.cleanup()
                except Exception as e:
                    logger.error(
                        f"🚨 清理工具 '{tool_name}' 时出错：{e}", exc_info=True
                    )
        logger.info(f"✨ 智能体 '{self.name}' 清理完成。")

    async def run(self, request: Optional[str] = None) -> str:
        """运行智能体，结束后执行清理。"""
        try:
            return await super().run(request)
        finally:
            await self.cleanup()
