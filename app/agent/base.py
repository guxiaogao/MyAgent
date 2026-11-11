from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.llm import LLM
from app.logger import logger
from app.sandbox.client import SANDBOX_CLIENT
from app.schema import ROLE_TYPE, AgentState, Memory, Message


class BaseAgent(BaseModel, ABC):
    """管理智能体状态与执行的抽象基类。

    提供状态转换、记忆管理以及基于步骤的执行循环等基础能力。
    子类必须实现 `step` 方法。
    """

    # 核心属性
    name: str = Field(..., description="智能体的唯一名称")
    description: Optional[str] = Field(None, description="可选的智能体描述")

    # 提示词
    system_prompt: Optional[str] = Field(
        None, description="系统级指令提示词"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="用于决定下一步动作的提示词"
    )

    # 依赖组件
    llm: LLM = Field(default_factory=LLM, description="语言模型实例")
    memory: Memory = Field(default_factory=Memory, description="智能体的记忆存储")
    state: AgentState = Field(
        default=AgentState.IDLE, description="智能体当前状态"
    )

    # 执行控制
    max_steps: int = Field(default=10, description="终止前允许的最大步数")
    current_step: int = Field(default=0, description="当前执行到的步数")

    duplicate_threshold: int = 2

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # 允许额外字段，便于子类灵活扩展

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """若未提供 LLM 或记忆，则使用默认设置初始化智能体。"""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """用于安全进行智能体状态转换的上下文管理器。

        进入时切换到目标状态，退出时无论成功与否都恢复原状态；
        若代码块抛出异常，则先将状态置为 ERROR 再向上抛出。

        Args:
            new_state: 上下文期间要切换到的状态。

        Yields:
            None: 允许在新状态下执行代码块。

        Raises:
            ValueError: 当 new_state 不是合法的 AgentState 时抛出。
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")

        # 保存进入前的状态，确保 finally 中能恢复
        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # 执行出错时切换到 ERROR 状态
            raise e
        finally:
            self.state = previous_state  # 恢复到进入上下文前的状态

    def update_memory(
        self,
        role: ROLE_TYPE,  # type: ignore
        content: str,
        base64_image: Optional[str] = None,
        **kwargs,
    ) -> None:
        """向智能体记忆中添加一条消息。

        Args:
            role: 消息发送者的角色（user、system、assistant、tool）。
            content: 消息内容。
            base64_image: 可选的 base64 编码图片。
            **kwargs: 额外参数（如 tool 消息的 tool_call_id）。

        Raises:
            ValueError: 当角色不受支持时抛出。
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")

        # 按角色构造消息：仅 tool 角色需要透传额外参数（如 tool_call_id）
        kwargs = {"base64_image": base64_image, **(kwargs if role == "tool" else {})}
        self.memory.add_message(message_map[role](content, **kwargs))

    async def run(self, request: Optional[str] = None) -> str:
        """异步执行智能体的主循环。

        Args:
            request: 可选的初始用户请求。

        Returns:
            汇总各步骤执行结果的字符串。

        Raises:
            RuntimeError: 若智能体起始状态不是 IDLE 则抛出。
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.update_memory("user", request)

        results: List[str] = []
        async with self.state_context(AgentState.RUNNING):
            # 主执行循环：未达最大步数且状态不是 FINISHED 时持续执行
            while (
                self.current_step < self.max_steps and self.state != AgentState.FINISHED
            ):
                self.current_step += 1
                logger.info(f"正在执行第 {self.current_step}/{self.max_steps} 步")
                step_result = await self.step()

                # 检查是否陷入循环卡住状态
                if self.is_stuck():
                    self.handle_stuck_state()

                results.append(f"第 {self.current_step} 步：{step_result}")

            if self.current_step >= self.max_steps:
                self.current_step = 0
                self.state = AgentState.IDLE
                results.append(f"已终止：达到最大步数（{self.max_steps}）")
        await SANDBOX_CLIENT.cleanup()
        return "\n".join(results) if results else "未执行任何步骤"

    @abstractmethod
    async def step(self) -> str:
        """执行智能体工作流中的单个步骤。

        必须由子类实现，以定义具体行为。
        """

    def handle_stuck_state(self):
        """处理卡住状态：在下一步提示词前追加要求更换策略的指令"""
        stuck_prompt = "\
        检测到重复的响应。请考虑新的策略，避免重复尝试已经证明无效的路径。"
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"检测到智能体陷入卡住状态，已添加提示词：{stuck_prompt}")

    def is_stuck(self) -> bool:               # 读者认为当前判断逻辑不好，完全重复消息几乎很难出现，几乎不会触发True。
        """通过检测重复内容判断智能体是否陷入循环卡住"""
        if len(self.memory.messages) < 2:
            return False

        last_message = self.memory.messages[-1]
        if not last_message.content:
            return False

        # 统计与最后一条内容完全相同的 assistant 消息出现次数
        duplicate_count = sum(
            1
            for msg in reversed(self.memory.messages[:-1])
            if msg.role == "assistant" and msg.content == last_message.content
        )

        return duplicate_count >= self.duplicate_threshold

    @property
    def messages(self) -> List[Message]:
        """从智能体记忆中获取消息列表。"""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """设置智能体记忆中的消息列表。"""
        self.memory.messages = value
