from abc import ABC, abstractmethod
from typing import Optional

from pydantic import Field

from app.agent.base import BaseAgent
from app.llm import LLM
from app.schema import AgentState, Memory


class ReActAgent(BaseAgent, ABC):
    """实现 ReAct（推理-行动）范式的抽象智能体。"""
    name: str
    description: Optional[str] = None

    system_prompt: Optional[str] = None
    next_step_prompt: Optional[str] = None

    llm: Optional[LLM] = Field(default_factory=LLM)
    memory: Memory = Field(default_factory=Memory)
    state: AgentState = AgentState.IDLE

    max_steps: int = 10
    current_step: int = 0

    @abstractmethod
    async def think(self) -> bool:
        """根据当前状态决策下一步动作"""

    @abstractmethod
    async def act(self) -> str:
        """执行已决定的动作"""

    async def step(self) -> str:
        """执行单个步骤：先思考后行动。"""
        # ReAct 模式：先由 LLM 思考决策（think），根据返回结果决定是否执行动作（act）
        should_act = await self.think()
        if not should_act:
            return "思考完成，无需行动"
        return await self.act()
