import json
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from app.utils.logger import logger


# class BaseTool(ABC, BaseModel):
#     name: str
#     description: str
#     parameters: Optional[dict] = None

#     class Config:
#         arbitrary_types_allowed = True

#     async def __call__(self, **kwargs) -> Any:
#         """以给定参数执行工具。"""
#         return await self.execute(**kwargs)

#     @abstractmethod
#     async def execute(self, **kwargs) -> Any:
#         """以给定参数执行工具。"""

#     def to_param(self) -> Dict:
#         """将工具转换为函数调用格式。"""
#         return {
#             "type": "function",
#             "function": {
#                 "name": self.name,
#                 "description": self.description,
#                 "parameters": self.parameters,
#             },
#         }


class ToolResult(BaseModel):
    """表示一次工具执行的结果。

    字段说明：
        output: 工具产生的正常输出内容。
        error: 错误信息（执行失败时存在）。
        base64_image: Base64 编码的图片数据（可选）。
        system: 附加的系统级信息（可选）。
    """

    output: Any = Field(default=None)
    error: Optional[str] = Field(default=None)
    base64_image: Optional[str] = Field(default=None)
    system: Optional[str] = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def __bool__(self):
        # 真值判断：任一字段有值即视为非空结果
        # 用 self.dict() 的键获取字段名，替代 Pydantic 动态注入的 __fields__（静态检查器无法识别）
        return any(getattr(self, field) for field in self.__fields__)

    def __add__(self, other: "ToolResult"):
        """合并两个工具结果，支持用 `result1 + result2` 拼接同类型结果。"""
        # 合并两个结果的同名字段：字符串字段拼接；base64_image 只允许保留一个值，
        # 因此以 concatenate=False 调用，两侧同时有值时直接报错

        def combine_fields(
            field: Optional[str], other_field: Optional[str], concatenate: bool = True
        ):
            # 双方均有值：可拼接字段（output/error/system）直接拼接；
            # 不可重复字段（base64_image）同时有值则视为冲突，抛出异常
            if field and other_field:
                if concatenate:
                    return field + other_field
                raise ValueError("无法合并工具结果")
            # 仅单侧有值（或均为空）：直接取有值的一侧（field or other_field）
            return field or other_field

        return ToolResult(
            output=combine_fields(self.output, other.output),
            error=combine_fields(self.error, other.error),
            base64_image=combine_fields(self.base64_image, other.base64_image, False),
            system=combine_fields(self.system, other.system),
        )

    def __str__(self):
        # 打印时优先展示错误信息，便于调试定位失败原因
        return f"错误：{self.error}" if self.error else self.output

    def replace(self, **kwargs):
        """返回一个用给定字段值替换后的新 ToolResult。"""
        # return self.copy(update=kwargs)
        # 基于当前实例的全部字段构造新对象，kwargs 中的字段覆盖原值；
        # 使用 type(self) 保证子类（如 CLIResult/ToolFailure）类型不丢失
        return type(self)(**{**self.dict(), **kwargs})


class BaseTool(ABC, BaseModel):
    """所有工具的统一基类，结合了 Pydantic BaseModel 与工具接口的能力。

    提供：
    - Pydantic 模型校验
    - Schema 注册（注册逻辑当前已被注释停用）
    - 标准化的结果处理（ToolResult）
    - 抽象执行接口

    属性：
        name (str): 工具名称（LLM 调用时使用的标识）
        description (str): 工具描述，供 LLM 理解工具用途
        parameters (dict): 工具参数的 JSON Schema
        _schemas (Dict[str, List[ToolSchema]]): 已注册的方法 Schema
    """

    name: str
    description: str
    parameters: Optional[dict] = None
    # _schemas: Dict[str, List[ToolSchema]] = {}

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # def __init__(self, **data):
    #     """初始化工具，并完成模型校验与 Schema 注册。"""
    #     super().__init__(**data)
    #     logger.debug(f"Initializing tool class: {self.__class__.__name__}")
    #     self._register_schemas()

    # def _register_schemas(self):
    #     """注册所有被装饰器标记的方法的 Schema。"""
    #     for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
    #         if hasattr(method, 'tool_schemas'):
    #             self._schemas[name] = method.tool_schemas
    #             logger.debug(f"Registered schemas for method '{name}' in {self.__class__.__name__}")

    async def __call__(self, **kwargs) -> Any:
        """以给定参数执行工具。"""
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """以给定参数执行工具。"""

    def to_param(self) -> Dict:
        """将工具转换为函数调用格式。

        Returns:
            dict: 符合 OpenAI function calling 格式的工具元数据（名称、描述与参数 Schema）
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    # def get_schemas(self) -> Dict[str, List[ToolSchema]]:
    #     """获取所有已注册的工具 Schema。

    #     Returns:
    #         dict: 方法名到其 Schema 定义的映射
    #     """
    #     return self._schemas

    def success_response(self, data: Union[Dict[str, Any], str]) -> ToolResult:
        """构造执行成功的工具结果。

        Args:
            data: 结果数据（字符串则直接作为文本；字典则序列化为格式化的 JSON）

        Returns:
            ToolResult: 携带格式化后的输出文本
        """
        if isinstance(data, str):
            text = data
        else:
            text = json.dumps(data, indent=2)
        logger.debug(f"为 {self.__class__.__name__} 创建了成功响应")
        return ToolResult(output=text)

    def fail_response(self, msg: str) -> ToolResult:
        """构造执行失败的工具结果。

        Args:
            msg: 描述失败原因的错误信息

        Returns:
            ToolResult: 携带错误信息
        """
        logger.debug(f"工具 {self.__class__.__name__} 返回了失败结果：{msg}")
        return ToolResult(error=msg)


class CLIResult(ToolResult):
    """可以按 CLI 输出形式渲染的 ToolResult。"""


class ToolFailure(ToolResult):
    """表示执行失败的 ToolResult。"""
