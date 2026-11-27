import re

from app.exceptions import ToolError
from app.tool.base import BaseTool, ToolResult


class RegexExtraction(BaseTool):
    """从工具输出、网页、日志等长文本里用正则表达式精确提取信息的工具。"""

    name: str = "regex_extraction"
    description: str = (
        "从工具输出、网页、日志等长文本中用正则表达式提取邮箱/URL/IP/日期等信息。"
        "pattern 必须使用 Python re 模块的正则语法（如 r'\\d+' 提取数字）。"
        "三种操作的区别：findall 搜索整个文本并返回所有匹配项，适合不确定位置、"
        "需要提取全部内容的场景；match 只从文本开头尝试匹配，开头不匹配即失败，"
        "适合校验文本是否以某模式开头；replace 把所有匹配替换为 replacement 文本。"
    )
    parameters: dict = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "正则表达式，使用 Python re 模块语法。",
            },
            "text": {
                "type": "string",
                "description": "要处理的原始文本。",
            },
            "operation": {
                "type": "string",
                "description": "提取操作类型。",
                "enum": ["findall", "match", "replace"],
            },
            "replacement": {
                "type": "string",
                "description": "替换文本，仅 operation 为 replace 时生效，默认替换为空字符串。",
            },
        },
        "required": ["pattern", "text"],
    }

    async def execute(
        self,
        pattern: str,
        text: str,
        operation: str = "findall",
        replacement: str = "",
    ) -> ToolResult:
        """
        执行正则提取操作。

        Args:
            pattern (str): 正则表达式，Python re 模块语法。
            text (str): 要处理的原始文本。
            operation (str): 操作类型，findall / match / replace。
            replacement (str): 替换文本，仅 replace 时生效。

        Returns:
            ToolResult: 匹配或替换结果；正则非法、操作未知时抛出 ToolError。
        """
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            # 选 raise ToolError 而非 fail_response：与项目其他工具（bash/planning）风格一致，
            # 且 ToolCollection 会统一捕获转成失败结果，LLM 能看到错误信息并修正 pattern 重试
            raise ToolError(f"非法正则表达式: {e}") from None

        if operation == "findall":
            result = compiled.findall(text)
            return self.success_response(
                f"找到 {len(result)} 个匹配: {result}" if result else "未找到匹配"
            )
        elif operation == "match":
            m = compiled.match(text)
            if m is None:
                # 开头未匹配属于正常业务结果（不是错误），与"正则非法"区分开
                return self.success_response("文本开头未匹配到该模式")
            return self.success_response(m.group(0))
        elif operation == "replace":
            return self.success_response(compiled.sub(replacement, text))
        else:
            raise ToolError(f"未知操作: {operation}")


if __name__ == "__main__":
    import asyncio

    async def main():
        tool = RegexExtraction()
        # 成功用例：提取文本中的数字
        result = await tool.execute(pattern=r"\d+", text="订单号 12345，共 3 件商品")
        print(f"成功用例: {result}")
        # 失败用例：坏正则；直接调用时 ToolError 不会被 ToolCollection 转换，需手动捕获
        try:
            await tool.execute(pattern=r"(", text="test")
        except ToolError as e:
            print(f"失败用例: {e}")

    asyncio.run(main())
