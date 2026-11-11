from app.tool.base import BaseTool, ToolResult

class RegexTool(BaseTool):

    name: str = "regex_tool"

    description: str = "..."        # 难点①：写清楚什么时候用/怎么写 pattern

    parameters: dict = { ... }      # 难点②：pattern/text 必填，operation 用 enum

    async def execute(self, pattern: str, text: str,

    operation: str = "findall") -> ToolResult: ...