from app.tool.base import BaseTool


_TERMINATE_DESCRIPTION = """当请求已满足、或助手无法继续推进任务时，终止本次交互。
完成所有任务后，调用此工具结束工作。"""


class Terminate(BaseTool):
    name: str = "terminate"
    description: str = _TERMINATE_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "本次交互的结束状态。",
                "enum": ["success", "failure"],
            }
        },
        "required": ["status"],
    }

    async def execute(self, status: str) -> str:
        """结束当前一轮执行"""
        return f"任务已结束，结束状态：{status}"
