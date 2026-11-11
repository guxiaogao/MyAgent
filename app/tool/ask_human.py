from app.tool import BaseTool


class AskHuman(BaseTool):
    """添加一个用于向人类求助的工具。"""

    name: str = "ask_human"
    description: str = "使用此工具向人类请求帮助。"
    parameters: str = {
        "type": "object",
        "properties": {
            "inquire": {
                "type": "string",
                "description": "你想询问人类的问题。",
            }
        },
        "required": ["inquire"],
    }

    async def execute(self, inquire: str) -> str:
        # 在终端打印问题并等待人类输入回答
        return input(f"""机器人：{inquire}\n\n你：""").strip()
