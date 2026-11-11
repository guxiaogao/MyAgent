class ToolError(Exception):
    """工具执行出错时抛出。"""

    def __init__(self, message):
        self.message = message


class OpenManusError(Exception):
    """所有 OpenManus 异常的基类"""


class TokenLimitExceeded(OpenManusError):
    """超出 token 上限时抛出的异常"""
