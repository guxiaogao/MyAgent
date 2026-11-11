"""沙箱系统的异常类定义。

本模块定义了沙箱系统全局使用的自定义异常，
以结构化的方式处理各类错误情况。
"""


class SandboxError(Exception):
    """沙箱相关错误的基础异常类。"""


class SandboxTimeoutError(SandboxError):
    """沙箱操作超时时抛出的异常。"""


class SandboxResourceError(SandboxError):
    """资源相关错误（如内存、CPU 超限）时抛出的异常。"""
