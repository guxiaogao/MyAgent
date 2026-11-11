import asyncio
import os
from typing import Optional

from app.exceptions import ToolError
from app.tool.base import BaseTool, CLIResult


# 提供给 LLM 的工具使用说明：经 to_param() 进入 function calling 的工具定义，
# 用于引导模型正确处理长时命令、交互模式与超时场景
_BASH_DESCRIPTION = """在终端中执行 bash 命令。
* 长时间运行的命令：对于可能无限期运行的命令，应将其放到后台执行并把输出重定向到文件，例如 command = `python3 app.py > server.log 2>&1 &`。
* 交互模式：如果 bash 命令返回退出码 `-1`，表示进程尚未结束。此时助手可以再次调用终端并传入空的 `command`（用于获取后续日志），也可以向运行中进程的标准输入发送额外文本（将 `command` 设为该文本），或发送 command=`ctrl+c` 中断进程。
* 超时：如果执行结果提示 "超时：bash 在 N 秒内未返回，必须重启会话"，助手应改为在后台重新运行该命令。
"""


class _BashSession:
    """一个 bash shell 会话。

    内部维护一个常驻 bash 进程，跨多次命令复用；
    注意这是普通类（非 Pydantic 模型），状态字段需手动初始化。
    """

    # 仅类型声明、不创建属性：真正赋值发生在 __init__（_started）与 start()（_process）
    _started: bool
    _process: asyncio.subprocess.Process

    # 类属性：所有会话实例共享的默认配置
    command: str = "C:/Program Files/Git/bin/bash.exe"
    _output_delay: float = 0.2  # 轮询输出的间隔（秒）
    _timeout: float = 120.0  # 单次命令执行的超时时间（秒）
    _sentinel: str = "<<exit>>"  # 哨兵标记：命令尾部追加 echo，用于判断命令执行完毕（见 run()）

    def __init__(self):
        self._started = False
        self._timed_out = False

    async def start(self):
        if self._started:
            return

        self._process = await asyncio.create_subprocess_shell(
            self.command,
            # preexec_fn=os.setsid,  # （Unix）让子进程创建独立进程组，便于 terminate 时整组杀掉，避免残留孙进程
            shell=True,  # 经由 shell 解释执行
            bufsize=0,  # 不做 Python 侧缓冲，保证输出能被下方的轮询尽快读到
            stdin=asyncio.subprocess.PIPE,  # 三个流全部接管为管道，供会话持续读写
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._started = True

    def stop(self):
        """终止 bash shell 进程。"""
        if not self._started:
            raise ToolError("会话尚未启动。")
        if self._process.returncode is not None:
            # returncode 非 None 表示进程已经退出，无需再终止
            return
        self._process.terminate()

    async def run(self, command: str):
        """在 bash shell 会话中执行一条命令。"""
        if not self._started:
            raise ToolError("会话尚未启动。")
        if self._process.returncode is not None:
            # 进程已退出：无法再接收命令，提示必须重启会话
            return CLIResult(
                system="工具必须重启会话",
                error=f"bash 已退出，返回码 {self._process.returncode}",
            )
        if self._timed_out:
            # 之前发生过超时：缓冲区残留状态不可信，直接拒绝新命令
            raise ToolError(
                f"超时：bash 在 {self._timeout} 秒内未返回，必须重启会话",
            )

        # 由于创建进程时使用的是 PIPE，这三个流必定不为 None
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # 将命令写入进程标准输入，并追加哨兵标记，用于判断命令是否执行完毕
        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()  # 等待写入的数据真正刷入子进程（写缓冲清空，兼作背压控制）

        # 持续读取进程输出，直到出现哨兵标记
        try:
            async with asyncio.timeout(self._timeout):  # Python 3.11+ 的异步超时上下文管理器
                while True:
                    await asyncio.sleep(self._output_delay)
                    # 若直接读取 stdout/stderr 会无限等待 EOF，
                    # 因此直接读取 StreamReader 的内部缓冲区
                    output = (
                        self._process.stdout._buffer.decode()
                    )                            # pyright: ignore[reportAttributeAccessIssue]
                    if self._sentinel in output:
                        # 截掉哨兵标记并结束等待
                        output = output[: output.index(self._sentinel)]
                        break
        except asyncio.TimeoutError:
            # 标记会话失效，后续调用会在方法开头被拦截
            self._timed_out = True
            # `from None` 抑制异常链：不显示 TimeoutError 与本异常的关联信息，让报错更简洁
            raise ToolError(
                f"超时：bash 在 {self._timeout} 秒内未返回，必须重启会话",
            ) from None

        # 哨兵 echo 之前的输出通常以换行结尾，截掉末尾换行保持结果整洁
        if output.endswith("\n"):
            output = output[:-1]

        error = (
            self._process.stderr._buffer.decode()
        )                                      # pyright: ignore[reportAttributeAccessIssue]
        if error.endswith("\n"):
            error = error[:-1]

        # 清空缓冲区，确保下一条命令能读到正确的输出
        self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

        return CLIResult(output=output, error=error)


class Bash(BaseTool):
    """用于执行 bash 命令的工具。

    安全与状态控制：
    - 会话在首次调用时懒启动，跨调用保持复用；
    - 命令执行超时后标记会话失效，必须重启才能继续；
    - restart=True 时终止旧会话并重新创建。
    """

    name: str = "bash"
    description: str = _BASH_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的 bash 命令。可为空，用于在上一次退出码为 `-1` 时查看更多日志；也可传 `ctrl+c` 中断当前正在运行的进程。",
            },
        },
        "required": ["command"],
    }

    # 会话句柄：类级默认 None，各工具实例分别持有并懒创建（见 execute）
    _session: Optional[_BashSession] = None

    async def execute(
        # command 传 None 配合描述中的"交互模式"读取后续日志；
        # `str | None` 是 Python 3.10+ 的联合类型写法，等价于 Optional[str]
        self, command: str | None = None, restart: bool = False, **kwargs
    ) -> CLIResult:
        if restart:
            # 重启分支：终止旧会话并重新创建
            if self._session:
                self._session.stop()
            self._session = _BashSession()
            await self._session.start()

            return CLIResult(system="工具会话已重启。")

        # 懒启动：首次调用时创建 bash 会话
        if self._session is None:
            self._session = _BashSession()
            await self._session.start()

        if command is not None:
            return await self._session.run(command)

        raise ToolError("未提供命令。")


if __name__ == "__main__":
    bash = Bash()
    rst = asyncio.run(bash.execute("ls -l"))
    print(rst)
