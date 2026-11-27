"""面向本地与沙箱两种环境的文件操作接口及实现。"""

import asyncio
import locale
from pathlib import Path
from typing import Optional, Protocol, Tuple, Union, runtime_checkable

from app.config import SandboxSettings
from app.exceptions import ToolError
from app.sandbox.client import SANDBOX_CLIENT


PathLike = Union[str, Path]


def _decode_process_output(data: bytes) -> str:
    """解码子进程输出，兼容 Windows 控制台的本地编码（如中文系统的 GBK/cp936）。

    依次尝试 UTF-8 与各系统编码，均失败时用 UTF-8 容错解码，
    保证任何情况下都不会因编码问题抛出 UnicodeDecodeError。
    注意：Python 处于 UTF-8 模式时 getpreferredencoding() 也返回 utf-8，
    因此需要 locale.getencoding() 和 mbcs（Windows ANSI 代码页）兜底。
    """
    import sys

    candidates = [
        "utf-8",
        locale.getpreferredencoding(False),
        locale.getencoding(),
    ]
    if sys.platform == "win32":
        candidates.append("mbcs")
    tried = set()
    for encoding in candidates:
        if not encoding or encoding in tried:
            continue
        tried.add(encoding)
        try:
            return data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


@runtime_checkable
class FileOperator(Protocol):
    """定义在不同运行环境下进行文件操作的统一接口。"""

    async def read_file(self, path: PathLike) -> str:
        """读取文件内容。"""
        ...

    async def write_file(self, path: PathLike, content: str) -> None:
        """写入内容到文件。"""
        ...

    async def is_directory(self, path: PathLike) -> bool:
        """判断路径是否指向目录。"""
        ...

    async def exists(self, path: PathLike) -> bool:
        """判断路径是否存在。"""
        ...

    async def run_command(
        self, cmd: str, timeout: Optional[float] = 120.0
    ) -> Tuple[int, str, str]:
        """执行 shell 命令，返回 (返回码, 标准输出, 标准错误)。"""
        ...


class LocalFileOperator(FileOperator):
    """基于本地文件系统的文件操作实现。"""

    encoding: str = "utf-8"

    async def read_file(self, path: PathLike) -> str:
        """读取本地文件内容。"""
        try:
            return Path(path).read_text(encoding=self.encoding)
        except Exception as e:
            raise ToolError(f"读取文件失败 {path}：{str(e)}") from None

    async def write_file(self, path: PathLike, content: str) -> None:
        """写入内容到本地文件。"""
        try:
            Path(path).write_text(content, encoding=self.encoding)
        except Exception as e:
            raise ToolError(f"写入文件失败 {path}：{str(e)}") from None

    async def is_directory(self, path: PathLike) -> bool:
        """判断路径是否指向目录。"""
        return Path(path).is_dir()

    async def exists(self, path: PathLike) -> bool:
        """判断路径是否存在。"""
        return Path(path).exists()

    async def run_command(
        self, cmd: str, timeout: Optional[float] = 120.0
    ) -> Tuple[int, str, str]:
        """在本地执行 shell 命令。"""
        process = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        try:
            # 带超时等待命令结束，超时则杀掉进程并抛出 TimeoutError
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )
            return (
                process.returncode or 0,
                _decode_process_output(stdout),
                _decode_process_output(stderr),
            )
        except asyncio.TimeoutError as exc:
            try:
                process.kill()
            except ProcessLookupError:
                pass
            raise TimeoutError(
                f"命令 '{cmd}' 在 {timeout} 秒后超时"
            ) from exc


class SandboxFileOperator(FileOperator):
    """基于沙箱环境的文件操作实现。"""

    def __init__(self):
        self.sandbox_client = SANDBOX_CLIENT

    async def _ensure_sandbox_initialized(self):
        """确保沙箱已初始化（懒创建）。"""
        if not self.sandbox_client.sandbox:
            await self.sandbox_client.create(config=SandboxSettings())

    async def read_file(self, path: PathLike) -> str:
        """读取沙箱内文件内容。"""
        await self._ensure_sandbox_initialized()
        try:
            return await self.sandbox_client.read_file(str(path))
        except Exception as e:
            raise ToolError(f"在沙箱中读取文件失败 {path}：{str(e)}") from None

    async def write_file(self, path: PathLike, content: str) -> None:
        """写入内容到沙箱内文件。"""
        await self._ensure_sandbox_initialized()
        try:
            await self.sandbox_client.write_file(str(path), content)
        except Exception as e:
            raise ToolError(f"在沙箱中写入文件失败 {path}：{str(e)}") from None

    async def is_directory(self, path: PathLike) -> bool:
        """判断沙箱内路径是否指向目录。"""
        await self._ensure_sandbox_initialized()
        result = await self.sandbox_client.run_command(
            f"test -d {path} && echo 'true' || echo 'false'"
        )
        return result.strip() == "true"

    async def exists(self, path: PathLike) -> bool:
        """判断沙箱内路径是否存在。"""
        await self._ensure_sandbox_initialized()
        result = await self.sandbox_client.run_command(
            f"test -e {path} && echo 'true' || echo 'false'"
        )
        return result.strip() == "true"

    async def run_command(
        self, cmd: str, timeout: Optional[float] = 120.0
    ) -> Tuple[int, str, str]:
        """在沙箱环境中执行命令。"""
        await self._ensure_sandbox_initialized()
        try:
            stdout = await self.sandbox_client.run_command(
                cmd, timeout=int(timeout) if timeout else None
            )
            return (
                0,  # 沙箱未提供独立的返回码，此处始终返回 0
                stdout,
                "",  # 当前沙箱实现未采集标准错误
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"命令 '{cmd}' 在沙箱中执行超时（{timeout} 秒）"
            ) from exc
        except Exception as exc:
            return 1, "", f"在沙箱中执行命令出错：{str(exc)}"
