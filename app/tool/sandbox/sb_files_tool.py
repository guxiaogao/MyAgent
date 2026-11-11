import asyncio
from typing import Optional, TypeVar

from pydantic import Field

from app.daytona.tool_base import Sandbox, SandboxToolsBase
from app.tool.base import ToolResult
from app.utils.files_utils import clean_path, should_exclude_file
from app.utils.logger import logger


Context = TypeVar("Context")

_FILES_DESCRIPTION = """\
一个基于沙箱的文件系统工具，可在安全沙箱环境中执行文件操作。
* 本工具提供在工作区内创建、读取、更新、删除文件的各类命令
* 出于安全考虑，所有操作均相对于 /workspace 目录执行
* 当你需要在沙箱中管理文件、编辑代码或处理文件内容时使用本工具
* 每个操作需要按照工具 dependencies 定义提供相应参数
主要能力包括：
* 文件创建：按指定内容与权限创建新文件
* 文件修改：替换指定字符串或完全重写文件
* 文件删除：从工作区移除文件
* 文件读取：读取文件内容，可指定行范围
"""


class SandboxFilesTool(SandboxToolsBase):
    name: str = "sandbox_files"
    description: str = _FILES_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "create_file",
                    "str_replace",
                    "full_file_rewrite",
                    "delete_file",
                ],
                "description": "要执行的文件操作",
            },
            "file_path": {
                "type": "string",
                "description": "文件路径，相对于 /workspace（例如 'src/main.py'）",
            },
            "file_contents": {
                "type": "string",
                "description": "要写入文件的内容",
            },
            "old_str": {
                "type": "string",
                "description": "被替换的文本（必须恰好出现一次）",
            },
            "new_str": {
                "type": "string",
                "description": "替换后的文本",
            },
            "permissions": {
                "type": "string",
                "description": "八进制格式的文件权限（例如 '644'）",
                "default": "644",
            },
        },
        "required": ["action"],
        "dependencies": {
            "create_file": ["file_path", "file_contents"],
            "str_replace": ["file_path", "old_str", "new_str"],
            "full_file_rewrite": ["file_path", "file_contents"],
            "delete_file": ["file_path"],
        },
    }
    SNIPPET_LINES: int = Field(default=4, exclude=True)
    # workspace_path: str = Field(default="/workspace", exclude=True)
    # sandbox: Optional[Sandbox] = Field(default=None, exclude=True)

    def __init__(
        self, sandbox: Optional[Sandbox] = None, thread_id: Optional[str] = None, **data
    ):
        """使用可选的 sandbox 与 thread_id 进行初始化。"""
        super().__init__(**data)
        if sandbox is not None:
            self._sandbox = sandbox

    def clean_path(self, path: str) -> str:
        """清理并规范化路径，使其相对于 /workspace"""
        return clean_path(path, self.workspace_path)

    def _should_exclude_file(self, rel_path: str) -> bool:
        """根据路径、文件名或扩展名判断文件是否应被排除"""
        return should_exclude_file(rel_path)

    def _file_exists(self, path: str) -> bool:
        """检查文件是否存在于沙箱中"""
        try:
            self.sandbox.fs.get_file_info(path)
            return True
        except Exception:
            return False

    async def get_workspace_state(self) -> dict:
        """读取全部文件以获取当前工作区状态"""
        files_state = {}
        try:
            # 确保沙箱已初始化
            await self._ensure_sandbox()

            files = self.sandbox.fs.list_files(self.workspace_path)
            for file_info in files:
                rel_path = file_info.name

                # 跳过被排除的文件与目录
                if self._should_exclude_file(rel_path) or file_info.is_dir:
                    continue

                try:
                    full_path = f"{self.workspace_path}/{rel_path}"
                    content = self.sandbox.fs.download_file(full_path).decode()
                    files_state[rel_path] = {
                        "content": content,
                        "is_dir": file_info.is_dir,
                        "size": file_info.size,
                        "modified": file_info.mod_time,
                    }
                except Exception as e:
                    print(f"读取文件 {rel_path} 出错: {e}")
                except UnicodeDecodeError:
                    print(f"跳过二进制文件: {rel_path}")

            return files_state

        except Exception as e:
            print(f"获取工作区状态出错: {str(e)}")
            return {}

    async def execute(
        self,
        action: str,
        file_path: Optional[str] = None,
        file_contents: Optional[str] = None,
        old_str: Optional[str] = None,
        new_str: Optional[str] = None,
        permissions: Optional[str] = "644",
        **kwargs,
    ) -> ToolResult:
        """
        在沙箱环境中执行文件操作。
        Args:
            action: 要执行的文件操作
            file_path: 相对于 /workspace 的文件路径
            file_contents: 要写入文件的内容
            old_str: 被替换的文本（用于 str_replace）
            new_str: 替换后的文本（用于 str_replace）
            permissions: 八进制格式的文件权限
        Returns:
            包含操作输出或错误信息的 ToolResult
        """
        async with asyncio.Lock():
            try:
                # 文件创建
                if action == "create_file":
                    if not file_path or not file_contents:
                        return self.fail_response(
                            "file_path and file_contents are required for create_file"
                        )
                    return await self._create_file(
                        file_path, file_contents, permissions
                    )

                # 字符串替换
                elif action == "str_replace":
                    if not file_path or not old_str or not new_str:
                        return self.fail_response(
                            "file_path, old_str, and new_str are required for str_replace"
                        )
                    return await self._str_replace(file_path, old_str, new_str)

                # 整文件重写
                elif action == "full_file_rewrite":
                    if not file_path or not file_contents:
                        return self.fail_response(
                            "file_path and file_contents are required for full_file_rewrite"
                        )
                    return await self._full_file_rewrite(
                        file_path, file_contents, permissions
                    )

                # 文件删除
                elif action == "delete_file":
                    if not file_path:
                        return self.fail_response(
                            "file_path is required for delete_file"
                        )
                    return await self._delete_file(file_path)

                else:
                    return self.fail_response(f"Unknown action: {action}")

            except Exception as e:
                logger.error(f"执行文件操作时出错: {e}")
                return self.fail_response(f"Error executing file action: {e}")

    async def _create_file(
        self, file_path: str, file_contents: str, permissions: str = "644"
    ) -> ToolResult:
        """使用给定内容创建新文件"""
        try:
            # 确保沙箱已初始化
            await self._ensure_sandbox()

            file_path = self.clean_path(file_path)
            full_path = f"{self.workspace_path}/{file_path}"
            if self._file_exists(full_path):
                return self.fail_response(
                    f"File '{file_path}' already exists. Use full_file_rewrite to modify existing files."
                )

            # 按需创建父目录
            parent_dir = "/".join(full_path.split("/")[:-1])
            if parent_dir:
                self.sandbox.fs.create_folder(parent_dir, "755")

            # 写入文件内容并设置权限
            self.sandbox.fs.upload_file(file_contents.encode(), full_path)
            self.sandbox.fs.set_file_permissions(full_path, permissions)

            message = f"File '{file_path}' created successfully."

            # 若创建的是 index.html（仅限工作区根目录），附加 8080 预览服务器信息
            if file_path.lower() == "index.html":
                try:
                    website_link = self.sandbox.get_preview_link(8080)
                    website_url = (
                        website_link.url
                        if hasattr(website_link, "url")
                        else str(website_link).split("url='")[1].split("'")[0]
                    )
                    message += f"\n\n[Auto-detected index.html - HTTP server available at: {website_url}]"
                    message += "\n[Note: Use the provided HTTP server URL above instead of starting a new server]"
                except Exception as e:
                    logger.warning(f"获取 index.html 的网站 URL 失败: {str(e)}")

            return self.success_response(message)
        except Exception as e:
            return self.fail_response(f"Error creating file: {str(e)}")

    async def _str_replace(
        self, file_path: str, old_str: str, new_str: str
    ) -> ToolResult:
        """替换文件中的指定文本"""
        try:
            # 确保沙箱已初始化
            await self._ensure_sandbox()

            file_path = self.clean_path(file_path)
            full_path = f"{self.workspace_path}/{file_path}"
            if not self._file_exists(full_path):
                return self.fail_response(f"File '{file_path}' does not exist")

            content = self.sandbox.fs.download_file(full_path).decode()
            old_str = old_str.expandtabs()
            new_str = new_str.expandtabs()

            occurrences = content.count(old_str)
            if occurrences == 0:
                return self.fail_response(f"String '{old_str}' not found in file")
            if occurrences > 1:
                # 目标字符串出现多次时拒绝替换，避免误改
                lines = [
                    i + 1
                    for i, line in enumerate(content.split("\n"))
                    if old_str in line
                ]
                return self.fail_response(
                    f"Multiple occurrences found in lines {lines}. Please ensure string is unique"
                )

            # 执行替换并回写
            new_content = content.replace(old_str, new_str)
            self.sandbox.fs.upload_file(new_content.encode(), full_path)

            # 计算编辑位置附近的代码片段用于展示
            replacement_line = content.split(old_str)[0].count("\n")
            start_line = max(0, replacement_line - self.SNIPPET_LINES)
            end_line = replacement_line + self.SNIPPET_LINES + new_str.count("\n")
            snippet = "\n".join(new_content.split("\n")[start_line : end_line + 1])

            message = f"Replacement successful."

            return self.success_response(message)

        except Exception as e:
            return self.fail_response(f"Error replacing string: {str(e)}")

    async def _full_file_rewrite(
        self, file_path: str, file_contents: str, permissions: str = "644"
    ) -> ToolResult:
        """使用新内容完全重写已存在的文件"""
        try:
            # 确保沙箱已初始化
            await self._ensure_sandbox()

            file_path = self.clean_path(file_path)
            full_path = f"{self.workspace_path}/{file_path}"
            if not self._file_exists(full_path):
                return self.fail_response(
                    f"File '{file_path}' does not exist. Use create_file to create a new file."
                )

            self.sandbox.fs.upload_file(file_contents.encode(), full_path)
            self.sandbox.fs.set_file_permissions(full_path, permissions)

            message = f"File '{file_path}' completely rewritten successfully."

            # 若重写的是 index.html（仅限工作区根目录），附加 8080 预览服务器信息
            if file_path.lower() == "index.html":
                try:
                    website_link = self.sandbox.get_preview_link(8080)
                    website_url = (
                        website_link.url
                        if hasattr(website_link, "url")
                        else str(website_link).split("url='")[1].split("'")[0]
                    )
                    message += f"\n\n[Auto-detected index.html - HTTP server available at: {website_url}]"
                    message += "\n[Note: Use the provided HTTP server URL above instead of starting a new server]"
                except Exception as e:
                    logger.warning(f"获取 index.html 的网站 URL 失败: {str(e)}")

            return self.success_response(message)
        except Exception as e:
            return self.fail_response(f"Error rewriting file: {str(e)}")

    async def _delete_file(self, file_path: str) -> ToolResult:
        """删除指定路径的文件"""
        try:
            # 确保沙箱已初始化
            await self._ensure_sandbox()

            file_path = self.clean_path(file_path)
            full_path = f"{self.workspace_path}/{file_path}"
            if not self._file_exists(full_path):
                return self.fail_response(f"File '{file_path}' does not exist")

            self.sandbox.fs.delete_file(full_path)
            return self.success_response(f"File '{file_path}' deleted successfully.")
        except Exception as e:
            return self.fail_response(f"Error deleting file: {str(e)}")

    async def cleanup(self):
        """清理沙箱资源。"""

    @classmethod
    def create_with_context(cls, context: Context) -> "SandboxFilesTool[Context]":
        """使用指定上下文创建 SandboxFilesTool 的工厂方法。"""
        raise NotImplementedError(
            "create_with_context not implemented for SandboxFilesTool"
        )
