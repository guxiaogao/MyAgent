"""支持沙箱环境的文件与目录操作工具。"""

from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, List, Literal, Optional, get_args

from app.config import config
from app.exceptions import ToolError
from app.tool import BaseTool
from app.tool.base import CLIResult, ToolResult
from app.tool.file_operators import (
    FileOperator,
    LocalFileOperator,
    PathLike,
    SandboxFileOperator,
)


Command = Literal[
    "view",
    "create",
    "str_replace",
    "insert",
    "undo_edit",
]

# 常量定义：SNIPPET_LINES 为编辑结果展示的上下文行数，MAX_RESPONSE_LEN 为响应长度上限
SNIPPET_LINES: int = 4
MAX_RESPONSE_LEN: int = 16000
TRUNCATED_MESSAGE: str = (
    "<response clipped><NOTE>To save on context only part of this file has been shown to you. "
    "You should retry this tool after you have searched inside the file with `grep -n` "
    "in order to find the line numbers of what you are looking for.</NOTE>"
)

# 工具描述（供 LLM 理解工具用途）
_STR_REPLACE_EDITOR_DESCRIPTION = """用于查看、创建和编辑文件的自定义编辑工具
* 状态在多次命令调用和与用户的对话间持久保留
* 若 `path` 是文件，`view` 显示 `cat -n` 的执行结果；若 `path` 是目录，`view` 列出其中最深 2 层的非隐藏文件和目录
* 若指定的 `path` 已存在同名文件，则不能使用 `create` 命令
* 若某个 `command` 产生过长输出，输出会被截断并以 `<response clipped>` 标记
* `undo_edit` 命令会撤销对 `path` 文件的最近一次编辑

使用 `str_replace` 命令的注意事项：
* `old_str` 参数必须与原文件中一个或多个连续行完全一致，注意空格！
* 若 `old_str` 在文件中不唯一，则不会执行替换。请确保 `old_str` 中包含足够的上下文使其唯一
* `new_str` 参数应包含用于替换 `old_str` 的编辑后内容
"""


def maybe_truncate(
    content: str, truncate_after: Optional[int] = MAX_RESPONSE_LEN
) -> str:
    """若内容超过指定长度则截断，并追加截断提示。"""
    # 结果截断控制：超出上限时截断并追加标记，提示 LLM 先用 grep -n 定位行号后重试
    if not truncate_after or len(content) <= truncate_after:
        return content
    return content[:truncate_after] + TRUNCATED_MESSAGE


class StrReplaceEditor(BaseTool):
    """用于查看、创建和编辑文件的工具，支持在沙箱环境中执行。"""

    name: str = "str_replace_editor"
    description: str = _STR_REPLACE_EDITOR_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "description": "要执行的命令。可选值：`view`、`create`、`str_replace`、`insert`、`undo_edit`。",
                "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                "type": "string",
            },
            "path": {
                "description": "文件或目录的绝对路径。",
                "type": "string",
            },
            "file_text": {
                "description": "`create` 命令的必填参数，为要创建文件的内容。",
                "type": "string",
            },
            "old_str": {
                "description": "`str_replace` 命令的必填参数，为 `path` 文件中要被替换的字符串。",
                "type": "string",
            },
            "new_str": {
                "description": "`str_replace` 命令的可选参数，为替换后的新字符串（不传则不添加任何内容）；也是 `insert` 命令的必填参数，为要插入的字符串。",
                "type": "string",
            },
            "insert_line": {
                "description": "`insert` 命令的必填参数。`new_str` 将插入到 `path` 的 `insert_line` 行之后。",
                "type": "integer",
            },
            "view_range": {
                "description": "`view` 命令在 `path` 指向文件时的可选参数。不传则显示整个文件；传入时按指定行号范围显示，例如 [11, 12] 显示第 11、12 行。行号从 1 开始。设置 `[start_line, -1]` 表示显示从 `start_line` 到文件末尾的所有行。",
                "items": {"type": "integer"},
                "type": "array",
            },
        },
        "required": ["command", "path"],
    }
    _file_history: DefaultDict[PathLike, List[str]] = defaultdict(list)
    _local_operator: LocalFileOperator = LocalFileOperator()
    _sandbox_operator: SandboxFileOperator = SandboxFileOperator()

    # def _get_operator(self, use_sandbox: bool) -> FileOperator:
    def _get_operator(self) -> FileOperator:
        """根据执行模式（沙箱/本地）获取对应的文件操作器。"""
        return (
            self._sandbox_operator
            if config.sandbox.use_sandbox
            else self._local_operator
        )

    async def execute(
        self,
        *,
        command: Command,
        path: str,
        file_text: str | None = None,
        view_range: list[int] | None = None,
        old_str: str | None = None,
        new_str: str | None = None,
        insert_line: int | None = None,
        **kwargs: Any,
    ) -> str:
        """执行文件操作命令。"""
        # 根据执行模式（沙箱/本地）选择对应的文件操作器
        operator = self._get_operator()

        # 校验路径与命令组合是否合法
        await self.validate_path(command, Path(path), operator)

        # 分发并执行对应的命令
        if command == "view":
            result = await self.view(path, view_range, operator)
        elif command == "create":
            if file_text is None:
                raise ToolError("Parameter `file_text` is required for command: create")
            await operator.write_file(path, file_text)
            self._file_history[path].append(file_text)
            result = ToolResult(output=f"File created successfully at: {path}")
        elif command == "str_replace":
            if old_str is None:
                raise ToolError(
                    "Parameter `old_str` is required for command: str_replace"
                )
            result = await self.str_replace(path, old_str, new_str, operator)
        elif command == "insert":
            if insert_line is None:
                raise ToolError(
                    "Parameter `insert_line` is required for command: insert"
                )
            if new_str is None:
                raise ToolError("Parameter `new_str` is required for command: insert")
            result = await self.insert(path, insert_line, new_str, operator)
        elif command == "undo_edit":
            result = await self.undo_edit(path, operator)
        else:
            # 正常情况下类型检查会拦截到此处，这里作为兜底保护
            raise ToolError(
                f'Unrecognized command {command}. The allowed commands for the {self.name} tool are: {", ".join(get_args(Command))}'
            )

        return str(result)

    async def validate_path(
        self, command: str, path: Path, operator: FileOperator
    ) -> None:
        """基于执行环境校验路径与命令组合的合法性。"""
        # 检查路径是否为绝对路径
        if not path.is_absolute():
            raise ToolError(f"The path {path} is not an absolute path")

        # 非 create 命令要求路径必须已存在
        if command != "create":
            if not await operator.exists(path):
                raise ToolError(
                    f"The path {path} does not exist. Please provide a valid path."
                )

            # 检查路径是否为目录：目录仅允许使用 view 命令
            is_dir = await operator.is_directory(path)
            if is_dir and command != "view":
                raise ToolError(
                    f"The path {path} is a directory and only the `view` command can be used on directories"
                )

        # create 命令要求目标文件不能已存在（防止误覆盖）
        elif command == "create":
            exists = await operator.exists(path)
            if exists:
                raise ToolError(
                    f"File already exists at: {path}. Cannot overwrite files using command `create`."
                )

    async def view(
        self,
        path: PathLike,
        view_range: Optional[List[int]] = None,
        operator: FileOperator = None,
    ) -> CLIResult:
        """展示文件或目录内容。"""
        # 判断路径是否为目录
        is_dir = await operator.is_directory(path)

        if is_dir:
            # 目录分支：不支持 view_range，仅列出最深两层的条目
            if view_range:
                raise ToolError(
                    "The `view_range` parameter is not allowed when `path` points to a directory."
                )

            return await self._view_directory(path, operator)
        else:
            # 文件分支：支持按行号范围展示
            return await self._view_file(path, operator, view_range)

    @staticmethod
    async def _view_directory(path: PathLike, operator: FileOperator) -> CLIResult:
        """展示目录内容。"""
        find_cmd = f"find {path} -maxdepth 2 -not -path '*/\\.*'"

        # 通过文件操作器执行 find 命令
        returncode, stdout, stderr = await operator.run_command(find_cmd)

        if not stderr:
            stdout = (
                f"Here's the files and directories up to 2 levels deep in {path}, "
                f"excluding hidden items:\n{stdout}\n"
            )

        return CLIResult(output=stdout, error=stderr)

    async def _view_file(
        self,
        path: PathLike,
        operator: FileOperator,
        view_range: Optional[List[int]] = None,
    ) -> CLIResult:
        """展示文件内容，可限定在指定行号范围内。"""
        # 读取文件内容
        file_content = await operator.read_file(path)
        init_line = 1

        # 指定了 view_range 时解析并校验行号范围
        if view_range:
            if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
                raise ToolError(
                    "Invalid `view_range`. It should be a list of two integers."
                )

            file_lines = file_content.split("\n")
            n_lines_file = len(file_lines)
            init_line, final_line = view_range

            # 校验行号范围的边界
            if init_line < 1 or init_line > n_lines_file:
                raise ToolError(
                    f"Invalid `view_range`: {view_range}. Its first element `{init_line}` should be "
                    f"within the range of lines of the file: {[1, n_lines_file]}"
                )
            if final_line > n_lines_file:
                raise ToolError(
                    f"Invalid `view_range`: {view_range}. Its second element `{final_line}` should be "
                    f"smaller than the number of lines in the file: `{n_lines_file}`"
                )
            if final_line != -1 and final_line < init_line:
                raise ToolError(
                    f"Invalid `view_range`: {view_range}. Its second element `{final_line}` should be "
                    f"larger or equal than its first `{init_line}`"
                )

            # 按范围切出对应行（-1 表示直到文件末尾）
            if final_line == -1:
                file_content = "\n".join(file_lines[init_line - 1 :])
            else:
                file_content = "\n".join(file_lines[init_line - 1 : final_line])

        # 格式化（附行号）并返回结果
        return CLIResult(
            output=self._make_output(file_content, str(path), init_line=init_line)
        )

    async def str_replace(
        self,
        path: PathLike,
        old_str: str,
        new_str: Optional[str] = None,
        operator: FileOperator = None,
    ) -> CLIResult:
        """将文件中唯一出现的字符串替换为新字符串。"""
        # 读取文件内容并展开制表符，避免匹配时错位
        file_content = (await operator.read_file(path)).expandtabs()
        old_str = old_str.expandtabs()
        new_str = new_str.expandtabs() if new_str is not None else ""

        # 唯一性校验：old_str 不唯一时拒绝替换，防止误改
        occurrences = file_content.count(old_str)
        if occurrences == 0:
            raise ToolError(
                f"No replacement was performed, old_str `{old_str}` did not appear verbatim in {path}."
            )
        elif occurrences > 1:
            # 定位各次出现的行号，便于报错提示
            file_content_lines = file_content.split("\n")
            lines = [
                idx + 1
                for idx, line in enumerate(file_content_lines)
                if old_str in line
            ]
            raise ToolError(
                f"No replacement was performed. Multiple occurrences of old_str `{old_str}` "
                f"in lines {lines}. Please ensure it is unique"
            )

        # 执行替换
        new_file_content = file_content.replace(old_str, new_str)

        # 将新内容写回文件
        await operator.write_file(path, new_file_content)

        # 把原始内容存入历史，供 undo_edit 回滚
        self._file_history[path].append(file_content)

        # 生成编辑位置附近的代码片段，便于复核
        replacement_line = file_content.split(old_str)[0].count("\n")
        start_line = max(0, replacement_line - SNIPPET_LINES)
        end_line = replacement_line + SNIPPET_LINES + new_str.count("\n")
        snippet = "\n".join(new_file_content.split("\n")[start_line : end_line + 1])

        # 组装成功信息：片段加后续修改建议
        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet, f"a snippet of {path}", start_line + 1
        )
        success_msg += "Review the changes and make sure they are as expected. Edit the file again if necessary."

        return CLIResult(output=success_msg)

    async def insert(
        self,
        path: PathLike,
        insert_line: int,
        new_str: str,
        operator: FileOperator = None,
    ) -> CLIResult:
        """在文件的指定行之后插入文本。"""
        # 读取文件并预处理内容
        file_text = (await operator.read_file(path)).expandtabs()
        new_str = new_str.expandtabs()
        file_text_lines = file_text.split("\n")
        n_lines_file = len(file_text_lines)

        # 校验插入行号的取值范围
        if insert_line < 0 or insert_line > n_lines_file:
            raise ToolError(
                f"Invalid `insert_line` parameter: {insert_line}. It should be within "
                f"the range of lines of the file: {[0, n_lines_file]}"
            )

        # 在 insert_line 之后插入新行
        new_str_lines = new_str.split("\n")
        new_file_text_lines = (
            file_text_lines[:insert_line]
            + new_str_lines
            + file_text_lines[insert_line:]
        )

        # 生成前后若干行组成的片段，便于预览
        snippet_lines = (
            file_text_lines[max(0, insert_line - SNIPPET_LINES) : insert_line]
            + new_str_lines
            + file_text_lines[insert_line : insert_line + SNIPPET_LINES]
        )

        # 合并行并写入文件
        new_file_text = "\n".join(new_file_text_lines)
        snippet = "\n".join(snippet_lines)

        await operator.write_file(path, new_file_text)
        self._file_history[path].append(file_text)

        # 组装成功信息
        success_msg = f"The file {path} has been edited. "
        success_msg += self._make_output(
            snippet,
            "a snippet of the edited file",
            max(1, insert_line - SNIPPET_LINES + 1),
        )
        success_msg += "Review the changes and make sure they are as expected (correct indentation, no duplicate lines, etc). Edit the file again if necessary."

        return CLIResult(output=success_msg)

    async def undo_edit(
        self, path: PathLike, operator: FileOperator = None
    ) -> CLIResult:
        """撤销对文件的最近一次编辑。"""
        if not self._file_history[path]:
            raise ToolError(f"No edit history found for {path}.")

        old_text = self._file_history[path].pop()
        await operator.write_file(path, old_text)

        return CLIResult(
            output=f"Last edit to {path} undone successfully. {self._make_output(old_text, str(path))}"
        )

    def _make_output(
        self,
        file_content: str,
        file_descriptor: str,
        init_line: int = 1,
        expand_tabs: bool = True,
    ) -> str:
        """格式化文件内容以供展示，附带行号。"""
        # 展示前先做长度截断控制
        file_content = maybe_truncate(file_content)
        if expand_tabs:
            file_content = file_content.expandtabs()

        # 为每行添加行号（6 位宽度 + 制表符）
        file_content = "\n".join(
            [
                f"{i + init_line:6}\t{line}"
                for i, line in enumerate(file_content.split("\n"))
            ]
        )

        return (
            f"Here's the result of running `cat -n` on {file_descriptor}:\n"
            + file_content
            + "\n"
        )
