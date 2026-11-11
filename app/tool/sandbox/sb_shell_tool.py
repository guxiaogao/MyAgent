import asyncio
import time
from typing import Any, Dict, Optional, TypeVar
from uuid import uuid4

from app.daytona.tool_base import Sandbox, SandboxToolsBase
from app.tool.base import ToolResult
from app.utils.logger import logger


Context = TypeVar("Context")
_SHELL_DESCRIPTION = """\
在工作区目录中执行 shell 命令。
重要：命令默认非阻塞执行，运行在 tmux 会话中。
非常适合启动服务器、构建进程等长时间运行的操作。
通过会话在多条命令之间维持状态。
本工具是运行 CLI 工具、安装软件包和管理系统操作的必备工具。
"""


class SandboxShellTool(SandboxToolsBase):
    """在 Daytona 沙箱中执行任务的工具（含浏览器自动化能力）。
    通过会话在命令之间维持状态，并提供完善的进程管理能力。
    """

    name: str = "sandbox_shell"
    description: str = _SHELL_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "execute_command",
                    "check_command_output",
                    "terminate_command",
                    "list_commands",
                ],
                "description": "要执行的 shell 操作",
            },
            "command": {
                "type": "string",
                "description": "要执行的 shell 命令。用于运行 CLI 工具、安装软件包"
                "或执行系统操作。命令可用 &&、|| 和 | 操作符串联。",
            },
            "folder": {
                "type": "string",
                "description": "可选，命令执行时所在 /workspace 子目录的相对路径。"
                "示例：'data/pdfs'",
            },
            "session_name": {
                "type": "string",
                "description": "可选，要使用的 tmux 会话名称。需要维持状态的关联命令"
                "应使用命名会话。默认使用随机会话名。",
            },
            "blocking": {
                "type": "boolean",
                "description": "是否等待命令执行完成。默认 false，即非阻塞执行。",
                "default": False,
            },
            "timeout": {
                "type": "integer",
                "description": "可选，阻塞式命令的超时时间（秒）。默认 60。对非阻塞命令无效。",
                "default": 60,
            },
            "kill_session": {
                "type": "boolean",
                "description": "检查输出后是否终止 tmux 会话。命令执行完毕不再需要时设为 true。",
                "default": False,
            },
        },
        "required": ["action"],
        "dependencies": {
            "execute_command": ["command"],
            "check_command_output": ["session_name"],
            "terminate_command": ["session_name"],
            "list_commands": [],
        },
    }

    def __init__(
        self, sandbox: Optional[Sandbox] = None, thread_id: Optional[str] = None, **data
    ):
        """使用可选的 sandbox 与 thread_id 进行初始化。"""
        super().__init__(**data)
        if sandbox is not None:
            self._sandbox = sandbox

    async def _ensure_session(self, session_name: str = "default") -> str:
        """确保会话存在并返回其 ID。"""
        if session_name not in self._sessions:
            session_id = str(uuid4())
            try:
                await self._ensure_sandbox()  # 确保沙箱已初始化
                self.sandbox.process.create_session(session_id)
                self._sessions[session_name] = session_id
            except Exception as e:
                raise RuntimeError(f"Failed to create session: {str(e)}")
        return self._sessions[session_name]

    async def _cleanup_session(self, session_name: str):
        """若会话存在则清理它。"""
        if session_name in self._sessions:
            try:
                await self._ensure_sandbox()  # 确保沙箱已初始化
                self.sandbox.process.delete_session(self._sessions[session_name])
                del self._sessions[session_name]
            except Exception as e:
                print(f"警告: 清理会话 {session_name} 失败: {str(e)}")

    async def _execute_raw_command(self, command: str) -> Dict[str, Any]:
        """直接在沙箱中执行原始命令。"""
        # 原始命令也需要在会话中执行
        session_id = await self._ensure_session("raw_commands")

        # 在会话中执行命令
        from app.daytona.sandbox import SessionExecuteRequest

        req = SessionExecuteRequest(
            command=command, run_async=False, cwd=self.workspace_path
        )

        response = self.sandbox.process.execute_session_command(
            session_id=session_id,
            req=req,
            timeout=30,  # 工具类命令使用较短超时
        )

        logs = self.sandbox.process.get_session_command_logs(
            session_id=session_id, command_id=response.cmd_id
        )

        return {"output": logs, "exit_code": response.exit_code}

    async def _execute_command(
        self,
        command: str,
        folder: Optional[str] = None,
        session_name: Optional[str] = None,
        blocking: bool = False,
        timeout: int = 60,
    ) -> ToolResult:
        try:
            # 确保沙箱已初始化
            await self._ensure_sandbox()

            # 设置工作目录
            cwd = self.workspace_path
            if folder:
                folder = folder.strip("/")
                cwd = f"{self.workspace_path}/{folder}"

            # 未指定会话名时生成随机会话名
            if not session_name:
                session_name = f"session_{str(uuid4())[:8]}"

            # 检查 tmux 会话是否已存在
            check_session = await self._execute_raw_command(
                f"tmux has-session -t {session_name} 2>/dev/null || echo 'not_exists'"
            )
            session_exists = "not_exists" not in check_session.get("output", "")

            if not session_exists:
                # 创建新的 tmux 会话
                await self._execute_raw_command(
                    f"tmux new-session -d -s {session_name}"
                )

            # 切换到目标目录后把命令发送进 tmux 会话
            full_command = f"cd {cwd} && {command}"
            wrapped_command = full_command.replace('"', '\\"')  # 转义双引号

            # 通过 send-keys 将命令注入 tmux 会话
            await self._execute_raw_command(
                f'tmux send-keys -t {session_name} "{wrapped_command}" Enter'
            )

            if blocking:
                # 阻塞模式：轮询等待并捕获输出
                start_time = time.time()
                while (time.time() - start_time) < timeout:
                    # 检查前先稍作等待
                    time.sleep(2)

                    # 检查会话是否仍存在（命令可能已退出）
                    check_result = await self._execute_raw_command(
                        f"tmux has-session -t {session_name} 2>/dev/null || echo 'ended'"
                    )
                    if "ended" in check_result.get("output", ""):
                        break

                    # 抓取当前输出并检查常见的完成标志
                    output_result = await self._execute_raw_command(
                        f"tmux capture-pane -t {session_name} -p -S - -E -"
                    )
                    current_output = output_result.get("output", "")

                    # 检查末尾几行是否出现提示命令完成的指示符
                    last_lines = current_output.split("\n")[-3:]
                    completion_indicators = [
                        "$",
                        "#",
                        ">",
                        "Done",
                        "Completed",
                        "Finished",
                        "✓",
                    ]
                    if any(
                        indicator in line
                        for indicator in completion_indicators
                        for line in last_lines
                    ):
                        break

                # 抓取最终输出
                output_result = await self._execute_raw_command(
                    f"tmux capture-pane -t {session_name} -p -S - -E -"
                )
                final_output = output_result.get("output", "")

                # 捕获完成后销毁会话
                await self._execute_raw_command(f"tmux kill-session -t {session_name}")

                return self.success_response(
                    {
                        "output": final_output,
                        "session_name": session_name,
                        "cwd": cwd,
                        "completed": True,
                    }
                )
            else:
                # 非阻塞模式：立即返回，由调用方稍后查询输出
                return self.success_response(
                    {
                        "session_name": session_name,
                        "cwd": cwd,
                        "message": f"Command sent to tmux session '{session_name}'. Use check_command_output to view results.",
                        "completed": False,
                    }
                )

        except Exception as e:
            # 出错时尝试清理会话
            if session_name:
                try:
                    await self._execute_raw_command(
                        f"tmux kill-session -t {session_name}"
                    )
                except:
                    pass
            return self.fail_response(f"Error executing command: {str(e)}")

    async def _check_command_output(
        self, session_name: str, kill_session: bool = False
    ) -> ToolResult:
        try:
            # 确保沙箱已初始化
            await self._ensure_sandbox()

            # 检查会话是否存在
            check_result = await self._execute_raw_command(
                f"tmux has-session -t {session_name} 2>/dev/null || echo 'not_exists'"
            )
            if "not_exists" in check_result.get("output", ""):
                return self.fail_response(
                    f"Tmux session '{session_name}' does not exist."
                )

            # 从 tmux 面板抓取输出
            output_result = await self._execute_raw_command(
                f"tmux capture-pane -t {session_name} -p -S - -E -"
            )
            output = output_result.get("output", "")

            # 按需终止会话
            if kill_session:
                await self._execute_raw_command(f"tmux kill-session -t {session_name}")
                termination_status = "Session terminated."
            else:
                termination_status = "Session still running."

            return self.success_response(
                {
                    "output": output,
                    "session_name": session_name,
                    "status": termination_status,
                }
            )

        except Exception as e:
            return self.fail_response(f"Error checking command output: {str(e)}")

    async def _terminate_command(self, session_name: str) -> ToolResult:
        try:
            # 确保沙箱已初始化
            await self._ensure_sandbox()

            # 检查会话是否存在
            check_result = await self._execute_raw_command(
                f"tmux has-session -t {session_name} 2>/dev/null || echo 'not_exists'"
            )
            if "not_exists" in check_result.get("output", ""):
                return self.fail_response(
                    f"Tmux session '{session_name}' does not exist."
                )

            # 终止会话
            await self._execute_raw_command(f"tmux kill-session -t {session_name}")

            return self.success_response(
                {"message": f"Tmux session '{session_name}' terminated successfully."}
            )

        except Exception as e:
            return self.fail_response(f"Error terminating command: {str(e)}")

    async def _list_commands(self) -> ToolResult:
        try:
            # 确保沙箱已初始化
            await self._ensure_sandbox()

            # 列出所有 tmux 会话
            result = await self._execute_raw_command(
                "tmux list-sessions 2>/dev/null || echo 'No sessions'"
            )
            output = result.get("output", "")

            if "No sessions" in output or not output.strip():
                return self.success_response(
                    {"message": "No active tmux sessions found.", "sessions": []}
                )

            # 解析会话列表（每行形如 "name: ..."，取冒号前的会话名）
            sessions = []
            for line in output.split("\n"):
                if line.strip():
                    parts = line.split(":")
                    if parts:
                        session_name = parts[0].strip()
                        sessions.append(session_name)

            return self.success_response(
                {
                    "message": f"Found {len(sessions)} active sessions.",
                    "sessions": sessions,
                }
            )

        except Exception as e:
            return self.fail_response(f"Error listing commands: {str(e)}")

    async def execute(
        self,
        action: str,
        command: str,
        folder: Optional[str] = None,
        session_name: Optional[str] = None,
        blocking: bool = False,
        timeout: int = 60,
        kill_session: bool = False,
    ) -> ToolResult:
        """
        在沙箱环境中执行 shell 操作。
        Args:
            timeout: 阻塞式命令的超时时间（秒）
            blocking: 是否阻塞等待命令完成
            session_name: tmux 会话名称
            folder: 命令执行的子目录
            command: 要执行的 shell 命令
            kill_session: 检查输出后是否终止会话
            action: 要执行的 shell 操作
        Returns:
            包含操作输出或错误信息的 ToolResult
        """
        async with asyncio.Lock():
            try:
                # 命令执行类操作
                if action == "execute_command":
                    if not command:
                        return self.fail_response("command is required for navigation")
                    return await self._execute_command(
                        command, folder, session_name, blocking, timeout
                    )
                elif action == "check_command_output":
                    if session_name is None:
                        return self.fail_response(
                            "session_name is required for navigation"
                        )
                    return await self._check_command_output(session_name, kill_session)
                elif action == "terminate_command":
                    if session_name is None:
                        return self.fail_response(
                            "session_name is required for click_element"
                        )
                    return await self._terminate_command(session_name)
                elif action == "list_commands":
                    return await self._list_commands()
                else:
                    return self.fail_response(f"Unknown action: {action}")
            except Exception as e:
                logger.error(f"执行 shell 操作时出错: {e}")
                return self.fail_response(f"Error executing shell action: {e}")

    async def cleanup(self):
        """清理所有会话。"""
        for session_name in list(self._sessions.keys()):
            await self._cleanup_session(session_name)

        # 同时清理所有 tmux 会话
        try:
            await self._ensure_sandbox()
            await self._execute_raw_command("tmux kill-server 2>/dev/null || true")
        except Exception as e:
            logger.error(f"清理 shell 沙箱操作出错: {e}")
