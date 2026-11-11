import multiprocessing
import sys
from io import StringIO
from typing import Dict

from app.tool.base import BaseTool


class PythonExecute(BaseTool):
    """带超时控制与安全限制的 Python 代码执行工具。"""

    name: str = "python_execute"
    description: str = "执行 Python 代码字符串。注意：只有 print 输出可见，函数返回值不会被捕获。请使用 print 语句查看结果。"
    parameters: dict = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "要执行的 Python 代码。",
            },
        },
        "required": ["code"],
    }

    def _run_code(self, code: str, result_dict: dict, safe_globals: dict) -> None:
        # 在子进程中运行：重定向 stdout 到缓冲区以捕获 print 输出，
        # 执行结果（输出或异常信息）写入共享字典带回主进程
        original_stdout = sys.stdout
        try:
            output_buffer = StringIO()
            sys.stdout = output_buffer
            exec(code, safe_globals, safe_globals)
            result_dict["observation"] = output_buffer.getvalue()
            result_dict["success"] = True
        except Exception as e:
            result_dict["observation"] = str(e)
            result_dict["success"] = False
        finally:
            sys.stdout = original_stdout

    async def execute(
        self,
        code: str,
        timeout: int = 5,
    ) -> Dict:
        """
        执行给定的 Python 代码，并施加超时限制。

        Args:
            code (str): 要执行的 Python 代码。
            timeout (int): 执行超时时间（秒）。

        Returns:
            Dict: 包含 'output'（执行输出或错误信息）与 'success'（是否成功）。
        """

        # 关键安全控制：在独立子进程中执行代码，避免死循环/阻塞拖垮主进程；
        # 通过 Manager 创建跨进程共享字典，用于收集子进程的执行结果
        with multiprocessing.Manager() as manager:
            result = manager.dict({"observation": "", "success": False})
            if isinstance(__builtins__, dict):
                safe_globals = {"__builtins__": __builtins__}
            else:
                safe_globals = {"__builtins__": __builtins__.__dict__.copy()}
            proc = multiprocessing.Process(
                target=self._run_code, args=(code, result, safe_globals)
            )
            proc.start()
            proc.join(timeout)

            # 超时处理：超过限定时间后强制终止子进程并返回超时提示
            if proc.is_alive():
                proc.terminate()
                proc.join(1)
                return {
                    "observation": f"执行超时（{timeout} 秒）",
                    "success": False,
                }
            return dict(result)
