from app.config import config
from app.tool.python_execute import PythonExecute


class NormalPythonExecute(PythonExecute):
    """在超时与安全限制下执行 Python 代码的工具。"""

    name: str = "python_execute"
    description: str = """执行 Python 代码，用于深入数据分析 / 数据报告（任务结论）/ 其他无需直接可视化的常规任务。"""
    parameters: dict = {
        "type": "object",
        "properties": {
            "code_type": {
                "description": "代码类型，数据处理 / 数据报告 / 其他",
                "type": "string",
                "default": "process",
                "enum": ["process", "report", "others"],
            },
            "code": {
                "type": "string",
                "description": """要执行的 Python 代码。
# 注意
1. 代码应生成一份内容全面的文本报告，包含数据集概览、字段详情、基础统计、派生指标、时间序列对比、异常值与关键洞察。
2. 所有输出都使用 print()，确保分析内容（如 'Dataset Overview'、'Preprocessing Results' 等章节）清晰可见并保存。
3. 将所有报告 / 处理后的文件 / 每项分析结果保存到工作区目录：{directory}
4. 数据报告需要内容丰富，包含完整的分析过程及相应的数据可视化。
5. 可以分步调用本工具，从概要到深入逐步完成数据分析，并同步保存数据报告""".format(
                    directory=config.workspace_root
                ),
            },
        },
        "required": ["code"],
    }

    async def execute(self, code: str, code_type: str | None = None, timeout=5):
        return await super().execute(code, timeout)
