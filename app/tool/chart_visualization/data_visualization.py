import asyncio
import json
import os
from typing import Any, Hashable

import pandas as pd
from pydantic import Field, model_validator

from app.config import config
from app.llm import LLM
from app.logger import logger
from app.tool.base import BaseTool


class DataVisualization(BaseTool):
    name: str = "data_visualization"
    description: str = """根据 visualization_preparation 工具生成的 JSON 信息绘制统计图表或向图表添加洞察。可按以下步骤操作：
1. 绘制统计图表
2. 基于第 1 步的结果为图表挑选洞察（可选）
输出：
1. 图表（png/html）
2. 图表洞察（.md）（可选）"""
    parameters: dict = {
        "type": "object",
        "properties": {
            "json_path": {
                "type": "string",
                "description": """json 信息文件的路径，以 ".json" 结尾""",
            },
            "output_type": {
                "description": "渲染格式（html=可交互）",
                "type": "string",
                "default": "html",
                "enum": ["png", "html"],
            },
            "tool_type": {
                "description": "绘制图表或添加洞察",
                "type": "string",
                "default": "visualization",
                "enum": ["visualization", "insight"],
            },
            "language": {
                "description": "英文(en) / 中文(zh)",
                "type": "string",
                "default": "en",
                "enum": ["zh", "en"],
            },
        },
        "required": ["code"],
    }
    llm: LLM = Field(default_factory=LLM, description="语言模型实例")

    @model_validator(mode="after")
    def initialize_llm(self):
        """若未提供 llm，则使用默认配置初始化。"""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        return self

    def get_file_path(
        self,
        json_info: list[dict[str, str]],
        path_str: str,
        directory: str = None,
    ) -> list[str]:
        """解析 JSON 信息中的文件路径：优先按绝对路径取，其次拼接工作目录，均不存在则抛出异常。"""
        res = []
        for item in json_info:
            if os.path.exists(item[path_str]):
                res.append(item[path_str])
            elif os.path.exists(
                os.path.join(f"{directory or config.workspace_root}", item[path_str])
            ):
                res.append(
                    os.path.join(
                        f"{directory or config.workspace_root}", item[path_str]
                    )
                )
            else:
                raise Exception(f"文件或目录不存在：{item[path_str]}")
        return res

    def success_output_template(self, result: list[dict[str, str]]) -> str:
        """将成功的图表结果渲染为 Markdown 模板输出。"""
        content = ""
        if len(result) == 0:
            return "结果为空！"
        for item in result:
            content += f"""## {item['title']}\n图表已保存至：{item['chart_path']}"""
            if "insight_path" in item and item["insight_path"] and "insight_md" in item:
                content += "\n" + item["insight_md"]
            else:
                content += "\n"
        return f"图表生成成功！\n{content}"

    async def data_visualization(
        self, json_info: list[dict[str, str]], output_type: str, language: str
    ) -> str:
        """读取每个 CSV 并转为 JSON 记录，并发调用 vmind 子进程绘制图表，最后汇总成功/失败结果。"""
        data_list = []
        csv_file_path = self.get_file_path(json_info, "csvFilePath")
        for index, item in enumerate(json_info):
            # 读取 CSV 并做规范化：转 object 类型、空值统一置为 None，便于 JSON 序列化
            df = pd.read_csv(csv_file_path[index], encoding="utf-8")
            df = df.astype(object)
            df = df.where(pd.notnull(df), None)
            data_dict_list = df.to_json(orient="records", force_ascii=False)

            data_list.append(
                {
                    "file_name": os.path.basename(csv_file_path[index]).replace(
                        ".csv", ""
                    ),
                    "dict_data": data_dict_list,
                    "chartTitle": item["chartTitle"],
                }
            )
        # 为每个数据集并发创建绘图任务
        tasks = [
            self.invoke_vmind(
                dict_data=item["dict_data"],
                chart_description=item["chartTitle"],
                file_name=item["file_name"],
                output_type=output_type,
                task_type="visualization",
                language=language,
            )
            for item in data_list
        ]

        results = await asyncio.gather(*tasks)
        # 按结果中是否含 "error"（且无 "chart_path"）划分成功与失败
        error_list = []
        success_list = []
        for index, result in enumerate(results):
            csv_path = csv_file_path[index]
            if "error" in result and "chart_path" not in result:
                error_list.append(f"Error in {csv_path}: {result['error']}")
            else:
                success_list.append(
                    {
                        **result,
                        "title": json_info[index]["chartTitle"],
                    }
                )
        if len(error_list) > 0:
            # 存在失败时标记 success=False，同时保留已成功生成的图表信息
            return {
                "observation": f"# Error chart generated{'\n'.join(error_list)}\n{self.success_output_template(success_list)}",
                "success": False,
            }
        else:
            return {"observation": f"{self.success_output_template(success_list)}"}

    async def add_insighs(
        self, json_info: list[dict[str, str]], output_type: str
    ) -> str:
        """按 JSON 信息中的 insights_id 为已生成的图表添加洞察，并发调用 vmind 子进程。"""
        data_list = []
        chart_file_path = self.get_file_path(
            json_info, "chartPath", os.path.join(config.workspace_root, "visualization")
        )
        for index, item in enumerate(json_info):
            if "insights_id" in item:
                data_list.append(
                    {
                        "file_name": os.path.basename(chart_file_path[index]).replace(
                            f".{output_type}", ""
                        ),
                        "insights_id": item["insights_id"],
                    }
                )
        tasks = [
            self.invoke_vmind(
                insights_id=item["insights_id"],
                file_name=item["file_name"],
                output_type=output_type,
                task_type="insight",
            )
            for item in data_list
        ]
        results = await asyncio.gather(*tasks)
        error_list = []
        success_list = []
        for index, result in enumerate(results):
            chart_path = chart_file_path[index]
            if "error" in result and "chart_path" not in result:
                error_list.append(f"Error in {chart_path}: {result['error']}")
            else:
                success_list.append(chart_path)
        success_template = (
            f"# Charts Update with Insights\n{','.join(success_list)}"
            if len(success_list) > 0
            else ""
        )
        if len(error_list) > 0:
            return {
                "observation": f"# Error in chart insights:{'\n'.join(error_list)}\n{success_template}",
                "success": False,
            }
        else:
            return {"observation": f"{success_template}"}

    async def execute(
        self,
        json_path: str,
        output_type: str | None = "html",
        tool_type: str | None = "visualization",
        language: str | None = "en",
    ) -> str:
        """工具入口：读取 JSON 信息后按 tool_type 分发到绘图或添加洞察流程。"""
        try:
            logger.info(f"📈 data_visualization 使用 {json_path}，类型: {tool_type} ")
            with open(json_path, "r", encoding="utf-8") as file:
                json_info = json.load(file)
            if tool_type == "visualization":
                return await self.data_visualization(json_info, output_type, language)
            else:
                return await self.add_insighs(json_info, output_type)
        except Exception as e:
            return {
                "observation": f"Error: {e}",
                "success": False,
            }

    async def invoke_vmind(
        self,
        file_name: str,
        output_type: str,
        task_type: str,
        insights_id: list[str] = None,
        dict_data: list[dict[Hashable, Any]] = None,
        chart_description: str = None,
        language: str = "en",
    ):
        llm_config = {
            "base_url": self.llm.base_url,
            "model": self.llm.model,
            "api_key": self.llm.api_key,
        }
        vmind_params = {
            "llm_config": llm_config,
            "user_prompt": chart_description,
            "dataset": dict_data,
            "file_name": file_name,
            "output_type": output_type,
            "insights_id": insights_id,
            "task_type": task_type,
            "directory": str(config.workspace_root),
            "language": language,
        }
        # 构建异步子进程：通过 npx + ts-node 启动 Node.js 调用 vmind，参数经 stdin 以 JSON 传入
        process = await asyncio.create_subprocess_exec(
            "npx",
            "ts-node",
            "src/chartVisualize.ts",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.path.dirname(__file__),
        )
        input_json = json.dumps(vmind_params, ensure_ascii=False).encode("utf-8")
        try:
            stdout, stderr = await process.communicate(input_json)
            stdout_str = stdout.decode("utf-8")
            stderr_str = stderr.decode("utf-8")
            if process.returncode == 0:
                # 子进程成功时结果以 JSON 输出到 stdout
                return json.loads(stdout_str)
            else:
                return {"error": f"Node.js Error: {stderr_str}"}
        except Exception as e:
            return {"error": f"Subprocess Error: {str(e)}"}
