from app.tool.chart_visualization.python_execute import NormalPythonExecute


class VisualizationPrepare(NormalPythonExecute):
    """用于图表生成准备阶段的工具"""

    name: str = "visualization_preparation"
    description: str = "使用 Python 代码生成 data_visualization 工具所需的元数据。输出：1) JSON 信息；2) 清洗后的 CSV 数据文件（可选）。"
    parameters: dict = {
        "type": "object",
        "properties": {
            "code_type": {
                "description": "代码类型，visualization：csv -> 图表；insight：为图表挑选洞察",
                "type": "string",
                "default": "visualization",
                "enum": ["visualization", "insight"],
            },
            "code": {
                "type": "string",
                "description": """用于 data_visualization 准备阶段的 Python 代码。
## visualization 类型
1. 数据加载逻辑
2. 生成 CSV 数据与图表描述
2.1 CSV 数据（你要可视化的数据，从原始数据清洗/转换而来，保存为 .csv）
2.2 CSV 数据对应的图表描述（图表标题或描述应简洁清晰。示例：'产品销量分布'、'月度营收趋势'。）
3. 将信息保存到 json 文件（格式：{"csvFilePath": string, "chartTitle": string}[]）
## insight 类型
1. 从 data_visualization 结果中选出你想添加到图表上的洞察。
2. 将信息保存到 json 文件（格式：{"chartPath": string, "insights_id": number[]}[]）
# 注意
1. 你可以根据不同的可视化需求生成一个或多个 csv 数据文件。
2. 每个图表的数据应简单、干净且彼此不同。
3. json 文件以 utf-8 保存，并用 print 输出路径：print(json_path)
""",
            },
        },
        "required": ["code", "code_type"],
    }
