"""
OpenManus 的 Crawl4AI 网页爬取工具

本工具集成了 Crawl4AI——一个专为 LLM 与 AI 智能体设计的高性能网页爬虫，
提供快速、精准、面向 AI 的数据提取，并生成干净的 Markdown。
"""

import asyncio
from typing import List, Union
from urllib.parse import urlparse

from app.logger import logger
from app.tool.base import BaseTool, ToolResult


class Crawl4aiTool(BaseTool):
    """
    由 Crawl4AI 驱动的网页爬取工具。

    提供面向 AI 处理优化的干净 Markdown 内容提取。
    """

    name: str = "crawl4ai"
    description: str = """从网页中提取干净、可直接供 AI 使用的内容的爬虫。

    特性：
    - 提取为 LLM 优化的干净 Markdown 内容
    - 可处理重 JavaScript 的站点与动态内容
    - 单次请求支持多个 URL
    - 内置错误处理，快速可靠

    适用于内容分析、研究，以及向 AI 模型提供网页内容。"""

    parameters: dict = {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "description": "（必填）要爬取的 URL 列表，可以是单个或多个 URL。",
                "minItems": 1,
            },
            "timeout": {
                "type": "integer",
                "description": "（可选）每个 URL 的超时时间（秒），默认为 30。",
                "default": 30,
                "minimum": 5,
                "maximum": 120,
            },
            "bypass_cache": {
                "type": "boolean",
                "description": "（可选）是否绕过缓存以获取最新内容，默认为 false。",
                "default": False,
            },
            "word_count_threshold": {
                "type": "integer",
                "description": "（可选）内容块的最低字数阈值，默认为 10。",
                "default": 10,
                "minimum": 1,
            },
        },
        "required": ["urls"],
    }

    async def execute(
        self,
        urls: Union[str, List[str]],
        timeout: int = 30,
        bypass_cache: bool = False,
        word_count_threshold: int = 10,
    ) -> ToolResult:
        """
        对指定 URL 执行网页爬取。

        Args:
            urls: 单个 URL 字符串或 URL 列表
            timeout: 每个 URL 的超时时间（秒）
            bypass_cache: 是否绕过缓存
            word_count_threshold: 内容块的最低字数阈值

        Returns:
            携带爬取结果的 ToolResult
        """
        # 将 URL 输入规范化为列表
        if isinstance(urls, str):
            url_list = [urls]
        else:
            url_list = urls

        # 校验并过滤掉无效 URL
        valid_urls = []
        for url in url_list:
            if self._is_valid_url(url):
                valid_urls.append(url)
            else:
                logger.warning(f"跳过无效 URL：{url}")

        if not valid_urls:
            return ToolResult(error="未提供有效的 URL")

        try:
            # 按需导入 crawl4ai 组件
            from crawl4ai import (
                AsyncWebCrawler,
                BrowserConfig,
                CacheMode,
                CrawlerRunConfig,
            )

            # 浏览器配置：无头模式运行、忽略 HTTPS 证书错误、启用 JS
            browser_config = BrowserConfig(
                headless=True,
                verbose=False,
                browser_type="chromium",
                ignore_https_errors=True,
                java_script_enabled=True,
            )

            # 爬取配置：缓存策略、内容过滤、iframe 处理等
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS if bypass_cache else CacheMode.ENABLED,
                word_count_threshold=word_count_threshold,
                process_iframes=True,
                remove_overlay_elements=True,
                excluded_tags=["script", "style"],
                page_timeout=timeout * 1000,  # 秒转换为毫秒
                verbose=False,
                wait_until="domcontentloaded",
            )

            results = []
            successful_count = 0
            failed_count = 0

            # 逐个爬取 URL 并收集结果
            async with AsyncWebCrawler(config=browser_config) as crawler:
                for url in valid_urls:
                    try:
                        logger.info(f"🕷️ 正在爬取 URL：{url}")
                        start_time = asyncio.get_event_loop().time()

                        result = await crawler.arun(url=url, config=run_config)

                        end_time = asyncio.get_event_loop().time()
                        execution_time = end_time - start_time

                        if result.success:
                            # 统计 Markdown 中的词数
                            word_count = 0
                            if hasattr(result, "markdown") and result.markdown:
                                word_count = len(result.markdown.split())

                            # 统计链接数量
                            links_count = 0
                            if hasattr(result, "links") and result.links:
                                internal_links = result.links.get("internal", [])
                                external_links = result.links.get("external", [])
                                links_count = len(internal_links) + len(external_links)

                            # 统计图片数量
                            images_count = 0
                            if hasattr(result, "media") and result.media:
                                images = result.media.get("images", [])
                                images_count = len(images)

                            results.append(
                                {
                                    "url": url,
                                    "success": True,
                                    "status_code": getattr(result, "status_code", 200),
                                    "title": result.metadata.get("title")
                                    if result.metadata
                                    else None,
                                    "markdown": result.markdown
                                    if hasattr(result, "markdown")
                                    else None,
                                    "word_count": word_count,
                                    "links_count": links_count,
                                    "images_count": images_count,
                                    "execution_time": execution_time,
                                }
                            )
                            successful_count += 1
                            logger.info(
                                f"✅ 成功爬取 {url}，耗时 {execution_time:.2f} 秒"
                            )

                        else:
                            results.append(
                                {
                                    "url": url,
                                    "success": False,
                                    "error_message": getattr(
                                        result, "error_message", "Unknown error"
                                    ),
                                    "execution_time": execution_time,
                                }
                            )
                            failed_count += 1
                            logger.warning(f"❌ 爬取 {url} 失败")

                    except Exception as e:
                        error_msg = f"爬取 {url} 时出错：{str(e)}"
                        logger.error(error_msg)
                        results.append(
                            {"url": url, "success": False, "error_message": error_msg}
                        )
                        failed_count += 1

            # 格式化输出摘要
            output_lines = [f"🕷️ Crawl4AI 爬取结果摘要："]
            output_lines.append(f"📊 URL 总数：{len(valid_urls)}")
            output_lines.append(f"✅ 成功：{successful_count}")
            output_lines.append(f"❌ 失败：{failed_count}")
            output_lines.append("")

            for i, result in enumerate(results, 1):
                output_lines.append(f"{i}. {result['url']}")

                if result["success"]:
                    output_lines.append(
                        f"   ✅ 状态：成功（HTTP {result.get('status_code', 'N/A')}）"
                    )
                    if result.get("title"):
                        output_lines.append(f"   📄 标题：{result['title']}")

                    if result.get("markdown"):
                        # 仅展示 Markdown 内容的前 300 个字符
                        content_preview = result["markdown"]
                        if len(result["markdown"]) > 300:
                            content_preview += "..."
                        output_lines.append(f"   📝 内容：{content_preview}")

                    output_lines.append(
                        f"   📊 统计：{result.get('word_count', 0)} 词，{result.get('links_count', 0)} 个链接，{result.get('images_count', 0)} 张图片"
                    )

                    if result.get("execution_time"):
                        output_lines.append(
                            f"   ⏱️ 耗时：{result['execution_time']:.2f} 秒"
                        )
                else:
                    output_lines.append(f"   ❌ 状态：失败")
                    if result.get("error_message"):
                        output_lines.append(f"   🚫 错误：{result['error_message']}")

                output_lines.append("")

            return ToolResult(output="\n".join(output_lines))

        except ImportError:
            error_msg = "Crawl4AI 未安装。请执行以下命令安装：pip install crawl4ai"
            logger.error(error_msg)
            return ToolResult(error=error_msg)
        except Exception as e:
            error_msg = f"Crawl4AI 执行失败：{str(e)}"
            logger.error(error_msg)
            return ToolResult(error=error_msg)

    def _is_valid_url(self, url: str) -> bool:
        """校验 URL 格式是否合法（仅允许 http/https 协议）。"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc]) and result.scheme in [
                "http",
                "https",
            ]
        except Exception:
            return False
