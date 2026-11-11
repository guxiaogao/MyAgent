from typing import List

from duckduckgo_search import DDGS

from app.tool.search.base import SearchItem, WebSearchEngine


class DuckDuckGoSearchEngine(WebSearchEngine):
    def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[SearchItem]:
        """
        DuckDuckGo 搜索引擎。

        返回按 SearchItem 模型格式化后的结果。
        """
        raw_results = DDGS().text(query, max_results=num_results)

        results = []
        for i, item in enumerate(raw_results):
            if isinstance(item, str):
                # 结果仅为 URL 字符串
                results.append(
                    SearchItem(
                        title=f"DuckDuckGo Result {i + 1}", url=item, description=None
                    )
                )
            elif isinstance(item, dict):
                # 从字典中提取字段（DuckDuckGo 返回 title/href/body 键）
                results.append(
                    SearchItem(
                        title=item.get("title", f"DuckDuckGo Result {i + 1}"),
                        url=item.get("href", ""),
                        description=item.get("body", None),
                    )
                )
            else:
                # 尝试直接读取对象属性
                try:
                    results.append(
                        SearchItem(
                            title=getattr(item, "title", f"DuckDuckGo Result {i + 1}"),
                            url=getattr(item, "href", ""),
                            description=getattr(item, "body", None),
                        )
                    )
                except Exception:
                    # 回退为仅含 URL 的基础结果
                    results.append(
                        SearchItem(
                            title=f"DuckDuckGo Result {i + 1}",
                            url=str(item),
                            description=None,
                        )
                    )

        return results
