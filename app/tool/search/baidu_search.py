from typing import List

from baidusearch.baidusearch import search

from app.tool.search.base import SearchItem, WebSearchEngine


class BaiduSearchEngine(WebSearchEngine):
    def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[SearchItem]:
        """
        百度搜索引擎。

        返回按 SearchItem 模型格式化后的结果。
        """
        raw_results = search(query, num_results=num_results)

        # 将原始结果转换为 SearchItem 格式
        results = []
        for i, item in enumerate(raw_results):
            if isinstance(item, str):
                # 结果仅为 URL 字符串
                results.append(
                    SearchItem(title=f"Baidu Result {i+1}", url=item, description=None)
                )
            elif isinstance(item, dict):
                # 结果为包含详细信息的字典
                results.append(
                    SearchItem(
                        title=item.get("title", f"Baidu Result {i+1}"),
                        url=item.get("url", ""),
                        description=item.get("abstract", None),
                    )
                )
            else:
                # 尝试直接读取对象属性
                try:
                    results.append(
                        SearchItem(
                            title=getattr(item, "title", f"Baidu Result {i+1}"),
                            url=getattr(item, "url", ""),
                            description=getattr(item, "abstract", None),
                        )
                    )
                except Exception:
                    # 回退为仅含 URL 的基础结果
                    results.append(
                        SearchItem(
                            title=f"Baidu Result {i+1}", url=str(item), description=None
                        )
                    )

        return results
