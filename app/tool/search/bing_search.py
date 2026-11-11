from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from app.logger import logger
from app.tool.search.base import SearchItem, WebSearchEngine


ABSTRACT_MAX_LENGTH = 300  # 单条结果摘要的最大长度，超出则截断

# 随机 UA 池：模拟不同浏览器请求，降低被 Bing 反爬机制识别的概率
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36",
    "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/49.0.2623.108 Chrome/49.0.2623.108 Safari/537.36",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; pt-BR) AppleWebKit/533.3 (KHTML, like Gecko) QtWeb Internet Browser/3.7 http://www.QtWeb.net",
    "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.36",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US) AppleWebKit/532.2 (KHTML, like Gecko) ChromePlus/4.0.222.3 Chrome/4.0.222.3 Safari/532.2",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.4pre) Gecko/20070404 K-Ninja/2.1.3",
    "Mozilla/5.0 (Future Star Technologies Corp.; Star-Blade OS; x86_64; U; en-US) iNet Browser 4.7",
    "Mozilla/5.0 (Windows; U; Windows NT 6.1; rv:2.2) Gecko/20110201",
    "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.13) Gecko/20080414 Firefox/2.0.0.13 Pogo/2.0.0.13.6866",
]

HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Content-Type": "application/x-www-form-urlencoded",
    "User-Agent": USER_AGENTS[0],
    "Referer": "https://www.bing.com/",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

BING_HOST_URL = "https://www.bing.com"
BING_SEARCH_URL = "https://www.bing.com/search?q="


class BingSearchEngine(WebSearchEngine):
    session: Optional[requests.Session] = None

    def __init__(self, **data):
        """初始化 BingSearch 工具，创建携带请求头的 requests 会话。"""
        super().__init__(**data)
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _search_sync(self, query: str, num_results: int = 10) -> List[SearchItem]:
        """
        同步实现 Bing 搜索以获取搜索结果。

        Args:
            query (str): 提交给 Bing 的查询词。
            num_results (int, optional): 最多返回的结果数量，默认为 10。

        Returns:
            List[SearchItem]: 包含标题、URL 与描述的搜索结果列表。
        """
        if not query:
            return []

        list_result = []
        first = 1  # 当前页码对应的结果起始序号，Bing 每页约 10 条
        next_url = BING_SEARCH_URL + query

        # 结果不足时沿"下一页"链接翻页，直到凑够数量或无下一页
        while len(list_result) < num_results:
            data, next_url = self._parse_html(
                next_url, rank_start=len(list_result), first=first
            )
            if data:
                list_result.extend(data)
            if not next_url:
                break
            first += 10

        return list_result[:num_results]

    def _parse_html(
        self, url: str, rank_start: int = 0, first: int = 1
    ) -> Tuple[List[SearchItem], str]:
        """
        解析 Bing 搜索结果 HTML，提取搜索结果与下一页 URL。

        Returns:
            tuple: (SearchItem 对象列表, 下一页 URL 或 None)
        """
        try:
            res = self.session.get(url=url)
            res.encoding = "utf-8"
            root = BeautifulSoup(res.text, "lxml")

            list_data = []
            # 搜索结果列表位于 <ol id="b_results"> 节点中
            ol_results = root.find("ol", id="b_results")
            if not ol_results:
                return [], None

            # 每条自然结果对应一个 <li class="b_algo">
            for li in ol_results.find_all("li", class_="b_algo"):
                title = ""
                url = ""
                abstract = ""
                try:
                    # 标题与链接位于 <h2><a href="..."> 结构中
                    h2 = li.find("h2")
                    if h2:
                        title = h2.text.strip()
                        url = h2.a["href"].strip()

                    # 摘要位于结果块内的第一个 <p> 中
                    p = li.find("p")
                    if p:
                        abstract = p.text.strip()

                    if ABSTRACT_MAX_LENGTH and len(abstract) > ABSTRACT_MAX_LENGTH:
                        abstract = abstract[:ABSTRACT_MAX_LENGTH]

                    rank_start += 1

                    # 构造 SearchItem 对象
                    list_data.append(
                        SearchItem(
                            title=title or f"Bing Result {rank_start}",
                            url=url,
                            description=abstract,
                        )
                    )
                except Exception:
                    # 单条结果解析失败时跳过，不影响整体
                    continue

            # 定位"下一页"按钮，存在则拼接绝对 URL 以便继续翻页
            next_btn = root.find("a", title="Next page")
            if not next_btn:
                return list_data, None

            next_url = BING_HOST_URL + next_btn["href"]
            return list_data, next_url
        except Exception as e:
            logger.warning(f"解析 HTML 时出错: {e}")
            return [], None

    def perform_search(
        self, query: str, num_results: int = 10, *args, **kwargs
    ) -> List[SearchItem]:
        """
        Bing 搜索引擎。

        返回按 SearchItem 模型格式化后的结果。
        """
        return self._search_sync(query, num_results=num_results)
