import json
import threading
import tomllib
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


def get_project_root() -> Path:
    """获取项目根目录"""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"


class LLMSettings(BaseModel):
    model: str = Field(..., description="模型名称")
    base_url: str = Field(..., description="API 基础地址")
    api_key: str = Field(..., description="API 密钥")
    max_tokens: int = Field(4096, description="每次请求允许的最大 token 数")
    max_input_tokens: Optional[int] = Field(
        None,
        description="所有请求累计可用的最大输入 token 数（None 表示不限制）",
    )
    temperature: float = Field(1.0, description="采样温度")
    api_type: str = Field(..., description="API 类型：Azure、Openai 或 Ollama")
    api_version: str = Field(..., description="使用 AzureOpenai 时所需的 Azure Openai 版本")


class ProxySettings(BaseModel):
    server: str = Field(None, description="代理服务器地址")
    username: Optional[str] = Field(None, description="代理用户名")
    password: Optional[str] = Field(None, description="代理密码")


class SearchSettings(BaseModel):
    engine: str = Field(default="Google", description="LLM 使用的搜索引擎")
    fallback_engines: List[str] = Field(
        default_factory=lambda: ["DuckDuckGo", "Baidu", "Bing"],
        description="主搜索引擎失败时依次尝试的备用搜索引擎",
    )
    retry_delay: int = Field(
        default=60,
        description="所有搜索引擎均失败后，重新重试全部引擎前等待的秒数",
    )
    max_retries: int = Field(
        default=3,
        description="所有搜索引擎均失败时重试全部引擎的最大次数",
    )
    lang: str = Field(
        default="en",
        description="搜索结果的语言代码（如 en、zh、fr）",
    )
    country: str = Field(
        default="us",
        description="搜索结果的国家代码（如 us、cn、uk）",
    )


class RunflowSettings(BaseModel):
    use_data_analysis_agent: bool = Field(
        default=False, description="是否在运行流程中启用数据分析智能体"
    )


class BrowserSettings(BaseModel):
    headless: bool = Field(False, description="是否以无头模式运行浏览器")
    disable_security: bool = Field(
        True, description="是否禁用浏览器安全特性"
    )
    extra_chromium_args: List[str] = Field(
        default_factory=list, description="传递给浏览器的额外启动参数"
    )
    chrome_instance_path: Optional[str] = Field(
        None, description="要使用的 Chrome 实例路径"
    )
    wss_url: Optional[str] = Field(
        None, description="通过 WebSocket 连接浏览器实例的地址"
    )
    cdp_url: Optional[str] = Field(
        None, description="通过 CDP 连接浏览器实例的地址"
    )
    proxy: Optional[ProxySettings] = Field(
        None, description="浏览器的代理设置"
    )
    max_content_length: int = Field(
        2000, description="内容获取操作允许的最大长度"
    )


class SandboxSettings(BaseModel):
    """执行沙箱的配置"""

    use_sandbox: bool = Field(False, description="是否启用沙箱")
    image: str = Field("python:3.12-slim", description="基础镜像")
    work_dir: str = Field("/workspace", description="容器内的工作目录")
    memory_limit: str = Field("512m", description="内存上限")
    cpu_limit: float = Field(1.0, description="CPU 上限")
    timeout: int = Field(300, description="默认命令超时时间（秒）")
    network_enabled: bool = Field(
        False, description="是否允许访问网络"
    )


class DaytonaSettings(BaseModel):
    daytona_api_key: str
    daytona_server_url: Optional[str] = Field(
        "https://app.daytona.io/api", description=""
    )
    daytona_target: Optional[str] = Field("us", description="枚举值 ['eu', 'us']")
    sandbox_image_name: Optional[str] = Field("whitezxj/sandbox:0.1.0", description="")
    sandbox_entrypoint: Optional[str] = Field(
        "/usr/bin/supervisord -n -c /etc/supervisor/conf.d/supervisord.conf",
        description="",
    )
    # sandbox_id: Optional[str] = Field(
    #     None, description="要使用的 Daytona 沙箱 ID（如有）"
    # )
    VNC_password: Optional[str] = Field(
        "123456", description="沙箱内 VNC 服务的密码"
    )


class MCPServerConfig(BaseModel):
    """单个 MCP 服务器的配置"""

    type: str = Field(..., description="服务器连接类型（sse 或 stdio）")
    url: Optional[str] = Field(None, description="SSE 连接使用的服务器 URL")
    command: Optional[str] = Field(None, description="stdio 连接使用的启动命令")
    args: List[str] = Field(
        default_factory=list, description="stdio 启动命令的参数列表"
    )


class MCPSettings(BaseModel):
    """MCP（模型上下文协议）配置"""

    server_reference: str = Field(
        "app.mcp.server", description="MCP 服务器的模块引用路径"
    )
    servers: Dict[str, MCPServerConfig] = Field(
        default_factory=dict, description="各 MCP 服务器的配置字典"
    )

    @classmethod
    def load_server_config(cls) -> Dict[str, MCPServerConfig]:
        """从 JSON 文件加载 MCP 服务器配置"""
        config_path = PROJECT_ROOT / "config" / "mcp.json"

        try:
            # 配置文件不存在时返回空字典，表示未配置任何 MCP 服务器
            config_file = config_path if config_path.exists() else None
            if not config_file:
                return {}

            with config_file.open() as f:
                data = json.load(f)
                servers = {}

                # 遍历 "mcpServers" 键下的每个服务器配置，转换为 MCPServerConfig 对象
                for server_id, server_config in data.get("mcpServers", {}).items():
                    servers[server_id] = MCPServerConfig(
                        type=server_config["type"],
                        url=server_config.get("url"),
                        command=server_config.get("command"),
                        args=server_config.get("args", []),
                    )
                return servers
        except Exception as e:
            raise ValueError(f"Failed to load MCP server config: {e}")


class AppConfig(BaseModel):
    llm: Dict[str, LLMSettings]
    sandbox: Optional[SandboxSettings] = Field(
        None, description="沙箱配置"
    )
    browser_config: Optional[BrowserSettings] = Field(
        None, description="浏览器配置"
    )
    search_config: Optional[SearchSettings] = Field(
        None, description="搜索配置"
    )
    mcp_config: Optional[MCPSettings] = Field(None, description="MCP 配置")
    run_flow_config: Optional[RunflowSettings] = Field(
        None, description="运行流程配置"
    )
    daytona_config: Optional[DaytonaSettings] = Field(
        None, description="Daytona 配置"
    )

    class Config:
        arbitrary_types_allowed = True


class Config:
    """全局配置单例：线程安全地加载并缓存整个应用的配置"""

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        # 双重检查锁定：避免每次实例化都竞争锁，同时保证仅创建一个实例
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # 仅在首次初始化时加载配置，之后直接复用缓存结果
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config = None
                    self._load_initial_config()
                    self._initialized = True

    @staticmethod
    def _get_config_path() -> Path:
        # 优先使用 config.toml；不存在时回退到 config.example.toml 示例配置
        root = PROJECT_ROOT
        config_path = root / "config" / "config.toml"
        if config_path.exists():
            return config_path
        example_path = root / "config" / "config.example.toml"
        if example_path.exists():
            return example_path
        raise FileNotFoundError("No configuration file found in config directory")

    def _load_config(self) -> dict:
        config_path = self._get_config_path()
        with config_path.open("rb") as f:
            return tomllib.load(f)

    def _load_initial_config(self):
        raw_config = self._load_config()
        base_llm = raw_config.get("llm", {})
        # llm 下的命名子表作为覆盖项，其余键构成默认配置
        llm_overrides = {
            k: v for k, v in raw_config.get("llm", {}).items() if isinstance(v, dict)
        }

        # 以 llm 顶层键构建默认 LLM 配置，缺失字段使用默认值
        default_settings = {
            "model": base_llm.get("model"),
            "base_url": base_llm.get("base_url"),
            "api_key": base_llm.get("api_key"),
            "max_tokens": base_llm.get("max_tokens", 4096),
            "max_input_tokens": base_llm.get("max_input_tokens"),
            "temperature": base_llm.get("temperature", 1.0),
            "api_type": base_llm.get("api_type", ""),
            "api_version": base_llm.get("api_version", ""),
        }

        # 处理浏览器配置。
        browser_config = raw_config.get("browser", {})
        browser_settings = None

        if browser_config:
            # 处理代理设置。
            proxy_config = browser_config.get("proxy", {})
            proxy_settings = None

            # 仅当配置了代理服务器地址时才构建 ProxySettings
            if proxy_config and proxy_config.get("server"):
                proxy_settings = ProxySettings(
                    **{
                        k: v
                        for k, v in proxy_config.items()
                        if k in ["server", "username", "password"] and v
                    }
                )

            # 过滤出合法的浏览器配置参数。
            valid_browser_params = {
                k: v
                for k, v in browser_config.items()
                if k in BrowserSettings.__annotations__ and v is not None
            }

            # 若存在代理设置，则并入浏览器参数。
            if proxy_settings:
                valid_browser_params["proxy"] = proxy_settings

            # 仅在存在合法参数时才创建 BrowserSettings。
            if valid_browser_params:
                browser_settings = BrowserSettings(**valid_browser_params)

        search_config = raw_config.get("search", {})
        search_settings = None
        if search_config:
            search_settings = SearchSettings(**search_config)
        sandbox_config = raw_config.get("sandbox", {})
        if sandbox_config:
            sandbox_settings = SandboxSettings(**sandbox_config)
        else:
            sandbox_settings = SandboxSettings()
        daytona_config = raw_config.get("daytona", {})
        if daytona_config:
            daytona_settings = DaytonaSettings(**daytona_config)
        else:
            daytona_settings = DaytonaSettings()

        mcp_config = raw_config.get("mcp", {})
        mcp_settings = None
        if mcp_config:
            # 从 JSON 文件加载 MCP 服务器配置
            mcp_config["servers"] = MCPSettings.load_server_config()
            mcp_settings = MCPSettings(**mcp_config)
        else:
            mcp_settings = MCPSettings(servers=MCPSettings.load_server_config())

        run_flow_config = raw_config.get("runflow")
        if run_flow_config:
            run_flow_settings = RunflowSettings(**run_flow_config)
        else:
            run_flow_settings = RunflowSettings()
        # "default" 为默认 LLM 配置，其余命名配置在默认值基础上合并各自的覆盖项
        config_dict = {
            "llm": {
                "default": default_settings,
                **{
                    name: {**default_settings, **override_config}
                    for name, override_config in llm_overrides.items()
                },
            },
            "sandbox": sandbox_settings,
            "browser_config": browser_settings,
            "search_config": search_settings,
            "mcp_config": mcp_settings,
            "run_flow_config": run_flow_settings,
            "daytona_config": daytona_settings,
        }

        self._config = AppConfig(**config_dict)

    @property
    def llm(self) -> Dict[str, LLMSettings]:
        return self._config.llm

    @property
    def sandbox(self) -> SandboxSettings:
        return self._config.sandbox

    @property
    def daytona(self) -> DaytonaSettings:
        return self._config.daytona_config

    @property
    def browser_config(self) -> Optional[BrowserSettings]:
        return self._config.browser_config

    @property
    def search_config(self) -> Optional[SearchSettings]:
        return self._config.search_config

    @property
    def mcp_config(self) -> MCPSettings:
        """获取 MCP 配置"""
        return self._config.mcp_config

    @property
    def run_flow_config(self) -> RunflowSettings:
        """获取运行流程（Run Flow）配置"""
        return self._config.run_flow_config

    @property
    def workspace_root(self) -> Path:
        """获取工作区根目录"""
        return WORKSPACE_ROOT

    @property
    def root_path(self) -> Path:
        """获取应用根路径"""
        return PROJECT_ROOT


config = Config()
