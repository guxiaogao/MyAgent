import math
from typing import Dict, List, Optional, Union

import tiktoken
from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.bedrock import BedrockClient
from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.logger import logger  # 使用应用中已配置的 logger
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
)


# 推理模型清单（o1 / o3-mini 这类）。
# 与普通对话模型的差异：① 不接受 temperature 参数；② 输出上限参数名是
# max_completion_tokens，而不是 max_tokens。下方 ask / ask_with_images / ask_tool
# 都会依据该清单切换请求参数，写错会导致 API 直接报 400。
REASONING_MODELS = ["o1", "o3-mini"]

# 支持图片输入的多模态模型清单。
# 只有模型的名称在此清单中，format_messages 才会把 base64_image 组装成
# OpenAI 的 image_url 结构；否则只丢弃图片字段，仅保留文本内容。
# 注意：此处是精确字符串匹配，新增模型（如 gpt-4.1）不会自动生效。
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


class TokenCounter:
    """本地 token 计数器：在发请求之前估算 token 数量，实现「超限前置拦截」。

    设计动机：把注定超限的请求发给服务端，既要等网络往返、又会拿到 400 报错，
    白白浪费一次调用。本地先用分词器算一遍，就能在客户端提前拦下。
    估算口径模仿 OpenAI 的计费规则：每条消息有固定开销，图片按切片数量计费。
    这不追求与官方完全一致（不同 provider 口径有差异），只要求「宁可高估」。
    """

    # Token 常量（数值对应 OpenAI 官方计费规则）
    # 每条消息在计费上固定额外计 4 个 token，用于 role 等结构开销
    BASE_MESSAGE_TOKENS = 4
    # 整个 messages 数组还要再加 2 个 token，作为整体格式开销
    FORMAT_TOKENS = 2
    # 低细节（low）图片固定按 85 token 计费
    LOW_DETAIL_IMAGE_TOKENS = 85
    # 高细节（high）图片按 512x512 切片，每片 170 token
    HIGH_DETAIL_TILE_TOKENS = 170

    # 图片处理常量
    # 高细节图片第一步先把长边压缩到不超过 2048
    MAX_SIZE = 2048
    # 第二步把最短边放大/缩小到 768
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768
    # 切片边长 512
    TILE_SIZE = 512

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        """计算文本字符串的 token 数"""
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        """
        根据细节级别与图片尺寸计算图片的 token 数

        "low" 细节级别：固定 85 token
        "high" 细节级别：
        1. 缩放到 2048x2048 正方形内
        2. 将最短边缩放至 768px
        3. 按 512px 切片计数（每片 170 token）
        4. 再加 85 token
        """
        # detail 缺省取 "medium"：OpenAI 请求里不写 detail 时默认就是中等级别
        detail = image_item.get("detail", "medium")

        # 低细节级别直接返回固定 token 数
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        # 中等细节级别（OpenAI 的默认值）沿用高细节级别的计算方式
        # OpenAI 未单独定义中等级别的计算规则

        # 高细节级别：若提供了尺寸信息则按尺寸计算
        if detail == "high" or detail == "medium":
            # image_item 中提供了图片尺寸时
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        # 兜底分支：
        # ① detail == "high" 但没给尺寸 → 按 1024x1024 估算（常见默认分辨率）
        # ② 其他未知 detail 取值       → 直接给 1024 的保守估值
        # 原则是"宁高不低"，避免漏判导致真实请求被服务端拒绝。
        return (
            self._calculate_high_detail_tokens(1024, 1024) if detail == "high" else 1024
        )

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """根据图片尺寸计算高细节级别图片的 token 数"""
        # 第 1 步：缩放到 MAX_SIZE x MAX_SIZE 正方形内
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # 第 2 步：缩放使最短边等于 HIGH_DETAIL_TARGET_SHORT_SIDE。
        # 这一步是"放大"而非"缩小"——短边不足 768 时会被放大，
        # 目的是保证模型能看清细节；放大后切片数随之增加、token 也增加。
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # 第 3 步：统计 512px 切片数量
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # 第 4 步：计算最终 token 数
        return (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """计算消息内容的 token 数"""
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """计算工具调用的 token 数"""
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        """估算整个 messages 数组的输入 token 数。

        计费模型（对齐 OpenAI 口径）：
            总量 = 2（整体格式开销）
                 + Σ 每条消息[ 4（消息固定开销）
                              + role 的 token
                              + content 的 token（文本 + 图片切片）
                              + tool_calls 的 token
                              + name / tool_call_id 的 token ]
        """
        total_tokens = self.FORMAT_TOKENS  # 整个列表的基础格式 token

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # 每条消息的基础 token

            # 累加角色 token
            tokens += self.count_text(message.get("role", ""))

            # 累加内容 token
            if "content" in message:
                tokens += self.count_content(message["content"])

            # 累加工具调用 token
            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            # 累加 name 与 tool_call_id 的 token
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens

        return total_tokens


class LLM:
    """LLM 客户端封装：按配置名以单例方式管理，负责消息格式化、token 统计与重试请求"""

    # 类级实例缓存：key 为 config_name（如 "default" / "manus" / "claude"），value 为 LLM 实例。
    # 这样不同组件按名字取到的是同一个客户端对象，token 累计统计也是共享的。
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        # 按 config_name 做单例：已存在则直接返回缓存实例，不再新建。
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            # 这里手动调了一次 __init__。注意：Python 在 __new__ 返回实例后
            # 还会自动再调一次 __init__，所以 __init__ 实际会被执行两次，
            # 必须靠 __init__ 里的 hasattr 判断保证真正的初始化只做一次。
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        # 幂等守卫：首次构造时 client 还不存在，执行完整初始化；
        # 之后无论是 Python 自动补调，还是从缓存取出实例，都会因 client 已存在而直接跳过。
        # 这同时保护了 token 累计计数，避免实例被重复初始化导致统计清零。
        if not hasattr(self, "client"):  # 仅在未初始化时执行初始化
            # 配置来源优先级：显式传入的 llm_config > 全局配置 config.llm。
            llm_config = llm_config or config.llm
            # 按配置名取对应段落（如 [llm.manus]）；取不到则回退到 [llm.default]，
            # 保证任何智能体都能拿到一份可用的兜底配置。
            llm_config = llm_config.get(config_name, llm_config["default"])
            self.model = llm_config.model
            self.max_tokens = llm_config.max_tokens
            self.temperature = llm_config.temperature
            self.api_type = llm_config.api_type
            self.api_key = llm_config.api_key
            self.api_version = llm_config.api_version
            self.base_url = llm_config.base_url

            # token 统计属性：单例实例上持续累计，不会随单次请求清零。
            self.total_input_tokens = 0
            self.total_completion_tokens = 0
            # max_input_tokens 为 None 表示不做上限控制；
            # 只有配置里显式填了数值，check_token_limit 才会执行拦截。
            self.max_input_tokens = (
                llm_config.max_input_tokens
                if hasattr(llm_config, "max_input_tokens")
                else None
            )

            # 初始化分词器：优先使用与模型精确匹配的编码表
            try:
                self.tokenizer = tiktoken.encoding_for_model(self.model)
            except KeyError:
                # 模型不在 tiktoken 预设中（DeepSeek、Qwen、Claude 等都会命中这里），
                # 退化为 cl100k_base。此时 token 数只是近似值，
                # 中英文混排下与真实计数可能偏差，因此超限判断留有余量更稳妥。
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

            # 按 api_type 选择对应的后端客户端：
            # azure  → Azure 专用客户端，需要额外传 api_version
            # aws    → Bedrock 客户端（内部做 OpenAI 格式与 Bedrock 格式互转）
            # 其他   → 标准 OpenAI 兼容接口（DeepSeek、通义等通过 base_url 接入）
            if self.api_type == "azure":
                self.client = AsyncAzureOpenAI(
                    base_url=self.base_url,
                    api_key=self.api_key,
                    api_version=self.api_version,
                )
            elif self.api_type == "aws":
                self.client = BedrockClient()
            else:
                self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

            self.token_counter = TokenCounter(self.tokenizer)

    def count_tokens(self, text: str) -> int:
        """计算文本中的 token 数量"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[dict]) -> int:
        return self.token_counter.count_message_tokens(messages)

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """更新 token 统计（无条件累计，与是否设置 max_input_tokens 无关）"""
        # 累计值是实例级、跨请求持久存在的；单例模式下所有调用方共享同一份统计。
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(
            f"Token 用量: 输入={input_tokens}, 补全={completion_tokens}, "
            f"累计输入={self.total_input_tokens}, 累计补全={self.total_completion_tokens}, "
            f"本次总计={input_tokens + completion_tokens}, 累计总计={self.total_input_tokens + self.total_completion_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        """检查「累计已用 + 本次预估」是否仍在 max_input_tokens 之内。

        注意这是累计量控制，不是单次请求控制：返回 True 只代表还没超总额度。
        """
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        # 未设置 max_input_tokens 时始终返回 True（不限制）
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        """生成超出 token 上限的错误消息"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"请求可能超出输入 token 上限（当前已用：{self.total_input_tokens}，本次需要：{input_tokens}，上限：{self.max_input_tokens}）"

        return "已超出 token 上限"

    @staticmethod   # 把下面的`format_messages` 变成 静态方法，写在类里的普通函数
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False
    ) -> List[dict]:
        """
        将消息转换为 OpenAI 消息格式以便发送给 LLM。

        Args:
            messages: 消息列表，元素可为 dict 或 Message 对象
            supports_images: 标记目标模型是否支持图片输入

        Returns:
            List[dict]: 转换为 OpenAI 格式的消息列表

        Raises:
            ValueError: 消息无效或缺少必需字段时抛出
            TypeError: 提供了不支持的消息类型时抛出

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            # 将 Message 对象转换为字典。
            # to_dict() 会返回一份全新的 dict，因此后续对它的改写不会影响原对象。
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # 注意：这里直接操作的是调用方传入的同一个 dict 对象（Message 已转成新 dict，
                # 但外部传进来的纯 dict 不会被复制）。因此下面修改 content、
                # 删除 base64_image 等操作，都会真实改动调用方的原始字典。
                # 复用同一批消息对象时需留意这一副作用。

                # dict 消息必须包含必需字段
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                # 模型支持图片且消息带 base64 图片时，转换为多模态内容格式
                if supports_images and message.get("base64_image"):
                    # 将 content 初始化或转换为合适的格式
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        # 将字符串元素转换为规范的文本对象
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    # 将图片追加到 content 中
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{message['base64_image']}"
                            },
                        }
                    )

                    # 移除 base64_image 字段
                    del message["base64_image"]
                # 模型不支持图片但消息带 base64 图片时，做兼容处理
                elif not supports_images and message.get("base64_image"):
                    # 仅移除 base64_image 字段并保留文本内容
                    del message["base64_image"]

                # 只有带 content 或 tool_calls 的消息才会被发送。
                # 二者皆无（例如只有 assistant 的空壳消息）会被静默丢弃。
                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                # else: 不收录该消息
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # 校验所有消息的角色字段均合法。
        # 这一步放在过滤之后，是对最终要发出去的集合做统一兜底检查，
        # 防止拼装过程中混入非法 role（如手写的字典里写了 "bot"）。
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages

    # 重试策略：随机指数退避（1 秒起、上限 60 秒），最多尝试 6 次。
    # 重试条件为排除法：TokenLimitExceeded（确定性失败，重试结果不变）与
    # AuthenticationError（401 认证错误，重试无意义）立即向上抛出，其余异常照常重试。

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(
            (TokenLimitExceeded, AuthenticationError)
        ),  # 重试条件说明见上方注释
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        向 LLM 发送提示词并获取响应。

        Args:
            messages: 会话消息列表
            system_msgs: 可选的、前置拼接的系统消息
            stream (bool): 是否以流式方式返回响应
            temperature (float): 响应的采样温度

        Returns:
            str: 生成的响应内容

        Raises:
            TokenLimitExceeded: 超出 token 上限时抛出
            ValueError: 消息无效或响应为空时抛出
            OpenAIError: 重试耗尽后 API 调用仍失败时抛出
            Exception: 其他意外错误
        """
        try:
            # 判断当前模型是否支持图片：决定 format_messages 是否把 base64 图片
            # 转成 image_url 多模态结构（不支持则直接丢弃图片只发文本）
            supports_images = self.model in MULTIMODAL_MODELS

            # 格式化消息：系统消息在前、会话消息在后。
            # 二者分开格式化是为了保证顺序固定——system 必须位于对话历史之前，
            # 否则部分 provider 会报错或降低指令遵循度。
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # 计算本次请求的预估输入 token 数（含消息结构与图片切片开销）
            input_tokens = self.count_message_tokens(messages)

            # 超限前置拦截：本地先算，避免把注定失败的请求发到服务端。
            # 这里的上限是"累计已用 + 本次预估"，见 check_token_limit 的说明。
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # 抛出 TokenLimitExceeded，由上层智能体决定如何处理
                raise TokenLimitExceeded(error_message)

            # 组装基础请求参数
            params = {
                "model": self.model,
                "messages": messages,
            }

            # 推理模型与普通模型的参数名不同：
            # o1/o3-mini 只能用 max_completion_tokens，且不接受 temperature；
            # 其余模型用 max_tokens + temperature。
            # 外部传入的 temperature 优先于实例默认值。
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            if not stream:
                # 非流式请求：一次性拿到完整响应，usage 字段由服务端返回，
                # 因此输入/补全 token 都是精确值，直接写入统计即可。
                response = await self.client.chat.completions.create(
                    **params, stream=False
                )

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                # 更新 token 统计（使用服务端返回的精确 usage）
                self.update_token_count(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )

                return response.choices[0].message.content

            # 流式请求：服务端不会在流里返回 usage，只能在请求前按本地估算值记账。
            # 注意：若本次因异常被重试，这段累计会重复执行，导致 total_input_tokens
            # 被多算——这是"估算 + 流式"组合下的已知代价，仅影响统计显示，不影响请求本身。
            self.update_token_count(input_tokens)

            response = await self.client.chat.completions.create(**params, stream=True)

            # 逐块接收增量内容：delta.content 是本次新增的片段，
            # 既打印到终端（实时打字机效果），也累积起来拼成完整响应。
            collected_messages = []
            completion_text = ""
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                completion_text += chunk_message
                print(chunk_message, end="", flush=True)

            print()  # 流式输出结束后换行
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            # 流式响应的 completion token 只能事后本地估算，再手动累加
            completion_tokens = self.count_tokens(completion_text)
            logger.info(
                f"流式响应的预估补全 token 数: {completion_tokens}"
            )
            self.total_completion_tokens += completion_tokens

            return full_response

        except TokenLimitExceeded:
            # token 上限错误直接重新抛出，不做日志记录
            raise
        except ValueError:
            logger.exception(f"校验错误")
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API 错误")
            if isinstance(oe, AuthenticationError):
                logger.error("认证失败，请检查 API 密钥。")
            elif isinstance(oe, RateLimitError):
                logger.error("已超出速率限制，可考虑增加重试次数。")
            elif isinstance(oe, APIError):
                logger.error(f"API 错误: {oe}")
            raise
        except Exception:
            logger.exception(f"ask 方法出现意外错误")
            raise

    # 重试策略：随机指数退避（1 秒起、上限 60 秒），最多尝试 6 次。
    # 重试条件为排除法：TokenLimitExceeded（确定性失败）与 AuthenticationError
    # （401 认证错误）立即向上抛出，其余异常照常重试。详见 ask 方法上方注释。
    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(
            (TokenLimitExceeded, AuthenticationError)
        ),  # 重试条件说明见 ask 方法上方的注释
    )
    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        向 LLM 发送带图片的提示词并获取响应。

        Args:
            messages: 会话消息列表
            images: 图片 URL 或图片数据字典的列表
            system_msgs: 可选的、前置拼接的系统消息
            stream (bool): 是否以流式方式返回响应
            temperature (float): 响应的采样温度

        Returns:
            str: 生成的响应内容

        Raises:
            TokenLimitExceeded: 超出 token 上限时抛出
            ValueError: 消息无效或响应为空时抛出
            OpenAIError: 重试耗尽后 API 调用仍失败时抛出
            Exception: 其他意外错误
        """
        try:
            # 该方法固定按"支持图片"处理，因此必须确保模型本身是多模态的；
            # 否则图片会被静默丢弃，调用方却以为图片已发出，属于隐性错误。
            if self.model not in MULTIMODAL_MODELS:
                raise ValueError(
                    f"Model {self.model} does not support images. Use a model from {MULTIMODAL_MODELS}"
                )

            # 以图片支持模式格式化消息（此时 base64_image 会被转成 image_url 结构）
            formatted_messages = self.format_messages(messages, supports_images=True)

            # 图片只能挂在"最后一条用户消息"上：这是多模态 API 的约定，
            # 挂在 assistant/system/tool 消息上不会被模型识别。
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "The last message must be from the user to attach images"
                )

            # 取最后一条用户消息作为图片的挂载点
            last_message = formatted_messages[-1]

            # 把 content 统一成"内容块列表"：
            # 多模态 API 要求 content 是 [{"type": "text", ...}, {"type": "image_url", ...}] 形式，
            # 因此字符串要包成 text 块，列表原样使用，其他类型（如 None）置为空列表。
            content = last_message["content"]
            multimodal_content = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content
                if isinstance(content, list)
                else []
            )

            # 逐张追加图片，兼容三种传入形态：
            # ① 纯字符串      → 视为图片 URL
            # ② {"url": ...}  → 补一层 {"type": "image_url", "image_url": {...}}
            # ③ {"image_url": ...} → 已经是标准结构，直接追加
            for image in images:
                if isinstance(image, str):
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image}}
                    )
                elif isinstance(image, dict) and "url" in image:
                    multimodal_content.append({"type": "image_url", "image_url": image})
                elif isinstance(image, dict) and "image_url" in image:
                    multimodal_content.append(image)
                else:
                    raise ValueError(f"Unsupported image format: {image}")

            # 用多模态内容更新该消息
            last_message["content"] = multimodal_content

            # 如有系统消息则前置拼接
            if system_msgs:
                all_messages = (
                    self.format_messages(system_msgs, supports_images=True)
                    + formatted_messages
                )
            else:
                all_messages = formatted_messages

            # 计算 token 并检查上限
            input_tokens = self.count_message_tokens(all_messages)
            if not self.check_token_limit(input_tokens):
                raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

            # 设置 API 请求参数
            params = {
                "model": self.model,
                "messages": all_messages,
                "stream": stream,
            }

            # 按模型类型追加参数
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # 处理非流式请求
            if not stream:
                response = await self.client.chat.completions.create(**params)

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                self.update_token_count(response.usage.prompt_tokens)
                return response.choices[0].message.content

            # 处理流式请求
            self.update_token_count(input_tokens)
            response = await self.client.chat.completions.create(**params)

            collected_messages = []
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                print(chunk_message, end="", flush=True)

            print()  # 流式输出结束后换行
            full_response = "".join(collected_messages).strip()

            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            return full_response

        except TokenLimitExceeded:
            raise
        except ValueError as ve:
            logger.error(f"ask_with_images 校验错误: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API 错误: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("认证失败，请检查 API 密钥。")
            elif isinstance(oe, RateLimitError):
                logger.error("已超出速率限制，可考虑增加重试次数。")
            elif isinstance(oe, APIError):
                logger.error(f"API 错误: {oe}")
            raise
        except Exception as e:
            logger.error(f"ask_with_images 出现意外错误: {e}")
            raise

    # 重试策略：随机指数退避（1 秒起、上限 60 秒），最多尝试 6 次。
    # 重试条件为排除法：TokenLimitExceeded（确定性失败）与 AuthenticationError
    # （401 认证错误）立即向上抛出，其余异常照常重试。详见 ask 方法上方注释。
    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_not_exception_type(
            (TokenLimitExceeded, AuthenticationError)
        ),  # 重试条件说明见 ask 方法上方的注释
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        **kwargs,
    ) -> ChatCompletionMessage | None:
        """
        以函数/工具方式询问 LLM 并返回响应。

        Args:
            messages: 会话消息列表
            system_msgs: 可选的、前置拼接的系统消息
            timeout: 请求超时时间（秒）
            tools: 要使用的工具列表
            tool_choice: 工具调用策略
            temperature: 响应的采样温度
            **kwargs: 其他补全请求参数

        Returns:
            ChatCompletionMessage: 模型的响应

        Raises:
            TokenLimitExceeded: 超出 token 上限时抛出
            ValueError: tools、tool_choice 或消息无效时抛出
            OpenAIError: 重试耗尽后 API 调用仍失败时抛出
            Exception: 其他意外错误
        """
        try:
            # 校验 tool_choice 取值
            if tool_choice not in TOOL_CHOICE_VALUES:
                raise ValueError(f"Invalid tool_choice: {tool_choice}")

            # 检查模型是否支持图片输入
            supports_images = self.model in MULTIMODAL_MODELS

            # 格式化消息
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # 计算输入 token 数
            input_tokens = self.count_message_tokens(messages)

            # 有工具时，工具定义（名称、描述、JSON Schema）本身也要占用输入 token。
            # 这里用 str(tool) 的 token 数做近似——工具定义是 dict，转字符串后
            # 与真正序列化给 API 的 JSON 长度接近，作为估算足够。
            # 注意工具较多时这部分开销不可忽略（十几个工具可能上千 token）。
            tools_tokens = 0
            if tools:
                for tool in tools:
                    tools_tokens += self.count_tokens(str(tool))

            input_tokens += tools_tokens

            # 检查是否超出 token 上限
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # 抛出 TokenLimitExceeded，由上层智能体决定如何处理
                raise TokenLimitExceeded(error_message)

            # 校验提供的工具格式
            if tools:
                for tool in tools:
                    if not isinstance(tool, dict) or "type" not in tool:
                        raise ValueError("Each tool must be a dict with 'type' field")

            # 构建补全请求参数
            params = {
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "tool_choice": tool_choice,
                "timeout": timeout,
                **kwargs,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # 工具调用必须是非流式：需要一次性拿到完整的 tool_calls 结构化结果，
            # 流式增量无法可靠地拼装出参数 JSON。
            params["stream"] = False  # 工具请求始终使用非流式
            response: ChatCompletion = await self.client.chat.completions.create(
                **params
            )

            # 校验响应有效性：无 choices 或无 message 时返回 None，
            # 由调用方（智能体）自行判断"模型没有给出有效响应"。
            # 这里刻意不抛异常——避免因个别空响应触发整轮重试。
            if not response.choices or not response.choices[0].message:
                print(response)
                # raise ValueError("Invalid or empty response from LLM")
                return None

            # 更新 token 统计（非流式，usage 为服务端精确值）
            self.update_token_count(
                response.usage.prompt_tokens, response.usage.completion_tokens
            )

            # 返回完整消息对象（含 content 与 tool_calls），由调用方自行解析
            return response.choices[0].message

        except TokenLimitExceeded:
            # token 上限错误直接重新抛出，不做日志记录
            raise
        except ValueError as ve:
            logger.error(f"ask_tool 校验错误: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API 错误: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("认证失败，请检查 API 密钥。")
            elif isinstance(oe, RateLimitError):
                logger.error("已超出速率限制，可考虑增加重试次数。")
            elif isinstance(oe, APIError):
                logger.error(f"API 错误: {oe}")
            raise
        except Exception as e:
            logger.error(f"ask_tool 出现意外错误: {e}")
            raise
