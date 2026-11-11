# 读者已经发现但尚未解决的问题
1.  `def is_stuck(self) -> bool:`               
    读者认为当前判断逻辑不好，完全重复消息几乎很难出现，几乎不会触发True。
    最后一条消息是tool，比较对象就错了
    LLM 走工具调用路线时`content` 经常是空字符串`""` ， L181`if not last_message.content: return False` 直接返回 False。而“卡死”恰恰最常发生在反复调用同一个工具的场景。
    `handle_stuck_state()` 会改写`next_step_prompt` ，后续消息内容必然变化，精确匹配就更不可能了。
    真正的“卡死”长什么样:
    Agent 陷入循环的典型特征不是“说一样的话”，而是**“做一样的事”**：反复调用同一个工具、传同样的参数。检测应该看 tool_calls 签名，而不是文本内容。

2. `class Memory(BaseModel):`记忆管理，只保留最近的N条消息

3. `logger.info(f"🔧 工具参数：{tool_calls[0].function.arguments}")`日志偷懒只记录第一个工具的工具参数 # 已解决

4. token超限依然重试  # 已解决

5. 工具返回结果不可控，截断结果，防止超出最大观察长度，但是截断损失未考虑