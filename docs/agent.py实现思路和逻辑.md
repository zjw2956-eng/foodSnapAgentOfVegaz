 现状：我们有 11 个工具函数，分成两类                                                                                                                                 
     
  无 RunContext（独立函数，直接调用）：
    image_analyze(image_url)             → 识别图片
    image_quality_check(image_url)       → 评估拍摄质量
    article_generate_with_reflection(...)  → 生成文案+反思

  有 RunContext[AgentDeps]（Agent 工具，框架注入依赖）：
    memory_get_profile(ctx)              → 查用户画像
    memory_query_history(ctx, query_text) → 查饮食历史
    memory_save_record(ctx, dishes)      → 保存记录
    memory_get_similar_articles_by_food(ctx, food_name) → 找历史文案
    amap_search_nearby_restaurants(ctx, keyword, location)
    amap_reverse_geocode(ctx, location)
    amap_walking_direction(ctx, origin, destination)

  ---
  agent.py 的核心设计

  一句话：image_analyze 先识别图片，然后把结果和工具交给 PydanticAI Agent，让它自己决定怎么做。PydanticAI 的 Agent 内置了 ReAct 循环——LLM 选工具 → 框架执行 → 返回结果 
  → LLM 决定下一步——你不需要手动写循环。

  请求进来
    │
    ▼
  ① image_analyze(image_url)          ← 固定第一步，多模态模型分析图片
    │                                    返回 list[DishItem]
    ▼
  ② 把识别结果 + 用户意图写成 prompt
    │
    ▼
  ③ orchestrator_agent.run(prompt)    ← Agent 开始工作，PydanticAI 自动 ReAct：
    │                                    第1轮：LLM 决定查用户记忆
    │                                    第2轮：LLM 决定搜附近餐厅
    │                                    第3轮：LLM 觉得够了，生成回应
    ▼
  ④ 根据 intent 决定是否生成文案       ← article_generate_with_reflection()
    │
    ▼
  ⑤ 组装 AnalysisResponse 返回

  ---
  PydanticAI Agent 的 ReAct 是怎么自动发生的

  当你注册了 tools 并调用 agent.run(prompt) 时，框架内部自动做了：

  1. 把 prompt + tools 描述发给 LLM
  2. LLM 返回 "我要调用 memory_get_profile(user_id=xxx)"
  3. PydanticAI 自动调用这个函数，拿到 UserProfile
  4. 把函数返回值作为 "Observation" 塞回消息历史
  5. LLM 看到 Observation，决定下一步：继续调工具，还是输出最终结果
  6. 重复 2-5，直到 LLM 说 "够了" 或达到 max_steps

  你没有写一行 while、没有写一步 if tool == "xxx"，PydanticAI 全自动。面试官问"ReAct 循环怎么实现的"，你答"PydanticAI Agent 内置了 tool-calling loop，框架自动管理     
  Thought-Action-Observation 循环，我们只需要定义工具和 system_prompt。"

  ---
  工具怎么注册

  main_agent = Agent(
      text_model,                    # qwen-plus，只管推理
      deps_type=AgentDeps,           # 工具函数通过 ctx.deps 拿到的东西
      tools=[                        # 框架自动做 ReAct 循环
          memory_get_profile,
          memory_query_history,
          memory_save_record,
          memory_get_similar_articles_by_food,
          amap_search_nearby_restaurants,
          amap_reverse_geocode,
          amap_walking_direction,
      ],
      system_prompt="你是美食管家...",
  )

  有 RunContext[AgentDeps] 的才能放进 tools 列表。没有 RunContext 的（image_analyze、article_generate_with_reflection）在 Agent
  外部直接调用——因为图片识别必须用多模态模型，文案生成是确定性流程（不是 Agent 自己决定"要不要写文案"）。

  ---
  agent.py 的代码结构

  # 1. 创建 Agent（带工具 + system_prompt）
  # 2. process_request(request, manager)  →  主函数
  #    a. image_analyze()            # 多模态识别
  #    b. 构造 prompt（拼入识别结果）
  #    c. agent.run(prompt, deps=)   # Agent 自主决策
  #    d. 如果需要文案 → article_generate_with_reflection()
  #    e. 返回 AnalysisResponse

  ---
  为什么 image_analyze 不放进 Agent tools？

  两个模型不同——vision_model（qwen-vl-plus）看图片，text_model（qwen-plus）做推理。Agent 用便宜的 text_model 做决策，只有必须看图的步骤才调用贵的 vision_model。省      
  token。

  ---
  为什么 article_generate_with_reflection 不放进 Agent tools？

  文案生成包含了 Reflection 循环（生成→自评→重写），是一个有内部状态和循环的复杂工具。让 Agent 在一次 tool call 里完成整个"生成+反思"流程，比让 Agent
  自己决定"要不要反思"更可控、更可靠。Agent 只管"要不要生成文案"，生成的细节由 article_generate_with_reflection 内部搞定。

  ---
  总结：agent.py 就做四件事

  1. 定义 main_agent（text_model + 7个工具）
  2. 调用 image_analyze（vision_model）
  3. 调用 main_agent.run(prompt) → 框架自动 ReAct
  4. 调用 article_generate_with_reflection（如果需要）

  PydanticAI 的 Agent 帮你扛了最重的活——ReAct 循环、工具调度、消息历史管理。你只负责定义工具、写对 system_prompt。

  ---