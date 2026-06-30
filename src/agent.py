"""Agent 核心 —— 工具注册 + ReAct 编排 + 请求处理入口

    这是整个项目的大脑：注册所有工具到 PydanticAI Agent，编排完整的感知-规划-执行-反思流程。
    PydanticAI Agent 内置 ReAct 循环（LLM选工具→框架执行→返回结果→LLM决定下一步），不需要手写 while。
    """

import time
import uuid
import httpx
from pydantic_ai import Agent
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.usage import UsageLimits
from src.config import (
    text_model, AMAP_API_KEY, QDRANT_HOST, QDRANT_PORT,
    MAX_REACT_STEPS,
)
from src.tools.base import AgentDeps
from src.tools.image_tools import image_analyze
from src.tools.memory_tools import (
    memory_get_profile, memory_query_history,
    memory_save_record, memory_get_similar_articles_by_food,
)
from src.tools.map_tools import (
    amap_search_nearby_restaurants, amap_reverse_geocode,
    amap_walking_direction,
)
from src.tools.article_tools import article_generate_with_reflection
from src.models.schemas import AnalysisRequest, AnalysisResponse, NutritionItem
from src.memory.manager import MemoryManager

# ==================== 主 Agent ====================
main_agent = Agent(
    text_model,
    deps_type=AgentDeps,
    tools=[
        memory_get_profile,
        memory_query_history,
        memory_save_record,
        memory_get_similar_articles_by_food,
        amap_search_nearby_restaurants,
        amap_reverse_geocode,
        amap_walking_direction,
    ],
    system_prompt="""你是美食社交管家，帮用户分析食物照片、记录美食足迹、推荐附近同款餐厅。

你的能力：
- 查看用户的口味偏好和饮食历史（memory 工具）
- 搜索附近餐厅（高德地图工具）
- 写朋友圈文案（article 工具，需要时你会被告知）

工作原则：
1. 先了解用户：查看用户画像和饮食历史，记住偏好
2. 根据用户意图行动：想发朋友圈就查历史文案找灵感，想找餐厅就搜附近
3. 美食社交基调：聊美食的快乐，不说教健康话题
4. 每次处理完记得保存饮食记录到记忆中

对话风格：热情、懂吃、像一个吃货朋友。""",
)

# ==================== 请求处理 ====================


async def process_request(
    request: AnalysisRequest,
    memory_manager: MemoryManager,
    http_client: httpx.AsyncClient,
) -> AnalysisResponse:
    """处理一次完整的 Agent 请求。

    流程：
    1. 图片识别（vision_model，固定第一步）
    2. Agent 自主决策（text_model + ReAct 循环）
    3. 文案生成（如果需要，含 Reflection）
    4. 组装响应

    Args:
        request: Java 主服务发来的请求
        memory_manager: 记忆管理器（SQLite + Qdrant）
        http_client: 共享的异步 HTTP 客户端

    Returns:
        结构化响应
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    # ===== 第1步：图片识别（必须，用贵的 vision_model） =====
    dishes = await image_analyze(request.image_url)
    # 只取高置信度的菜品（>= 0.6），避免"盘中误识别的餐巾纸"之类
    confident_dishes = [d for d in dishes if d.confidence >= 0.6]
    if not confident_dishes:
        # 全部低置信度时保留第一个（至少给用户一个结果）
        confident_dishes = dishes[:1] if dishes else []
    # ===== 第2步：构造依赖 =====
    deps = AgentDeps(
        user_id=request.user_id,
        image_url=request.image_url,
        intent=request.intent,
        location=request.location,
        amap_api_key=AMAP_API_KEY,
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
        http_client=http_client,
        memory_manager=memory_manager,
    )
    # ===== 第3步：Agent 自主推理 =====
    dish_names = [d.name for d in confident_dishes]
    ingredients_all = []
    for d in confident_dishes:
        ingredients_all.extend(d.ingredients)
    cuisine_types = list(
        set(d.cuisine_type for d in confident_dishes if d.cuisine_type))
    prompt = f"""用户上传了一张食物图片，已识别出以下菜品：{', '.join(dish_names)}

用户意图：{request.intent or '未指定（自由发挥）'}
用户位置：{request.location or '未知（无法搜索附近餐厅）'}

当前已知信息：
- 主要食材：{', '.join(ingredients_all[:10]) if ingredients_all else '待分析'}
- 涉及菜系：{', '.join(cuisine_types) if cuisine_types else '待分析'}

请根据用户意图行动：
- 如果用户想找附近餐厅且有位置 → 用高德地图搜索附近同款餐厅（关键词用"菜系+核心食材"，如"中式烧烤""川菜火锅"，不要直接用菜品名）
- 如果用户想发朋友圈 → 先查用户画像和文案偏好，然后告诉我"需要生成文案"
- 如果未指定意图 → 先识别菜品、查用户历史，然后问用户想干什么

完成后，总结你做了什么，包括：
- 用户画像的关键信息（如有）
- 查到的历史记录（如有）
- 搜到的餐厅（如有）
- 你对这顿饭的看法/建议（美食角度，不涉及健康话题）"""

    # PydanticAI 自动执行 ReAct 循环：LLM选工具→框架执行→返回结果→LLM决定下一步
    agent_result = await main_agent.run(
        prompt,
        deps=deps,
        usage_limits=UsageLimits(request_limit=MAX_REACT_STEPS),
    )
    # ===== 第4步：文案生成（如果用户要发朋友圈） =====
    article = None
    intent = request.intent or ""
    needs_article = (
        "article" in intent.lower() or
        "文案" in intent or
        "朋友圈" in intent or
        "发圈" in intent or
        "需要生成文案" in str(agent_result.output).lower()
    )
    if needs_article:
        # 从 Agent 结果中推断用户风格偏好
        user_style = "吃货风"  # 默认
        try:
            profile = memory_manager.get_profile(request.user_id)
            user_style = profile.article_style_preference
        except Exception:
            pass
        # 找历史文案做 few-shot 参考
        # ponytail: 文案历史存 SQLite 后，这里改为查 FoodRecord 关联的文案文本
        similar_articles: list[str] = []
        article = await article_generate_with_reflection(
            dish_name=dish_names[0] if dish_names else "美食",
            ingredients=ingredients_all[:5] if ingredients_all else ["未知"],
            user_style=user_style,
            user_history=similar_articles,
        )
    # ===== 第5步：保存记录到长期记忆 =====
    try:
        memory_manager.save_food_records(
            request.user_id, confident_dishes, request.image_url
        )
    except Exception:
        pass  # 保存失败不影响主流程
    # ===== 第6步：组装响应 =====
    processing_time_ms = int((time.time() - start_time) * 1000)

    # 从 Agent 工具调用结果中提取餐厅数据
    restaurants: list = []
    for msg in agent_result.all_messages():
        for part in msg.parts:
            if isinstance(part, ToolReturnPart) and part.tool_name == 'amap_search_nearby_restaurants':
                if isinstance(part.content, list):
                    restaurants = part.content
                    break
        if restaurants:
            break

    # RAG 菜品知识检索 —— 填充 nutrition
    nutrition = None
    if dish_names:
        knowledge = memory_manager.search_dish_knowledge(dish_names[0], limit=1)
        if knowledge:
            nutrition = NutritionItem(**knowledge[0])

    return AnalysisResponse(
        request_id=request_id,
        dishes=confident_dishes,
        nutrition=nutrition,
        article=article,
        restaurants=restaurants,
        processing_time_ms=processing_time_ms,
    )
