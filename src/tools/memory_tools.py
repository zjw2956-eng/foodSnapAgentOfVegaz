"""记忆操作工具--Agent的"记性"，查询和写入用户数据"""
from pydantic_ai import RunContext
from src.tools.base import AgentDeps
from src.models.schemas import DishItem
from src.models.tables import UserProfile


async def memory_get_profile(ctx: RunContext[AgentDeps]) -> UserProfile:
    """获取用户画像，包括口味偏好、文案风格、常吃菜品等。

    首次使用的用户会返回一个默认画像。

    Args:
        ctx: 运行时上下文

    Returns:
        用户画像
    """
    # TODO: 后面接入 SQLite 真实查询
    # 先返回默认画像，保证 Agent 流程能跑通
    return UserProfile(
        user_id=ctx.deps.user_id,
    )


async def memory_query_history(
    ctx: RunContext[AgentDeps],
    query_text: str,
    limit: int = 5,
) -> list[dict]:
    """语义检索用户饮食历史，用自然语言模糊查找。

    例如："上周吃的那个辣辣的菜" → 匹配到"麻辣香锅"。

    Args:
        ctx: 运行时上下文
        query_text: 模糊查询文本
        limit: 返回条数上限

    Returns:
        匹配到的历史记录列表，每条含菜品名、时间、图片URL
    """
    # TODO: 后面接入 Qdrant + SQLite 双查
    # 1. query_text → 向量化 → Qdrant 语义检索
    # 2. 用返回的 ID 去 SQLite 取完整数据
    return []


async def memory_save_record(
        ctx: RunContext[AgentDeps],
        dishes: list[DishItem],
) -> bool:
    """
    保存本次识别记录到用户饮食历史。

    Args:
        ctx: 运行时上下文
        dishes: 本次识别出的菜品列表

    Returns:
        是否保存成功
    """
    # TODO: 后面双写 Qdrant + SQLite
    return True



async def memory_get_similar_articles_by_food(
    ctx: RunContext[AgentDeps],
    food_name: str,
    limit: int = 3,
) -> list[str]:
    """检索用户历史文案中与当前食物最相似的文案，
    作为 Few-shot 示例注入文案生成 prompt。

    Args:
        ctx: 运行时上下文
        food_name: 当前食物名称
        limit: 返回条数

    Returns:
        相似文案列表，按相关度排序
    """
    # TODO: 后面接入 Qdrant 向量检索
    return []
