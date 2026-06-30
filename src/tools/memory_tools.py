"""记忆操作工具 —— Agent 的"记性"，查询和写入用户数据

所有工具通过 ctx.deps.memory_manager 访问 SQLite + Qdrant，
不再返回假数据。memory_manager 为 None 时降级返回安全默认值。
"""
from pydantic_ai import RunContext
from src.tools.base import AgentDeps
from src.models.schemas import DishItem
from src.models.tables import UserProfile


async def memory_get_profile(ctx: RunContext[AgentDeps]) -> UserProfile:
    """获取用户画像，包括口味偏好、文案风格、常吃菜品等。

    首次使用的用户会返回一个默认画像。
    """
    mgr = ctx.deps.memory_manager
    if mgr is None:
        return UserProfile(user_id=ctx.deps.user_id)
    return mgr.get_profile(ctx.deps.user_id)


async def memory_query_history(
    ctx: RunContext[AgentDeps],
    query_text: str,
    limit: int = 5,
) -> list[dict]:
    """语义检索用户饮食历史，用自然语言模糊查找。

    例如："上周吃的那个辣辣的菜" → 匹配到"麻辣香锅"。
    先走 Qdrant 向量检索，失败降级到 SQLite 关键词模糊查。
    """
    mgr = ctx.deps.memory_manager
    if mgr is None:
        return []
    records = mgr.query_history(ctx.deps.user_id, query_text, limit)
    return [
        {
            "dish_name": r.dish_name,
            "ingredients": r.ingredients,
            "cuisine_type": r.cuisine_type,
            "created_at": r.created_at.isoformat() if r.created_at else "",
            "image_url": r.image_url or "",
        }
        for r in records
    ]


async def memory_save_record(
    ctx: RunContext[AgentDeps],
    dishes: list[DishItem],
) -> bool:
    """保存本次识别记录到用户饮食历史（SQLite 双写）。"""
    mgr = ctx.deps.memory_manager
    if mgr is None:
        return False
    try:
        mgr.save_food_records(ctx.deps.user_id, dishes, ctx.deps.image_url)
        return True
    except Exception:
        return False


async def memory_get_similar_articles_by_food(
    ctx: RunContext[AgentDeps],
    food_name: str,
    limit: int = 3,
) -> list[str]:
    """检索用户历史饮食中与当前食物最相似的记录，
    作为 Few-shot 上下文注入文案生成 prompt。

    当前通过饮食记录名称匹配（Qdrant 语义检索），
    后续可扩展为检索已保存的文案文本。
    """
    mgr = ctx.deps.memory_manager
    if mgr is None:
        return []
    records = mgr.query_history(ctx.deps.user_id, food_name, limit)
    return [
        f"吃过「{r.dish_name}」（{r.cuisine_type}），食材：{r.ingredients}"
        for r in records
    ]
