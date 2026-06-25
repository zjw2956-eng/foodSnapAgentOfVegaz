"""文案生成工具 —— Agent 的"笔"，生成朋友圈文案 + Reflection 质量把关"""
from pydantic_ai import Agent
from src.config import text_model
from src.models.schemas import ArticleItem
from src.config import MAX_REFLECTION_RETRIES


async def article_generate(
    dish_name: str,
    ingredients: list[str],
    user_style: str = "吃货风",
    user_history: list[str] | None = None,
) -> ArticleItem:
    """生成朋友圈文案，带自评质量分。

    使用 PydanticAI Agent + result_type=ArticleItem，

    框架自动解析结构化输出（文案正文、风格、Hashtag、质量分、最佳发布时间）。

    Args:
        dish_name: 菜品名称
        ingredients: 食材列表
        user_style: 用户偏好的文案风格（吃货风/文艺风/幽默风/简约风）
        user_history: 用户历史文案（作为 Few-shot 参考，可选）

    Returns:
        生成的文案，含自评质量分
    """
    agent = Agent(
        text_model,
        result_type=ArticleItem,
        system_prompt=f"""你是美食朋友圈文案达人。为用户生成一条吸引人的朋友圈文案。

要求：
1. 风格严格遵循：{user_style}
2. 融入菜品名和至少一种食材的趣味点
3. 配 3-5 个相关 Hashtag
4. 推荐最佳发布时间
5. 给出 0-10 的自评质量分（10 为满分）
6. 基调：美食分享的快乐，不要说教、不要聊热量健康，除非用户主动设置

注意：这是美食社交场景，让文案有人情味、有记忆点，不要千篇一律。""",
    )
    prompt = f"""请为这道菜写朋友圈文案：
菜品：{dish_name}
食材：{', '.join(ingredients)}
"""

    if user_history:
        prompt += f"\n用户之前的优秀文案参考（学习风格但不要照抄）：\n"
        for i, h in enumerate(user_history[:3], 1):
            prompt += f"{i}. {h}\n"
    result = await agent.run(prompt)
    return result.data


async def article_reflect_and_rewrite(
    article: ArticleItem,
    dish_name: str,
    user_style: str,
) -> ArticleItem:
    """Reflection 关卡：评估文案质量，不达标则重写。

    这是 Agent"再感知"能力的体现——生成后不直接返回，
    而是让模型审视自己的输出，发现不足就改进。

    Args:
        article: 初始生成的文案
        dish_name: 菜品名（重写时参考）
        user_style: 目标风格

    Returns:
        评估通过的原文案，或重写后的新文案
    """
    # 质量达标（>=7分）直接返回，不浪费 token
    if article.quality_score >= 7.0:
        return article
    # 不达标，让模型反思问题并重写
    reflect_agent = Agent(
        text_model,
        result_type=ArticleItem,
        system_prompt=f"""你是文案质检员。下面是一条不够好的美食朋友圈文案（质量分 {article.quality_score}）。
请找出它的具体问题（太普通/没记忆点/风格不符/Hashtag不合适），
然后重写一条更好的，风格必须是：{user_style}。

重写要求：
1. 针对原文案的问题改进
2. 保留优点
3. 给出新的自评质量分
4. 仍然是美食社交基调，不说教""",
    )
    prompt = f"""原文案：
内容：{article.content}
Hashtag：{', '.join(article.hashtags)}
菜品：{dish_name}

请反思并重写。"""

    result = await reflect_agent.run(prompt)
    return result.data


async def article_generate_with_reflection(
    dish_name: str,
    ingredients: list[str],
    user_style: str = "吃货风",
    user_history: list[str] | None = None,
) -> ArticleItem:
    """完整流程：生成 → 反思 → （不达标）重写，最多重试
    MAX_REFLECTION_RETRIES 次。

    这是给 Agent 主流程调用的入口，封装了 Reflection 循环逻辑。
    面试亮点：体现"感知→执行→再感知"的 Agent 本质。

    Args:
        dish_name: 菜品名
        ingredients: 食材列表
        user_style: 用户风格
        user_history: 历史文案参考

    Returns:
        最终文案（达标或达到最大重试次数）
    """
    article = await article_generate(dish_name, ingredients, user_style, user_history)
    best = article
    for attempt in range(MAX_REFLECTION_RETRIES):
        if best.quality_score >= 7.0:
            break
        candidate = await article_reflect_and_rewrite(best, dish_name,user_style)
        if candidate.quality_score >= best.quality_score:
            best = candidate
    return best
