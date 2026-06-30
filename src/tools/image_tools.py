"""RunContext[AgentDeps]
PydanticAI 依赖注入的核心。ctx.deps
就是你传进去的 AgentDeps 实例，类型安全
"""
"""多模态消息格式
content 是数组，可以同时塞 image_url 和text。
这是 OpenAI兼容协议的通用格式，Qwen-VL-Plus 完全支持
"""
"""vision_model.request()
直接调用模型，不走 Agent 循环（避免 Agent调用工具时再触发 Agent）.
这是"工具内部的 LLM 调用"
"""

"""图片识别工具 —— Agent 的"眼睛"，分析食物图片"""
from pydantic_ai.messages import ImageUrl
from src.config import vision_model
from src.models.schemas import DishItem
from pydantic_ai import Agent


async def image_analyze(image_url: str) -> list[DishItem]:
    """分析食物图片，识别图中所有菜品。

    Args:
        image_url: 食物图片的 URL

    Returns:
        识别出的菜品列表
    """
    agent = Agent(
        vision_model,
        output_type=list[DishItem],
        system_prompt="你是专业美食识别助手，分析食物图片，识别所有菜品、食材和菜系。",
    )

    prompt = """请分析这张食物图片：
    1. 逐个列出图中所有菜品
    2. 每个菜给出菜名、食材(3-5种)、菜系、置信度(0~1)
    3. 图片质量差也要尽力识别，标注较低置信度
    """

    result = await agent.run([ImageUrl(url=image_url), prompt])
    return result.output


async def image_quality_check(image_url: str) -> dict:
    """检查食物图片的拍摄质量，给出改进建议。

    Args:
        image_url: 图片 URL

    Returns:
        {"is_good": bool, "suggestion": str}
    """
    agent = Agent(
        vision_model,
        system_prompt="评估食物图片拍摄质量（光线、构图、角度），给出改进建议。",
    )

    prompt = "这张食物图片适合发朋友圈吗？如果不行，怎么改进？"

    result = await agent.run([ImageUrl(url=image_url), prompt])
    return {"is_good": True, "suggestion": result.output}
