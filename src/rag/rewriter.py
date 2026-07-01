"""查询改写器 —— RAG 动态流程第①步

用户原始 query 与知识库文档表述存在语义 gap（如"烤鸡" vs 文档里的"烧鸡"），
改写扩展成多个同义变体，提升召回率。

面试考点：
- 为什么需要 query 改写：解决用户用词和文档用词不一致导致的召回遗漏
- Multi-Query 思想：一个 query 召回有限，多个变体召回后合并去重覆盖率更高
- 与 HyDE 的区别：Multi-Query 改 query，HyDE 改生成"假想文档"，本项目用 Multi-Query 更轻量

PydanticAI 用法对齐 article_tools.py：Agent(text_model, output_type=PydanticModel) + result.output
"""

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from src.config import text_model


class RewrittenQueries(BaseModel):
    """查询改写结果 —— 结构化输出，避免解析 LLM 自由文本出错"""
    queries: list[str] = Field(
        description="改写后的查询变体列表，含原始 query，最多5个，按相关度排序"
    )


# 改写专用 Agent（轻量，无需工具，纯文本生成）
_rewriter_agent = Agent(
    text_model,
    output_type=RewrittenQueries,
    system_prompt="""你是美食知识库的查询改写助手。
用户给你一个菜品名或食物描述，你要生成 3-5 个语义等价的搜索变体，用于向量检索召回。

变体生成方向：
1. 同义词/近义词（烤鸡→烧鸡、烤全鸡）
2. 俗称/学名（地三鲜→东北地三鲜）
3. 菜系归属（麻婆豆腐→川菜麻婆豆腐）
4. 英文/拼音（roast chicken）

要求：
- 第一个变体必须是原始输入（保底精确匹配）
- 变体之间语义相近但表述不同
- 不要编造不存在的菜名
- 只返回变体，不要解释""",
)


async def rewrite_query(original: str, max_variants: int = 5) -> list[str]:
    """查询改写入口。

    Args:
        original: 用户原始查询（通常是图片识别出的菜品名）
        max_variants: 最多返回的变体数（含原始 query）

    Returns:
        改写后的查询变体列表，第一个永远是原始 query

    降级策略：LLM 调用失败时只返回 [original]，保证后续检索不中断。
    """
    try:
        result = await _rewriter_agent.run(f"请改写这个查询：{original}")
        queries = result.output.queries
        # 兜底：LLM 没把原始 query 放第一个，强制补上
        if not queries or queries[0] != original:
            queries = [original] + [q for q in queries if q != original]
        return queries[:max_variants]
    except Exception:
        # 降级：改写失败不影响主流程，用原始 query 单路检索
        return [original]


# ==================== 自测 ====================
if __name__ == "__main__":
    import asyncio

    async def main():
        print("=== 查询改写自测 ===")
        for q in ["烤鸡", "麻婆豆腐", "汉堡"]:
            variants = await rewrite_query(q)
            print(f"\n原始: {q}")
            for i, v in enumerate(variants):
                print(f"  [{i}] {v}")

    asyncio.run(main())
