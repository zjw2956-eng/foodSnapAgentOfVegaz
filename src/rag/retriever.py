"""检索器 —— RAG 动态流程第②③④步

完整链路：多路召回 → RRF 融合 → LLM Rerank

- 多路召回：Dense（embedding 语义）+ Sparse（关键词精确）互补
  · Dense 容忍同义词（"烤鸡"→"烧鸡"），但会漂移到近义词（"烤鸡"→"烧烤"）
  · Sparse 靠 payload 关键词精确命中，保证菜名 100% 召回
- RRF 融合：多路结果合并排序，无需调参，k=60 是经典常数
  · score(doc) = Σ 1/(k + rank_i(doc))，排名越靠前贡献越高，多路命中叠加
- LLM Rerank：召回后精排，用 LLM 判断 query 与候选的真实相关度
  · 解决 embedding 漂移（"烤鸡"误召回"烧烤"会被 Rerank 排下去）
  · 比纯向量相似度更准，因为 LLM 能理解语义而非只算距离

复用 MemoryManager 的 _embed() 和 Qdrant 连接，不重复造轮子。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from src.config import text_model
from src.memory.manager import MemoryManager


class RerankResult(BaseModel):
    """Rerank 结果 —— 结构化输出，返回排序后的候选序号"""

    indices: list[int] = Field(
        description="按相关度从高到低排序的候选序号列表（0-based），只返回最相关的"
    )


# Rerank 专用 Agent
_rerank_agent = Agent(
    text_model,
    output_type=RerankResult,
    system_prompt="""你是菜品知识库的精排器。
给你一个查询和若干候选菜品知识，请判断每个候选与查询的真实相关度，返回最相关的候选序号。

判断标准：
- 菜名是否指同一种食物（"烤鸡"和"烧鸡"相关，"烤鸡"和"烧烤"不太相关）
- 营养信息是否适用于该查询
- 只返回最相关的 1-3 个，按相关度降序

只返回序号数组，不要解释。""",
)


class DishKnowledgeRetriever:
    """菜品知识检索器 —— 多路召回 + RRF + Rerank

    依赖 MemoryManager 提供 embedding 和 Qdrant 连接。
    """

    def __init__(self, memory_manager: MemoryManager, rrf_k: int = 60):
        self.mgr = memory_manager
        self.rrf_k = rrf_k  # RRF 经典常数

    # ==================== 完整链路入口 ====================

    async def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """完整 RAG 检索链路。

        流程：多路召回 → RRF 融合 → LLM Rerank

        Args:
            query: 查询文本（通常是图片识别出的菜名）
            top_k: 最终返回的条数

        Returns:
            排序后的知识条目列表（payload dict）
        """
        # 第②③步：多路召回 + RRF 融合（粗排，取较多候选给 Rerank 筛）
        candidates = self.hybrid_search(query, limit=10)
        if not candidates:
            return []

        # 第④步：LLM Rerank（精排，从候选里挑 top_k）
        if len(candidates) <= top_k:
            return candidates
        return await self.rerank(query, candidates, top_k)

    # ==================== 第②步：多路召回 + 第③步：RRF 融合 ====================

    def hybrid_search(self, query: str, limit: int = 10) -> list[dict]:
        """多路召回 + RRF 融合。

        第1路 Dense：query → embedding → Qdrant 向量检索（语义召回）
        第2路 Sparse：query 关键词 → Qdrant payload 过滤（精确召回）

        两路结果用 RRF 融合排序。
        """
        # 第1路：Dense 向量语义召回
        dense = self._dense_search(query, limit)

        # 第2路：Sparse 关键词召回（payload dish_name 模糊匹配）
        sparse = self._sparse_search(query, limit)

        if not dense and not sparse:
            return []
        if not dense:
            return [r["payload"] for r in sparse]
        if not sparse:
            return [r["payload"] for r in dense]

        # 第③步：RRF 融合两路结果
        return self._rrf_fusion(dense, sparse, limit)

    def _dense_search(self, query: str, limit: int) -> list[dict]:
        """Dense 路：embedding 语义召回"""
        try:
            client = self.mgr._ensure_qdrant()
            vector = self.mgr._embed(query)
            results = client.query_points(
                collection_name="dish_knowledge",
                query=vector,
                limit=limit,
            ).points
            return [
                {"id": r.id, "payload": r.payload, "score": r.score} for r in results
            ]
        except Exception:
            return []

    def _sparse_search(self, query: str, limit: int) -> list[dict]:
        """Sparse 路：关键词精确召回

        用 Qdrant payload filter 强制 dish_name 包含 query 关键词。
        保证菜名精确命中，弥补 Dense 路的语义漂移。
        """
        try:
            client = self.mgr._ensure_qdrant()
            # scroll 全量后内存过滤 dish_name（小数据量够用，避免复杂 filter 构造）
            all_points = client.scroll(
                collection_name="dish_knowledge",
                limit=100,
                with_payload=True,
                with_vectors=False,
            )[0]
            matched = []
            for p in all_points:
                dish_name = (p.payload or {}).get("dish_name", "")
                if query in dish_name or dish_name in query:
                    matched.append({"id": p.id, "payload": p.payload, "score": 1.0})
            return matched[:limit]
        except Exception:
            return []

    def _rrf_fusion(
        self, dense: list[dict], sparse: list[dict], limit: int
    ) -> list[dict]:
        """RRF 融合：score(doc) = Σ 1/(k + rank_i(doc))"""
        scores: dict[int, float] = {}
        payloads: dict[int, dict] = {}

        # Dense 路按 score 已降序，rank 从 0 开始
        for rank, r in enumerate(dense):
            scores[r["id"]] = scores.get(r["id"], 0) + 1 / (self.rrf_k + rank + 1)
            payloads[r["id"]] = r["payload"]

        # Sparse 路
        for rank, r in enumerate(sparse):
            scores[r["id"]] = scores.get(r["id"], 0) + 1 / (self.rrf_k + rank + 1)
            payloads[r["id"]] = r["payload"]

        # 按 RRF 分数降序
        sorted_ids = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
        return [payloads[id_] for id_ in sorted_ids[:limit] if id_ in payloads]

    # ==================== 第④步：LLM Rerank ====================

    async def rerank(
        self, query: str, candidates: list[dict], top_k: int = 3
    ) -> list[dict]:
        """LLM 精排：从粗排候选里挑最相关的 top_k 条。

        解决 embedding 漂移：如"烤鸡"误召回"烧烤"，Rerank 能识别二者不同并排下去。
        降级：LLM 解析失败时返回候选前 top_k 条（即 RRF 结果）。
        """
        # 构造候选列表文本
        items_text = "\n".join(
            f"[{i}] {c.get('dish_name', '')}：{c.get('fun_fact', '')}"
            for i, c in enumerate(candidates)
        )
        prompt = f"""查询：{query}

候选菜品知识：
{items_text}

请返回与查询最相关的候选序号。"""

        try:
            result = await _rerank_agent.run(prompt)
            indices = result.output.indices
            return [candidates[i] for i in indices if 0 <= i < len(candidates)][:top_k]
        except Exception:
            # 降级：Rerank 失败用 RRF 粗排结果
            return candidates[:top_k]


# ==================== 自测 ====================
if __name__ == "__main__":
    import asyncio

    from src.config import QDRANT_HOST, QDRANT_PORT, SQLITE_DB_PATH

    async def main():
        mgr = MemoryManager(SQLITE_DB_PATH, QDRANT_HOST, QDRANT_PORT)
        retriever = DishKnowledgeRetriever(mgr)

        for q in ["烤鸡", "麻辣豆腐", "汉堡"]:
            print(f"\n=== 查询: {q} ===")
            # 单路 Dense 对比
            dense = retriever._dense_search(q, limit=3)
            print("Dense 召回:", [r["payload"].get("dish_name") for r in dense])
            # 完整链路（含 RRF + Rerank）
            result = await retriever.retrieve(q, top_k=1)
            print("RAG 最终:", [r.get("dish_name") for r in result])

    asyncio.run(main())
