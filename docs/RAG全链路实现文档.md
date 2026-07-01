# RAG 全链路实现文档

> foodSnapAgent 项目 · 基于 PydanticAI v1.107.0 + Qdrant + 阿里云百炼 Embedding
> 本文档记录 RAG 链路各阶段的实现方式，区分「小项目做法」与「企业级 Agent 做法」。

---

## 总览：RAG 全链路

RAG 分两条流程线：

```
【静态流程 / 离线入库】启动时跑一次
  原始数据 → ① 文档切割 → ② 向量化 → ③ 入库索引

【动态流程 / 在线检索】每次请求跑
  用户 query → ④ 查询改写 → ⑤ 多路召回 → ⑥ RRF 融合 → ⑦ Rerank 重排 → ⑧ 返回 Top-K
```

### 当前进度

| 阶段 | 静态/动态 | 状态 | 代码位置 |
|------|-----------|------|----------|
| ① 文档切割 | 静态 | ✅ 已写（未接入主流程） | `src/rag/chunker.py` |
| ② 向量化 | 静态 | ✅ 已写 | `src/memory/manager.py:_embed()` |
| ③ 入库索引 | 静态 | ✅ 已写 | `src/memory/manager.py:_seed_dish_knowledge()` |
| ④ 查询改写 | 动态 | ❌ 未写 | 待写 `src/rag/rewriter.py` |
| ⑤ 多路召回 | 动态 | ⚠️ 仅单路向量 | `src/memory/manager.py:search_dish_knowledge()` |
| ⑥ RRF 融合 | 动态 | ❌ 未写 | 待写 `src/rag/retriever.py` |
| ⑦ Rerank 重排 | 动态 | ❌ 未写 | 待写 `src/rag/retriever.py` |
| ⑧ 返回 Top-K | 动态 | ✅ 已有 | `search_dish_knowledge()` 返回 |

---

## 静态流程详解

### ① 文档切割（Chunking）

**目的**：把长文档切成可被 embedding 模型处理、语义完整的片段。

#### 小项目做法

当前数据是「菜名 + 营养 JSON」短文本，单条天然就是最小语义单元，**不需要切割**，直接整条入库。

`src/rag/chunker.py` 提供两种切割策略供面试展示，但未接入主流程：

| 策略 | 原理 | 适用 |
|------|------|------|
| `sliding_window` | 固定字符窗口 + overlap 滑动 | 通用入门 |
| `recursive` | 按分隔符优先级（段落→句子→词）递归切 | LangChain 默认方案，断在自然标点 |

**核心参数**：
- `chunk_size`：片段大小，参考 embedding 模型 max_tokens 减安全余量
- `chunk_overlap`：相邻片段重叠量，防止关键信息落在切割边界被截断

#### 企业级 Agent 做法

| 方式 | 原理 | 适用场景 |
|------|------|----------|
| **RecursiveCharacterTextSplitter** | 按 `\n\n`→`\n`→`。`→` ` 优先级逐级尝试 | 通用文档，LangChain 默认 |
| **Semantic Chunking** | 相邻句子分别 embedding，相似度骤降处切分 | 对话、长文，靠语义断崖判断 |
| **Markdown/Header Splitter** | 按 `#` 标题层级切，标题作为 metadata | 技术文档、Wiki |
| **Agentic Chunking** | LLM 读全文自行决定切割点 | 质量最高，每次切都调 LLM |
| **Late Chunking（Jina）** | 先整篇 embedding 再在向量空间切 | 上下文不丢失，SOTA |
| **Contextual Retrieval（Anthropic）** | 每段前置拼接文档摘要再 embedding | 切割丢失的上下文靠元信息补回 |

#### 当前代码关键实现

```python
# src/rag/chunker.py
@dataclass
class ChunkConfig:
    chunk_size: int = 128
    chunk_overlap: int = 32
    strategy: str = "sliding_window"

class TextChunker:
    def chunk(self, text: str) -> list[str]: ...
    def _sliding_window_chunk(self, text) -> list[str]: ...
    def _recursive_chunk(self, text) -> list[str]: ...
    def chunk_by_tokens(self, text, tokenizer, max_tokens=512) -> list[str]: ...
```

自测：`uv run python src/rag/chunker.py`

---

### ② 向量化（Embedding）

**目的**：把文本转成向量，供 Qdrant 做语义相似度检索。

#### 小项目做法

调用阿里云百炼 OpenAI 兼容接口，复用 LLM 的 `DASHSCOPE_BASE_URL`：

```python
# src/memory/manager.py:_embed()
POST {DASHSCOPE_BASE_URL}/embeddings
Body: {"model": "text-embedding-v4", "input": text, "dimensions": 1024}
返回: data["data"][0]["embedding"]  # 1024 维向量
```

降级策略：embedding 失败返回零向量，不影响主流程。

#### 企业级 Agent 做法

| 方面 | 小项目 | 企业级 |
|------|--------|--------|
| 模型 | 云端 API（DashScope） | 自部署 BGE-M3 / bge-large-zh，省调用费 |
| 批量化 | 单条 `input` | 批量 `input: [...]` 一次上百条，降延迟 |
| 缓存 | 无 | 相同文本缓存向量（Redis），避免重复 embed |
| 维度 | 固定 1024 | 按需 64-2048，权衡存储和精度 |
| 多模态 | 纯文本 | 图文联合 embedding（如菜品图+文本对齐） |

---

### ③ 入库索引（Indexing）

**目的**：把向量 + payload 存入 Qdrant collection，供后续检索。

#### 小项目做法

```python
# src/memory/manager.py:_seed_dish_knowledge()
# 启动时幂等创建 dish_knowledge collection，写入 12 条种子菜品数据
# 守卫：if "dish_knowledge" not in names → 已存在则跳过，不重复向量化
points = [PointStruct(id=i, vector=embed(dish_name), payload={**dish, "text": dish_name})]
self._qdrant.upsert(collection_name="dish_knowledge", points=points)
```

payload 结构：`{dish_name, calories, protein, fat, carbs, fun_fact, text}`

#### 企业级 Agent 做法

| 方面 | 小项目 | 企业级 |
|------|--------|--------|
| 索引结构 | 纯 HNSW 向量索引 | HNSW + BM25 倒排 + payload 索引 |
| 增量更新 | 全量 upsert | 增量 upsert + 版本号 + 软删除 |
| 分布式 | 单节点 Qdrant | Qdrant 集群分片，按 collection 分片 |
| 元数据 | 扁平 payload | payload 建索引（菜系/地区/价格段），支持精确过滤 |
| 数据来源 | 手写 12 条 | ETL 管道：爬虫/数据库 → 清洗 → 切割 → embed → 入库 |

---

## 动态流程详解

### ④ 查询改写（Query Rewrite）—— 未写

**目的**：用户原始 query 与文档表述存在语义 gap（"烤鸡" vs 文档里的"烧鸡"），改写扩展提升召回率。

#### 小项目做法（建议）

直接用 LLM 生成 3-5 个同义变体：

```python
# 待写 src/rag/rewriter.py
async def rewrite(query: str) -> list[str]:
    # "烤鸡" → ["烤鸡", "烧鸡", "中式烤鸡", "roast chicken"]
    prompt = f"为菜品名生成3-5个语义等价变体：{query}"
    return [query] + llm.run(prompt).split('\n')
```

#### 企业级 Agent 做法

| 方式 | 原理 |
|------|------|
| **Multi-Query** | LLM 生成多个查询变体，各自召回后合并去重 |
| **HyDE** | LLM 先编一个"假想答案文档"，用假文档去检索（query 和 doc 同语义空间） |
| **Step-Back Prompting** | LLM 把具体问题抽象成更宽泛的查询 |
| **Query Decomposition** | 复杂问题拆成多个子问题分别检索 |

---

### ⑤ 多路召回（Multi-Recall）—— 当前仅单路

**目的**：单一检索方式有盲区，多路互补提升召回率。

#### 小项目做法

当前 `search_dish_knowledge()` 是**单路向量检索**：query → embed → Qdrant top-K。12 条数据够用。

#### 企业级 Agent 做法

| 路 | 方式 | 强项 |
|----|------|------|
| Dense | embedding 语义召回 | 容忍同义词、语义相近 |
| Sparse | BM25 / TF-IDF 关键词召回 | 专有名词、菜名精确命中 |
| Knowledge Graph | 图谱实体关系遍历 | 多跳推理（"和烤鸡同菜系更辣的菜"） |

```python
# 企业级伪代码
dense = qdrant.query_points(query=embed(q), limit=10)      # 语义
sparse = bm25.search(q, limit=10)                           # 关键词
```

---

### ⑥ RRF 融合（Reciprocal Rank Fusion）—— 未写

**目的**：多路召回结果合并排序，无需调参。

#### 公式

```
score(doc) = Σ 1 / (k + rank_i(doc))    # k=60 经典常数
```

每路检索中文档的排名越靠前，贡献分越高；多路都命中的文档分数叠加。

#### 实现（小项目和企业级通用）

```python
def rrf_fusion(dense_results, sparse_results, k=60, limit=10):
    scores = {}
    for rank, r in enumerate(dense_results):
        scores[r.id] = scores.get(r.id, 0) + 1 / (k + rank + 1)
    for rank, r in enumerate(sparse_results):
        scores[r.id] = scores.get(r.id, 0) + 1 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)[:limit]
```

---

### ⑦ Rerank 重排—— 未写

**目的**：粗排（召回）后精排，提升最终 Top-K 精度。召回用轻量模型追求覆盖率，Rerank 用重模型追求精度。

#### 小项目做法（建议）

用 LLM 当 Reranker，零额外依赖：

```python
async def llm_rerank(query, candidates, top_k=3):
    # 让 LLM 判断每个候选与 query 的相关度，返回排序序号
    prompt = f"根据与「{query}」的相关度，对以下候选重排，返回前{top_k}个序号：\n{candidates}"
    return parse(llm.run(prompt))
```

#### 企业级 Agent 做法

| 方式 | 原理 | 成本 |
|------|------|------|
| **Cross-Encoder**（bge-reranker） | query 和 doc 拼接输入，输出相关度分 | 自部署，快 |
| **LLM Rerank** | LLM 直接排序 | 慢但零依赖，面试友好 |
| **Cohere Rerank API** | 云端精排模型 | 按 API 计费 |

---

### ⑧ 返回 Top-K 并注入 Agent

当前 `agent.py` 调用 `search_dish_knowledge(dish_names[0], limit=1)`，取 Top-1 填充 `nutrition` 字段。

企业级会把 Top-K 文本拼进 Agent 的 system prompt 作为上下文，让 LLM 基于检索结果生成回答（而非直接取第一条）。

---

## 文件结构规划

```
src/rag/
├── chunker.py       ✅ 文档切割（已写）
├── rewriter.py      ⏳ 查询改写（待写）
└── retriever.py     ⏳ 多路召回 + RRF 融合 + Rerank（待写）
```

`MemoryManager` 保持数据持久层职责，`rag/` 负责检索逻辑层，单一职责。

---

## 后续写文档交接提示词

> 继续编写 `docs/RAG全链路实现文档.md`。
>
> 当前进度：静态流程（①文档切割 / ②向量化 / ③入库索引）已写完文档，动态流程尚未开始实现代码。
>
> 下一步任务：
> 1. 实现 `src/rag/rewriter.py`（动态流程第④步：查询改写），实现后更新本文档「④查询改写」章节的「当前代码关键实现」小节，把状态从「❌ 未写」改为「✅ 已写」。
> 2. 实现 `src/rag/retriever.py`（动态流程第⑤⑥⑦步：多路召回 + RRF 融合 + Rerank），替换 `src/memory/manager.py:search_dish_knowledge()` 的单路检索，并把 `agent.py` 中的 RAG 调用改为完整链路（rewrite → multi-recall → fusion → rerank）。实现后更新对应章节。
> 3. 每完成一个阶段，更新文档顶部「当前进度」表格的状态列和代码位置列。
>
> 文档规范：
> - 每个阶段必须包含「小项目做法」和「企业级 Agent 做法」两个小节
> - 代码关键实现用 `python` 代码块，标注文件路径
> - 保持与现有文档一致的 Markdown 风格
> - 不要删除已有内容，只追加和更新状态
