# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project identity

foodSnapAgent — Python AI Agent 微服务（PydanticAI + FastAPI + SQLModel + Qdrant）。
手机拍照识食物 → 生成朋友圈文案 → 推荐附近同款餐厅。Java 主服务通过 HTTP REST 调用。
基调：美食社交，"拍美食，分享快乐，吃遍天下"。健康只是锦上添花，不主动说教。

## Commands

```bash
# 环境管理
uv sync                    # 安装所有依赖（含 dev）
uv add <pkg>               # 添加生产依赖
uv add --dev <pkg>         # 添加开发依赖

# 运行
uv run python main.py      # 启动 FastAPI 服务

# 测试 & 检查
uv run pytest              # 运行所有测试
uv run ruff check .        # 代码 lint
uv run ruff format .       # 代码格式化
```

## Architecture

### 服务角色
Python Agent 是独立的、具备完整感知-规划-执行-再感知能力的 AI Agent。不是 API 封装壳。

### 双模型设计
- `vision_model`：qwen-vl-plus（阿里云百炼 MaaS），只做图片识别
- `text_model`：qwen-plus（阿里云百炼 MaaS），做 Agent 推理/规划/文案生成
- 分离原因：VL 模型贵，只有看图步骤才用；推理用便宜模型省 token
- 注意：qwen3.7-plus / deepseek-v4-flash 存在 tool_choice 兼容问题，已回退到上述模型

### Agent 范式：ReAct + Reflection
PydanticAI Agent 内置 ReAct 循环（LLM选工具→框架执行→返回结果→LLM决定下一步），不需要手写 while 循环。Reflection 关卡在 `article_tools.py` 中（生成文案→自评→不达标重写，最多3次）。

### 项目铁律（不可违反）
1. **所有 LLM 交互必须用 PydanticAI 原生 API**：`Agent.run()` / `agent.tool` / `tools=[...]`。严禁手写 OpenAI messages JSON。
2. **数据层用 SQLModel**：严禁原生 sqlite3 手写 SQL。表模型继承 `SQLModel, table=True`。
3. **多方案决策必须先列出对比问用户**：遇到有多个实现方案（数据库选型、API 调用方式等），必须先列方案让用户拍板，禁止自作主张选底层写法。

### 工具系统
工具分两类：
- **有 `RunContext[AgentDeps]` 的**（`memory_*`、`amap_*`）：注册进 `Agent(tools=[...])`，框架自动注入依赖，LLM 可自主调用
- **无 `RunContext` 的**（`image_analyze`、`article_generate_with_reflection`）：在 Agent 外部直接调用。图片识别必须用 vision_model，文案生成有内部 Reflection 循环，不适合作为单一 tool

所有工具分散在 `src/tools/` 各自模块中，在 `agent.py` 统一注册组装。不绑定到特定 Agent 实例，便于复用。

### 记忆系统
- 短期：PydanticAI 消息历史（框架自动管理）
- 工作：`AgentWorkState`（内存，单次请求）
- 长期：SQLite（SQLModel，精确存取）+ Qdrant（语义搜索）。Qdrant 负责召回"是哪个"，SQLite 取完整数据。Qdrant 不可用时降级到 SQLite LIKE 查询。

### 数据存储归属
SQLite/Qdrant 是 Agent 私有记忆，存 AI 特有的用户画像和饮食历史。Java 的 MySQL 存业务主数据（账号/帖子/点赞）。两边通过 `user_id` 对齐，Agent 不存账号密码。

### 降级策略
- 高德 API 超时/报错 → 返回空结果，Agent 继续走其他路径
- Qdrant 不可用 → 降级到 SQLite 关键词模糊查询
- 向量写入失败 → 静默跳过，SQLite 是主存储

### 关键文件
| 文件 | 角色 |
|---|---|
| `src/config.py` | 模型配置、环境变量、Agent 参数 |
| `src/agent.py` | Agent 核心（工具注册 + ReAct 编排） |
| `src/models/schemas.py` | API/Agent 数据模型（BaseModel），不入库 |
| `src/models/tables.py` | 数据库表模型（SQLModel），持久化 |
| `src/tools/base.py` | 依赖注入容器 AgentDeps + ToolRegistry |
| `src/tools/image_tools.py` | 图片识别（vision_model + ImageUrl） |
| `src/tools/article_tools.py` | 文案生成 + Reflection 重写 |
| `src/tools/map_tools.py` | 高德地图（异步 httpx + 降级） |
| `src/tools/memory_tools.py` | 记忆操作工具（stub，待连 manager） |
| `src/memory/manager.py` | SQLModel + Qdrant 统一封装 |
| `src/api/routes.py` | FastAPI 路由（Java 主服务调用的入口） |
| `memory/` | Project memory（非代码），Claude Code 跨会话加载 |

### .env 规范
- `.env` 包含真实 API Key，已加入 `.gitignore`，不能提交
- `.env.example` 是可提交的模板
- `config.py` 顶部 `load_dotenv(override=True)` 自动加载 `.env`（override 确保 .env 覆盖系统环境变量）
