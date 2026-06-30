import os
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
# TODO: 后续可能切换 DeepSeek，暂保留引用
# from pydantic_ai.providers.deepseek import DeepSeekProvider
from dotenv import load_dotenv

load_dotenv(override=True)  # override=True：.env 的值覆盖系统环境变量


# ======================LLM模型配置=================
DASHSCOPE_BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "your-base-url-here")
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "your-api-key-here")


# TODO: 后续可能切换 DeepSeek 官方 API，暂保留
# DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")

# 多模态模型（图片识别 + 文案生成）
# ponytail: qwen3.7-plus 思考模式不支持 tool_choice=required，回退到 qwen-vl-plus
vision_model = OpenAIChatModel(
    "qwen-vl-plus",
    provider=OpenAIProvider(
        base_url=DASHSCOPE_BASE_URL,
        api_key=DASHSCOPE_API_KEY,
    ),
)

# 纯文本模型（规划/反思 用阿里云百炼托管的 DeepSeek-V4-Flash）
# TODO: 后续可切换为 DeepSeek 官方直连：
# text_model = OpenAIChatModel(
#     "deepseek-chat",
#     provider=DeepSeekProvider(api_key=DEEPSEEK_API_KEY),
# )
# ponytail: deepseek-v4 同样不支持 tool_choice=required，改用 qwen-plus
text_model = OpenAIChatModel(
    "qwen-plus",
    provider=OpenAIProvider(
        base_url=DASHSCOPE_BASE_URL,
        api_key=DASHSCOPE_API_KEY,
    ),
)

# =======================高德地图配置=================
AMAP_API_KEY = os.getenv("AMAP_API_KEY", "your-amap-key-here")

# =====================Qdrant配置===================
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# ==================== SQLite 配置 ====================

SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/food_snap.db")

# ==================== Agent 参数 ====================
MAX_REACT_STEPS = int(os.getenv("MAX_REACT_STEPS", "10"))
MAX_REFLECTION_RETRIES = int(os.getenv("MAX_REFLECTION_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "30"))
