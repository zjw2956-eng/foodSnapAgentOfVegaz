import os
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.deepseek import DeepSeekProvider
from dotenv import load_dotenv

load_dotenv()  # 自动找当前目录的 .env 文件，把里面的 KEY=VALUE加载为环境变量
# ======================LLM模型配置=================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "your-api-key-here")
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-api-key-here")

# 多模态模型（图片识别 + 文案生成）
vision_model = OpenAIChatModel(
    "qwen-vl-plus",
    provider=OpenAIProvider(
        base_url=DASHSCOPE_BASE_URL,
        api_key=DASHSCOPE_API_KEY,
    ),
)

# 纯文本模型（规划/反思 可用更便宜的模型）
text_model = OpenAIChatModel(
    "deepseek-chat",
    provider=DeepSeekProvider(api_key=DEEPSEEK_API_KEY),
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
