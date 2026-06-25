from pydantic import BaseModel, Field
from typing import Optional

# =====================API层：请求/响应============


class AnalysisRequest(BaseModel):
    """图片分析请求（Java 主服务 → Python Agent）"""
    image_url: str = Field(description="食物图片的URL")
    user_id: str = Field(description="用户唯一标识")
    intent: Optional[str] = Field(
        default=None,
        description="用户意图：identify(仅识别) / article(要发朋友圈) / \
            recommend(找附近同款)"
    )


class DishItem(BaseModel):
    """单个菜品识别结果"""
    name: str = Field(description="菜品名称")
    confidence: float = Field(description="识别置信度0-1", ge=0, le=1)
    ingredients: list[str] = Field(description="识别出的食材列表")
    cuisine_type: str = Field(default="",description="菜系：川菜/粤菜/日料/西餐/...")

class NutritionItem(BaseModel):
    """营养信息 —— 仅供参考，不是核心功能"""
    calories: int = Field(description="热量（大卡）")
    protein: float = Field(description="蛋白质（克）")
    fat: float = Field(description="脂肪（克）")
    carbs: float = Field(description="碳水化合物（克）")
    fun_fact: str = Field(
        default="",
        description="趣味小知识，如'藕片高纤维，土豆是优质碳水'——轻松科普，不说教"
    )


class ArticleItem(BaseModel):
    """生成的朋友圈文案"""
    content: str = Field(description="文案正文")
    style: str = Field(description="风格：吃货风(热情)/文艺风(精致)/幽默风(有趣)/简约风(高级)/INS风")
    hashtags: list[str] = Field(description="推荐的话题标签")
    quality_score: float = Field(description="自评质量分 0-10", ge=0, le=10)
    best_time_to_post: str = Field(
        default="",
        description="推荐发布时间，如'中午12点 午饭时间'"
    )


class RestaurantItem(BaseModel):
    """附近餐厅推荐—— 核心功能"""
    name: str = Field(description="餐厅名称")
    address: str = Field(description="地址")
    distance: int = Field(description="距离（米）",ge=0)
    rating: float = Field(description="评分", ge=0, le=5)
    price_level: str = Field(description="人均消费，如'人均85元'")
    has_similar_dish: bool = Field(description="是否有相似菜品")
    recommendation_reason: str = Field(description="推荐理由")
    must_try_dishes: list[str] = Field(
        default_factory=list,
        description="该餐厅的招牌菜推荐"
    )


class AnalysisResponse(BaseModel):
    """分析结果（Python Agent → Java 主服务）"""
    request_id: str = Field(description="请求唯一标识")
    dishes: list[DishItem] = Field(description="识别出的菜品列表")
    nutrition: Optional[NutritionItem] = Field(
        default=None,
        description="营养参考（可选）"
    )
    article: Optional[ArticleItem] = Field(
        default=None,
        description="生成的文案"
    )
    restaurants: list[RestaurantItem] = Field(
        default_factory=list,
        description="附近同款餐厅"
    )
    processing_time_ms: int = Field(description="处理耗时（毫秒）")


class HealthResponse(BaseModel):
    """健康检查"""
    status: str = Field(default="ok")
    version: str = Field(default="0.1.0")


# ==================Agent层：内部数据结构==============
class AgentWorkState(BaseModel):
    """Agent工作状态--工作记忆的核心"""
    current_step: int = Field(
        default=0,
        description="当前ReAct步骤数")
    max_steps: int = Field(default=10)
    task_description: str = Field(
        default="",
        description="当前任务描述"
    )
    completed_actions: list[str] = Field(
        default_factory=list,
        description="已完成的动作"
    )
    pending_actions: list[str] = Field(
        default_factory=list,
        description="待执行的动作"
    )
    intermediate_results: dict = Field(
        default_factory=dict,
        description="中间结果缓存"
    )


# 注意：UserProfile 和 FoodRecord 是数据库表模型，已移至 tables.py
# 需要时：from src.models.tables import UserProfile, FoodRecord

