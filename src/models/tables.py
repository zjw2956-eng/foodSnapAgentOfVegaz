"""数据库表模型 —— SQLModel 定义，持久化到 SQLite

职责：只放需要落库的表模型（UserProfile、FoodRecord）。
API/Agent 内存态模型放 schemas.py，不混在这里。

设计要点：
- list 字段数据库存不了，统一存 JSON 字符串
- 原字段保持 str（给数据库），新增 _list property（给业务代码）
- table=True 的类用 SQLField（SQLModel），不用 Pydantic 的 Field
"""

import json
from typing import Optional

from pydantic import field_validator
from sqlmodel import SQLModel, Field as SQLField


class UserProfile(SQLModel, table=True):
    """用户画像 —— 美食社交为核心，健康只是锦上添花

    SQLModel 表模型：既是数据库表，又是数据模型。
    list 字段存为 JSON 字符串（数据库无数组类型），读写时自动转换。
    """
    __tablename__ = "user_profiles"

    user_id: str = SQLField(primary_key=True)

    # 🍔 核心：口味和偏好（存为 JSON 字符串）
    taste_preferences: str = SQLField(default="[]")
    favorite_cuisines: str = SQLField(default="[]")

    # ✍️ 核心：社交风格
    article_style_preference: str = SQLField(default="吃货风")

    # 📸 核心：使用习惯
    frequently_ordered_dishes: str = SQLField(default="[]")
    visited_restaurants: str = SQLField(default="[]")

    # 💪 可选：健康信息（用户不主动设置就永远不提）
    health_goal: Optional[str] = SQLField(default=None)
    allergies: str = SQLField(default="[]")

    created_at: Optional[str] = SQLField(default=None)
    updated_at: Optional[str] = SQLField(default=None)

    # ===== 字段转换器：list → JSON 字符串 =====
    @field_validator("taste_preferences", "favorite_cuisines",
                     "frequently_ordered_dishes", "visited_restaurants",
                     "allergies", mode="before")
    @classmethod
    def _list_to_json(cls, v):
        """存入前：list 自动转 JSON 字符串；已是 str 则保持"""
        if isinstance(v, list):
            return json.dumps(v, ensure_ascii=False)
        return v

    # ===== 便捷访问器：业务代码用 _list 拿到真正的 list =====
    @property
    def taste_preferences_list(self) -> list[str]:
        return json.loads(self.taste_preferences) if self.taste_preferences else []

    @property
    def favorite_cuisines_list(self) -> list[str]:
        return json.loads(self.favorite_cuisines) if self.favorite_cuisines else []

    @property
    def frequently_ordered_dishes_list(self) -> list[str]:
        return json.loads(self.frequently_ordered_dishes) if self.frequently_ordered_dishes else []

    @property
    def visited_restaurants_list(self) -> list[str]:
        return json.loads(self.visited_restaurants) if self.visited_restaurants else []

    @property
    def allergies_list(self) -> list[str]:
        return json.loads(self.allergies) if self.allergies else []


class FoodRecord(SQLModel, table=True):
    """饮食历史记录 —— Agent 长期记忆的核心数据

    每次识别菜品后保存一条，用于：
    - 查询用户吃过的菜（精确查询走 SQLite）
    - 语义检索"上次那个红红的菜"（向量检索走 Qdrant，这里只存结构化数据）
    """
    __tablename__ = "food_records"

    id: Optional[int] = SQLField(default=None, primary_key=True)
    user_id: str = SQLField(index=True)  # 加索引，按用户查很快
    dish_name: str
    ingredients: str = SQLField(default="[]")  # JSON 字符串
    cuisine_type: str = SQLField(default="")
    image_url: str = SQLField(default="")
    created_at: str = SQLField(default="")

    @field_validator("ingredients", mode="before")
    @classmethod
    def _ingredients_to_json(cls, v):
        if isinstance(v, list):
            return json.dumps(v, ensure_ascii=False)
        return v

    @property
    def ingredients_list(self) -> list[str]:
        return json.loads(self.ingredients) if self.ingredients else []
