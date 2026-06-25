"""记忆管理器 —— SQLModel + Qdrant 统一封装

    SQLite（SQLModel）存结构化数据，Qdrant 存向量数据。
    职责分工：Qdrant 负责语义召回（找到是哪个），SQLite 取完整数据。
    """

from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Session, create_engine, select
from src.config import SQLITE_DB_PATH
from src.models.tables import UserProfile, FoodRecord


class MemoryManager:
    """记忆系统统一入口

    对外暴露高层 API，内部协调 SQLite 和 Qdrant。
    调用方只需要 MemoryManager，不用关心数据存在哪种库里。
    """

    def __init__(self, db_path: str = "", qdrant_host: str = "localhost", qdrant_port: int = 6333):
        self.db_path = db_path or SQLITE_DB_PATH
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self._qdrant = None  # QdrantClient，延迟初始化
        # SQLModel 引擎（check_same_thread=False 允许跨线程/协程使用）
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            connect_args={"check_same_thread": False},
            echo=False,  # 生产环境关 SQL 日志
        )
        SQLModel.metadata.create_all(self.engine)  # 自动建表（幂等）

    # ==================== 用户画像 ====================
    def get_profile(self, user_id: str) -> UserProfile:
        """获取用户画像，不存在则返回默认画像"""
        with Session(self.engine) as session:
            profile = session.exec(
                select(UserProfile).where(UserProfile.user_id == user_id)
            ).first()
            if profile is None:
                return UserProfile(user_id=user_id)
            return profile

    def save_profile(self, profile: UserProfile) -> UserProfile:
        """保存或更新用户画像（upsert）

        用 merge() 实现：存在就更新，不存在就插入。
        """
        now = datetime.now().isoformat()
        if not profile.created_at:
            profile.created_at = now
        profile.updated_at = now

        with Session(self.engine) as session:
            merged = session.merge(profile)
            session.commit()
            session.refresh(merged)
            return merged

    # ==================== 饮食记录 ====================
    def save_food_records(
        self, user_id: str, dishes: list, image_url: str
    ) -> None:
        """批量保存饮食记录

        Args:
            user_id: 用户 ID
            dishes: DishItem 列表（来自 schemas.py）
            image_url: 图片 URL
        """
        now = datetime.now().isoformat()
        with Session(self.engine) as session:
            for dish in dishes:
                record = FoodRecord(
                    user_id=user_id,
                    dish_name=dish.name,
                    ingredients=", ".join(dish.ingredients),
                    cuisine_type=dish.cuisine_type,
                    image_url=image_url,
                    created_at=now,
                )
                session.add(record)
                # TODO: 向量双写 Qdrant（降级：失败不影响主流程）
            session.commit()

    def query_history(
        self, user_id: str, query_text: str = "", limit: int = 5
    ) -> list[FoodRecord]:
        """查询用户饮食历史

        先尝试 Qdrant 语义检索，失败则降级到 SQLite 关键词模糊查。
        query_text 为空时返回最近记录。
        """
        if query_text:
            try:
                ids = self._qdrant_search(user_id, query_text, limit)
                if ids:
                    return self._get_records_by_ids(ids)
            except Exception:
                pass  # 降级

        # 降级：SQLite 关键词模糊查询 + 时间倒序
        return self._keyword_search(user_id, query_text, limit)

    def get_recent_records(self, user_id: str, limit: int = 5) -> list[FoodRecord]:
        """获取用户最近的饮食记录"""
        with Session(self.engine) as session:
            return list(session.exec(
                select(FoodRecord)
                .where(FoodRecord.user_id == user_id)
                .order_by(FoodRecord.created_at.desc())
                .limit(limit)
            ).all())

    # ==================== Qdrant 操作（延迟初始化 + 降级） ====================
    def _ensure_qdrant(self):
        """延迟初始化 Qdrant（第一次用时才连）"""
        if self._qdrant is not None:
            return

        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        self._qdrant = QdrantClient(
            host=self.qdrant_host, port=self.qdrant_port)

        # 创建 Collection（幂等）
        names = [c.name for c in self._qdrant.get_collections().collections]
        if "user_food_history" not in names:
            self._qdrant.create_collection(
                collection_name="user_food_history",
                vectors_config=models.VectorParams(
                    size=1024, distance=models.Distance.COSINE
                ),
            )

    def _qdrant_search(self, user_id: str, query_text: str, limit: int) -> list[int]:
        """向量检索：query_text → 向量 → Qdrant 搜索 + user_id 过滤"""
        self._ensure_qdrant()
        from qdrant_client.http import models

        vector = self._embed(query_text)
        results = self._qdrant.query_points(
            collection_name="user_food_history",
            query=vector,
            query_filter=models.Filter(must=[
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                )
            ]),
            limit=limit,
        ).points

        return [int(r.id) for r in results if r.id]

    def _embed(self, text: str) -> list[float]:
        """文本向量化（TODO: 接入真实 embedding 模型）

        占位实现：1024 维零向量。
        后续接入：Qwen embedding / BGE-M3 / DashScope text-embedding。
        """
        return [0.0] * 1024

    def _get_records_by_ids(self, ids: list[int]) -> list[FoodRecord]:
        """用 Qdrant 返回的 ID 去 SQLite 取完整记录"""
        with Session(self.engine) as session:
            return list(session.exec(
                select(FoodRecord).where(FoodRecord.id.in_(ids))
            ).all())
    def _keyword_search(self, user_id: str, keyword: str, limit: int) -> list[FoodRecord]:
        """降级方案：SQLite LIKE 关键词模糊搜索"""
        with Session(self.engine) as session:
            if keyword:
                return list(session.exec(
                    select(FoodRecord)
                    .where(
                        FoodRecord.user_id == user_id,
                        FoodRecord.dish_name.like(f"%{keyword}%"),
                    )
                    .order_by(FoodRecord.created_at.desc())
                    .limit(limit)
                ).all())
            else:
                return list(session.exec(
                    select(FoodRecord)
                    .where(FoodRecord.user_id == user_id)
                    .order_by(FoodRecord.created_at.desc())
                    .limit(limit)
                ).all())
