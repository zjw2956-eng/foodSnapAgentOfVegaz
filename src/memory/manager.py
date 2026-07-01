"""记忆管理器 —— SQLModel + Qdrant 统一封装

    SQLite（SQLModel）存结构化数据，Qdrant 存向量数据。
    职责分工：Qdrant 负责语义召回（找到是哪个），SQLite 取完整数据。
    """

from typing import Optional, TYPE_CHECKING
from datetime import datetime
from sqlmodel import SQLModel, Session, create_engine, select
from src.config import (
    SQLITE_DB_PATH, DASHSCOPE_BASE_URL, DASHSCOPE_API_KEY,
    DASHSCOPE_EMBEDDING_MODEL, DASHSCOPE_EMBEDDING_DIM,
)
from src.models.tables import UserProfile, FoodRecord

if TYPE_CHECKING:
    from qdrant_client import QdrantClient


class MemoryManager:
    """记忆系统统一入口

    对外暴露高层 API，内部协调 SQLite 和 Qdrant。
    调用方只需要 MemoryManager，不用关心数据存在哪种库里。
    """

    def __init__(self, db_path: str = "", qdrant_host: str = "localhost", qdrant_port: int = 6333):
        self.db_path = db_path or SQLITE_DB_PATH
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self._qdrant: Optional["QdrantClient"] = None  # 延迟初始化，_ensure_qdrant() 填充
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
    def _ensure_qdrant(self) -> "QdrantClient":
        """延迟初始化 Qdrant（第一次用时才连），返回已就绪的 client"""
        if self._qdrant is not None:
            return self._qdrant

        from qdrant_client import QdrantClient
        from qdrant_client.http import models

        self._qdrant = QdrantClient(
            host=self.qdrant_host, port=self.qdrant_port)
        client = self._qdrant  # ponytail: 局部别名让类型检查器确认非空

        # 创建 Collection（幂等 —— collection 存在就跳过，不会重复向量化）
        names = [c.name for c in client.get_collections().collections]
        if "user_food_history" not in names:
            client.create_collection(
                collection_name="user_food_history",
                vectors_config=models.VectorParams(
                    size=DASHSCOPE_EMBEDDING_DIM, distance=models.Distance.COSINE
                ),
            )
        if "dish_knowledge" not in names:
            client.create_collection(
                collection_name="dish_knowledge",
                vectors_config=models.VectorParams(
                    size=DASHSCOPE_EMBEDDING_DIM, distance=models.Distance.COSINE
                ),
            )
            self._seed_dish_knowledge()
        return client

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
        """文本向量化 —— OpenAI 兼容模式，复用 DASHSCOPE_BASE_URL + /embeddings。"""
        import httpx
        try:
            resp = httpx.post(
                f"{DASHSCOPE_BASE_URL}/embeddings",
                headers={"Authorization": f"Bearer {DASHSCOPE_API_KEY}"},
                json={
                    "model": DASHSCOPE_EMBEDDING_MODEL,
                    "input": text,
                    "dimensions": DASHSCOPE_EMBEDDING_DIM,
                    "encoding_format": "float",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["data"][0]["embedding"]
        except Exception:
            # 降级：embedding 失败不影响主流程，返回零向量
            return [0.0] * DASHSCOPE_EMBEDDING_DIM

    def search_dish_knowledge(self, query_text: str, limit: int = 3) -> list[dict]:
        """语义检索菜品知识库，返回匹配的营养/趣味知识。

        Args:
            query_text: 查询文本（菜品名或描述）
            limit: 返回条数

        Returns:
            匹配的知识条目列表，每项含 dish_name/calories/protein/fat/carbs/fun_fact
        """
        self._ensure_qdrant()
        try:
            vector = self._embed(query_text)
            results = self._qdrant.query_points(
                collection_name="dish_knowledge",
                query=vector,
                limit=limit,
            ).points
            return [
                {k: v for k, v in r.payload.items() if k != "text"}
                for r in results if r.payload
            ]
        except Exception:
            return []

    def _seed_dish_knowledge(self):
        """种子数据：常见菜品的营养知识（首次创建 collection 时自动写入）。"""
        # ponytail: 硬编码种子数据，覆盖常见中餐/西餐/日料
        dishes = [
            {"dish_name": "烤鸡", "calories": 240, "protein": 27.0, "fat": 14.0, "carbs": 0.0,
             "fun_fact": "烤鸡带皮吃更香但脂肪翻倍，去皮鸡胸肉是健身党首选"},
            {"dish_name": "麻辣香锅", "calories": 200, "protein": 15.0, "fat": 14.0, "carbs": 8.0,
             "fun_fact": "香锅的灵魂在底料——郫县豆瓣+花椒+干辣椒的铁三角组合"},
            {"dish_name": "寿司", "calories": 150, "protein": 6.0, "fat": 0.5, "carbs": 28.0,
             "fun_fact": "江户前寿司的米饭用醋、糖、盐按秘方比例调制，温度要控制在人体体温"},
            {"dish_name": "汉堡", "calories": 295, "protein": 17.0, "fat": 14.0, "carbs": 30.0,
             "fun_fact": "汉堡肉饼的美拉德反应是香气的关键——高温煎制让表面焦化产生400+种风味物质"},
            {"dish_name": "火锅", "calories": 180, "protein": 12.0, "fat": 13.0, "carbs": 5.0,
             "fun_fact": "重庆老火锅用牛油增香，成都火锅用清油（菜籽油）更清爽"},
            {"dish_name": "披萨", "calories": 266, "protein": 11.0, "fat": 10.0, "carbs": 33.0,
             "fun_fact": "那不勒斯披萨协会规定：正宗玛格丽特只能用圣马扎诺番茄+水牛奶莫扎瑞拉+罗勒"},
            {"dish_name": "水煮鱼", "calories": 160, "protein": 18.0, "fat": 9.0, "carbs": 2.0,
             "fun_fact": "水煮鱼最后的泼油是灵魂一步——200°C热油浇上去瞬间激发花椒和干辣椒的香气"},
            {"dish_name": "牛排", "calories": 250, "protein": 26.0, "fat": 16.0, "carbs": 0.0,
             "fun_fact": "牛排熟度从一分到全熟，内部温度差仅30°C——55°C三分熟到75°C全熟"},
            {"dish_name": "酸菜鱼", "calories": 120, "protein": 16.0, "fat": 5.0, "carbs": 3.0,
             "fun_fact": "酸菜鱼的精髓在泡菜——老坛酸菜发酵至少15天才有那种醇厚的酸香"},
            {"dish_name": "烧烤", "calories": 220, "protein": 20.0, "fat": 15.0, "carbs": 2.0,
             "fun_fact": "炭烤比电烤多一层烟熏风味——木炭燃烧产生的愈创木酚是烧烤香气的来源"},
            {"dish_name": "沙拉", "calories": 80, "protein": 3.0, "fat": 5.0, "carbs": 8.0,
             "fun_fact": "沙拉酱热量炸弹预警——一勺蛋黄酱约90大卡，油醋汁不到30大卡"},
            {"dish_name": "蛋炒饭", "calories": 190, "protein": 7.0, "fat": 6.0, "carbs": 26.0,
             "fun_fact": "隔夜饭炒饭更粒粒分明——冷藏让淀粉回生，含水量降低，炒出来不粘锅"},
        ]
        from qdrant_client.http import models
        points = []
        for i, dish in enumerate(dishes):
            vec = self._embed(dish["dish_name"])
            payload = {**dish, "text": dish["dish_name"]}
            points.append(models.PointStruct(id=i + 1, vector=vec, payload=payload))
        self._qdrant.upsert(collection_name="dish_knowledge", points=points)

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
