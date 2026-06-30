"""FastAPI 路由 —— Java 主服务调用的 HTTP 入口"""

import httpx
from fastapi import APIRouter, Depends, FastAPI, HTTPException
from contextlib import asynccontextmanager

from src.config import SQLITE_DB_PATH, QDRANT_HOST, QDRANT_PORT, REQUEST_TIMEOUT
from src.models.schemas import AnalysisRequest, AnalysisResponse, HealthResponse
from src.memory.manager import MemoryManager
from src.agent import process_request


# ==================== 生命周期管理 ====================

_memory_manager: MemoryManager | None = None
_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动/关闭时的资源管理。

    启动时创建 MemoryManager 和 httpx.AsyncClient 单例，
    关闭时释放连接。
    """
    global _memory_manager, _http_client

    # 启动
    _memory_manager = MemoryManager(
        db_path=SQLITE_DB_PATH,
        qdrant_host=QDRANT_HOST,
        qdrant_port=QDRANT_PORT,
    )
    _http_client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)

    yield  # 应用运行中

    # 关闭
    await _http_client.aclose()


def get_memory_manager() -> MemoryManager:
    """依赖注入：获取 MemoryManager 单例"""
    assert _memory_manager is not None, "应用未启动"
    return _memory_manager


def get_http_client() -> httpx.AsyncClient:
    """依赖注入：获取共享的 httpx 客户端"""
    assert _http_client is not None, "应用未启动"
    return _http_client


# ==================== 路由 ====================

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查 —— Java 服务探活用"""
    return HealthResponse(status="ok", version="0.1.0")


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_food(
    request: AnalysisRequest,
    manager: MemoryManager = Depends(get_memory_manager),
    client: httpx.AsyncClient = Depends(get_http_client),
):
    """食物图片分析接口 —— Java 主服务的唯一调用入口。

    接收图片 URL + 用户 ID + 意图，返回识别结果 + 文案 + 餐厅推荐。
    超时/降级在工具层内部处理，接口层只负责转发。

    Args:
        request: 分析请求
        manager: MemoryManager（自动注入）
        client: httpx 客户端（自动注入）

    Returns:
        结构化分析结果

    Raises:
        HTTPException: Agent 处理失败时返回 500
    """
    try:
        result = await process_request(request, manager, client)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent 处理失败: {str(e)}",
        )
