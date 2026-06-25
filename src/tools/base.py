"""工具系统基类 -- 依赖注入容器+工具注册中心"""
from dataclasses import dataclass, field
from typing import Optional
import httpx

"""@dataclass
Python 标准库的数据类，自动生成 __init__，比普通
class 少写很多代码。PydanticAI 官方推荐用 dataclass做 Deps
"""
"""httpx.AsyncClient
异步 HTTP 客户端，连接池复用。不要每次请求都new，
一个 client 复用整个应用生命周期
"""
"""tool_registry全局单例
应用启动时注册所有工具，Agent运行时查询。
简单场景下够用，不需要依赖注入框架
"""
# ==================依赖注入容器=================


@dataclass
class AgentDeps:
    """
    Agent 运行时依赖 —— 通过 PydanticAI RunContext 注入到工具函数中
    这个类的实例会在每次请求时创建，持有所有外部资源的连接。
    PydanticAI 的依赖注入机制类似于 FastAPI 的 Depends：
    你在 agent.run(deps=...) 传入，工具函数通过 ctx.deps 访问。
    """
    user_id: str
    image_url: str
    intent: Optional[str] = None

    # API密钥
    amap_api_key: str = ""
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # 共享的 HTTP 客户端（复用连接池，不要每次请求都 new 一个）
    http_client: Optional[httpx.AsyncClient] = None


# ==================== 工具注册中心 ====================
@dataclass
class ToolMeta:
    """工具的元信息"""
    name: str
    description: str
    category: str  # perception / memory / knowledge / action / external
    requires_confirmation: bool = False  # 是否需要用户确认后才执行


class ToolRegistry:
    """工具注册中心 —— 管理所有可用工具

    作用：
    1. 注册工具（给每个工具一个名字和描述）
    2. 按类别列出工具（供 Planner 选择）
    3. 生成工具描述文本（注入 Agent system_prompt）
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolMeta] = {}

    def register(self, meta: ToolMeta) -> None:
        """注册一个工具"""
        self._tools[meta.name] = meta

    def get_tools_by_category(self, category: str) -> list[ToolMeta]:
        """按类别获取工具列表--Planner用它选择工具"""
        return [t for t in self._tools.values() if t.category == category]

    def list_all(self) -> list[ToolMeta]:
        """列出所有已注册的工具"""
        return list(self._tools.values())

    def get_tool_descriptions(self) -> str:
        """生成工具描述文本，用于注入Agent的system_prompt"""
        lines = ["可用工具列表："]
        for meta in self._tools.values():
            lines.append(f" -{meta.name} [{meta.category}]:{meta.description}")
        return "\n".join(lines)


# ==================== 全局工具注册中心（单例） ====================
tool_registry = ToolRegistry()
