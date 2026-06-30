import httpx
from pydantic_ai import RunContext
from src.tools.base import AgentDeps
from src.models.schemas import RestaurantItem
from src.config import REQUEST_TIMEOUT

AMAP_BASE_URL = "https://restapi.amap.com/v3"


async def _amap_get(ctx: RunContext[AgentDeps], path: str, params: dict) -> dict:
    """高德 API 通用 GET 请求封装。

    统一处理：API Key 注入、超时、降级（出错返回空结果而非抛异常）。
    这是项目"超时降级"要求的体现——地图查不到不能让整个 Agent 挂掉。

    Args:
        ctx: 运行时上下文
        path: API 路径，如 /place/around
        params: 查询参数

    Returns:
        高德返回的 JSON（dict）；出错时返回 {"status": "0", "pois": []}
    """
    params = {**params, "key": ctx.deps.amap_api_key}
    client = ctx.deps.http_client or httpx.AsyncClient()
    try:
        resp = await client.get(
            f"{AMAP_BASE_URL}{path}",
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, httpx.TimeoutException) as e:
        # 降级：返回空结果，让 Agent 能继续走别的路
        return {"status": "0", "error": str(e), "pois": []}
    finally:
        if ctx.deps.http_client is None:
            await client.aclose()


async def amap_search_nearby_restaurants(
    ctx: RunContext[AgentDeps],
    keyword: str,
    location: str,
    radius: int = 3000,
) -> list[RestaurantItem]:
    """搜索用户附近的餐厅 POI。

    Args:
        ctx: 运行时上下文
        keyword: 搜索关键词，如"麻辣香锅"或"川菜"
        location: 用户经纬度，格式 "经度,纬度"（如 "116.397,39.908"）
        radius: 搜索半径（米），默认 3000

    Returns:
        附近的餐厅列表，按距离排序
    """
    data = await _amap_get(ctx, "/place/around", {
        "keywords": keyword,
        "location": location,
        "radius": radius,
        "types": "050000",  # 高德 POI 类型码：餐饮服务
        "offset": 10,
        "page": 1,
        "extensions": "all",
    })
    # 高德 POI 类型码 → 餐厅类别名
    _TYPE_MAP = {
        "050100": "中餐厅", "050200": "外国餐厅", "050300": "快餐厅",
        "050400": "咖啡厅", "050500": "茶艺馆", "050600": "甜品店",
    }

    restaurants = []
    for poi in data.get("pois", []):
        biz = poi.get("biz_ext", {})
        cost = biz.get("cost", "0")
        price = f"人均{cost}元" if cost and cost != "0" else "人均未知"
        rating = float(biz.get("rating", 0) or 0)
        distance = int(poi.get("distance", 0) or 0)
        walk_min = max(1, round(distance / 80))  # 步行速度80m/min

        # 推荐理由：融合距离、评分、价位、区域信息
        parts = [f"步行约{walk_min}分钟"]
        if rating > 0:
            parts.append(f"评分{rating}")
        parts.append(price)
        poi_type = _TYPE_MAP.get(poi.get("type", ""), "")
        if poi_type:
            parts[-1] = f"{poi_type}·{parts[-1]}"
        reason = f"距你{poi.get('distance','?')}米（{'，'.join(parts)}）"

        restaurants.append(RestaurantItem(
            name=poi.get("name", "未知餐厅"),
            address=poi.get("address", "") or poi.get("pname", ""),
            distance=distance,
            rating=rating,
            price_level=price,
            has_similar_dish=True,
            recommendation_reason=reason,
            must_try_dishes=[],
        ))
    # 按距离升序
    restaurants.sort(key=lambda r: r.distance)
    return restaurants


async def amap_reverse_geocode(
    ctx: RunContext[AgentDeps],
    location: str,
) -> dict:
    """
    逆地理编码：把经纬度转成可读地址。

    Agent 用它把"116.397,39.908"变成"北京市东城区天安门"，
    这样文案和推荐理由里能出现具体地名。

    Args:
        ctx: 运行时上下文
        location: 经纬度 "经度,纬度"

    Returns:
        {"address": "完整地址", "city": "城市名"}
    """
    data = await _amap_get(ctx, "/geocode/regeo", {
        "location": location,
        "extensions": "base",
    })
    regeo = data.get("regeocode", {})
    addr_comp = regeo.get("addressComponent", {})
    return {
        "address": regeo.get("formatted_address", ""),
        "city": addr_comp.get("city") or addr_comp.get("province", ""),
    }


async def amap_walking_direction(
    ctx: RunContext[AgentDeps],
    origin: str,
    destination: str,
) -> dict:
    """
    步行路径规划：给用户"怎么走过去"的建议。

    Args:
        ctx: 运行时上下文
        origin: 起点经纬度 "经度,纬度"
        destination: 终点经纬度 "经度,纬度"

    Returns:
        {"distance": 米, "duration": 分钟, "steps": 步骤数}
    """
    data = await _amap_get(ctx, "/direction/walking", {
        "origin": origin,
        "destination": destination,
    })
    paths = data.get("route", {}).get("paths", [])
    path = paths[0] if paths else {}
    return {
        "distance": int(path.get("distance", 0)),
        "duration": round(int(path.get("duration", 0)) / 60),  # 秒转分钟
        "steps": len(path.get("steps", [])),
    }
