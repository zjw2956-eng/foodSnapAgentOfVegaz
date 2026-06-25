"""foodSnapAgent 启动入口"""

import uvicorn
from fastapi import FastAPI
from src.api.routes import router, lifespan

app = FastAPI(
    title="foodSnapAgent",
    description="食物拍照识别 AI Agent —— 拍美食、写文案、推荐同款餐厅",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


def main():
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
