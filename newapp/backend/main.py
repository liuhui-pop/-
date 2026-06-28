from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import hsi, model, result

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发阶段先用*，允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hsi.router,    prefix="/api/hsi")
app.include_router(model.router,  prefix="/api/model")
app.include_router(result.router, prefix="/api/result")