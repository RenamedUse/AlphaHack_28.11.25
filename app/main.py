from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .database import create_tables

app = FastAPI(title="TunTunDohod", version="0.1.0")

ALLOWED_ORIGINS = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def on_startup() -> None:
    """Инициализация БД и служебных каталогов при старте приложения"""
    await create_tables()
    imports_path = os.environ.get("IMPORTS_PATH", "/tmp/imports")
    os.makedirs(imports_path, exist_ok=True)


# Подключаем роутеры
from .api.public import router as public_router
from .api.admin import router as admin_router
from .api.model_health import router as monitoring_router

app.include_router(public_router)
app.include_router(admin_router)
app.include_router(monitoring_router)