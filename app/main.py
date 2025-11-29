"""Точка входа FastAPI-приложения.

Здесь создаётся объект приложения, настраивается хук старта и
подключаются роутеры (публичные и административные).
"""

from __future__ import annotations

import os

from fastapi import FastAPI

from .database import create_tables

app = FastAPI(title="TunTunDohod", version="0.1.0")


@app.on_event("startup")
async def on_startup() -> None:
    """Инициализация БД и служебных каталогов при старте приложения."""
    # Создаём таблицы, если их ещё нет
    await create_tables()

    # Гарантируем наличие каталога для импортируемых CSV
    imports_path = os.environ.get("IMPORTS_PATH", "/tmp/imports")
    os.makedirs(imports_path, exist_ok=True)


# Подключаем роутеры
from .api.public import router as public_router
from .api.admin import router as admin_router

app.include_router(public_router)
app.include_router(admin_router)