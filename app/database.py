"""Настройка базы данных и управление сессиями.

Модуль определяет асинхронный движок SQLAlchemy и фабрику сессий,
а также предоставляет зависимость для получения сессии в обработчиках FastAPI.
Функция ``create_tables`` может создавать таблицы автоматически при старте приложения.
"""

from __future__ import annotations

import os
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Базовый класс для всех ORM-моделей.

    Все модели должны наследоваться от этого класса.
    При необходимости можно переопределить __repr__ в конкретных моделях.
    """

    pass


# URL подключения к БД читаем из переменной окружения.
# Её нужно передавать при запуске (docker-compose или любым другим способом).
DATABASE_URL = os.environ.get("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable must be set")

# Асинхронный движок SQLAlchemy
engine = create_async_engine(DATABASE_URL, future=True, echo=False)

#: Фабрика AsyncSession, привязанная к движку.
SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Зависимость FastAPI, предоставляющая асинхронную сессию.

    Сессия используется в контексте запроса:
    при успешном завершении — выполняется commit,
    при исключении — выполняется rollback.
    В любом случае сессия корректно закрывается.
    """
    async with SessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_tables() -> None:
    """Создать все таблицы в базе данных.

    Функцию имеет смысл вызывать при старте приложения, чтобы гарантировать,
    что схема БД существует. В реальном проде вместо этого обычно используют
    Alembic и миграции.
    """
    async with engine.begin() as conn:
        # Импортируем модели, чтобы они зарегистрировались в metadata
        from . import models  # noqa: F401

        await conn.run_sync(Base.metadata.create_all)