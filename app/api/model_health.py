from __future__ import annotations

import datetime as dt
from fastapi import APIRouter, Depends
from sqlalchemy import select, func, desc, case
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from .. import models
from ..ml.model import _load_model_and_preprocessor

router = APIRouter(prefix="/api/monitoring", tags=["monitoring"])


@router.get("/model-health")
async def model_health(db: AsyncSession = Depends(get_db)):
    now = dt.datetime.utcnow()
    since = now - dt.timedelta(hours=24)

    try:
        model, feature_names = _load_model_and_preprocessor()
        model_loaded = True
        features_count = len(feature_names)
    except Exception:
        model_loaded = False
        features_count = None

    PL = models.PredictionLog

    # Агрегация по доступным полям: общее число, число симуляций, число уникальных клиентов
    agg = (
        await db.execute(
            select(
                func.count(PL.id),
                func.sum(case((PL.is_simulation.is_(True), 1), else_=0)),
                func.count(func.distinct(PL.client_id)),
            ).where(PL.created_at >= since)
        )
    ).one()

    total = int(agg[0] or 0)
    simulations = int(agg[1] or 0)
    unique_clients = int(agg[2] or 0)

    # latency / errors metrics are not available in current schema
    avg_latency = None
    p95_latency = None

    # версии
    versions = (
        await db.execute(
            select(PL.model_version, func.count())
            .where(PL.created_at >= since)
            .group_by(PL.model_version)
            .order_by(desc(func.count()))
        )
    ).all()

    versions_usage = [
        {"model_version": v[0], "count": v[1]} for v in versions
    ]

    # Топ источников запросов (так как в схеме нет error_code)
    top_request_sources = (
        await db.execute(
            select(PL.request_source, func.count())
            .where(PL.created_at >= since, PL.request_source.is_not(None))
            .group_by(PL.request_source)
            .order_by(desc(func.count()))
            .limit(5)
        )
    ).all()

    top_request_sources = [{"request_source": r[0], "count": r[1]} for r in top_request_sources]

    # --- client_state ---
    CS = models.ClientState

    # Клиенты: общее число клиентов и число активных за 24ч (по updated_at в ClientState)
    total_clients = (
        await db.execute(select(func.count()).select_from(models.Client))
    ).scalar_one()

    recent_clients = (
        await db.execute(
            select(func.count()).select_from(CS).where(CS.updated_at >= since)
        )
    ).scalar_one()

    # Сегменты в явном виде в ClientState не хранятся (есть поле features JSON).
    # Извлечение распределения сегментов требует специфичных JSON-операций и зависит
    # от СУБД. Пока возвращаем пустой список и флаг о недоступности.
    segments_dist = []
    segments_available = False

    # Результат
    return {
        "model": {
            "loaded": model_loaded,
            "features_count": features_count,
        },
        "traffic_24h": {
            "total": total,
            "simulations": simulations,
            "unique_clients": unique_clients,
            "avg_latency": avg_latency,
            "p95_latency": p95_latency,
            "versions": versions_usage,
            "top_request_sources": top_request_sources,
        },
        "clients": {
            "total": total_clients,
            "active_24h": recent_clients,
            "segments": segments_dist,
            "segments_available": segments_available,
        },
    }