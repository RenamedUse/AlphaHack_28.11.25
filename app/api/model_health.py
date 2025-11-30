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

    avg_latency = None
    p95_latency = None

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

    top_request_sources = (
        await db.execute(
            select(PL.request_source, func.count())
            .where(PL.created_at >= since, PL.request_source.is_not(None))
            .group_by(PL.request_source)
            .order_by(desc(func.count()))
            .limit(5)
        )
    ).all()
    top_request_sources = [
        {"request_source": r[0], "count": r[1]} for r in top_request_sources
    ]

    IJ = models.ImportJob

    jobs_agg = (
        await db.execute(
            select(
                func.count(IJ.id),
                func.sum(
                    case((IJ.status == "failed", 1), else_=0)
                ),
            ).where(IJ.created_at >= since)
        )
    ).one()

    import_jobs_total = int(jobs_agg[0] or 0)
    import_jobs_failed = int(jobs_agg[1] or 0)

    jobs_errors_rows = (
        await db.execute(
            select(IJ.error, func.count())
            .where(IJ.created_at >= since, IJ.error.is_not(None))
            .group_by(IJ.error)
            .order_by(desc(func.count()))
            .limit(5)
        )
    ).all()
    top_import_errors = [
        {"error": r[0], "count": r[1]} for r in jobs_errors_rows
    ]

    CS = models.Client

    total_clients = (
        await db.execute(select(func.count()).select_from(models.Client))
    ).scalar_one()

    recent_clients = (
        await db.execute(
            select(func.count()).select_from(models.ClientState).where(
                models.ClientState.updated_at >= since
            )
        )
    ).scalar_one()

    segments_dist = []
    segments_available = False

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
        "errors_24h": {
            "import_jobs_total": import_jobs_total,
            "import_jobs_failed": import_jobs_failed,
            "top_import_errors": top_import_errors,
        },
        "clients": {
            "total": total_clients,
            "active_24h": recent_clients,
            "segments": segments_dist,
            "segments_available": segments_available,
        },
    }