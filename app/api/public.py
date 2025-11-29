"""Публичные маршруты API

Сюда вынесены эндпоинты:
- импорт CSV с клиентами,
- статус задач импорта,
- список клиентов,
- карточка клиента,
- симуляция параметров
"""

from __future__ import annotations

import os
import uuid
import shutil
import datetime
from typing import List, Optional, Dict, Any

import asyncio

from fastapi import (
    APIRouter,
    Depends,
    UploadFile,
    File,
    BackgroundTasks,
    HTTPException,
    Path,
    Query,
)
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.ml.model import IncomeModel

from ..database import get_db
from .. import models, schemas
from ..tasks import compute_recommendations, process_import_job
from ..utils import canonicalize_features

router = APIRouter(prefix="/api", tags=["public"])


@router.post("/imports/income-csv", response_model=schemas.ImportJobResponse)
async def import_income_csv(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
) -> schemas.ImportJobResponse:
    """Загрузить CSV с клиентами и поставить его в очередь на фоновую обработку

    Требования к CSV:
    - должна быть колонка ``external_id`` (или аналогичная), которая идентифицирует клиента;
    - все остальные колонки воспринимаются как признаки.
    """
    imports_path = os.environ.get("IMPORTS_PATH", "/tmp/imports")
    os.makedirs(imports_path, exist_ok=True)

    filename = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(imports_path, filename)

    try:
        with open(file_path, "wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
    finally:
        file.file.close()

    job = models.ImportJob(
        file_name=file.filename,
        file_path=file_path,
        status="queued",
        total_rows=None,
        processed_rows=0,
    )
    db.add(job)
    await db.flush()
    await db.commit()

    # Wrap async function to run in background
    def run_async_task():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(process_import_job(job.id))
        finally:
            loop.close()

    background_tasks.add_task(run_async_task)

    return schemas.ImportJobResponse.model_validate(job)


@router.get("/imports/{job_id}", response_model=schemas.ImportJobResponse)
async def get_import_job(
    job_id: int = Path(..., gt=0),
    db: AsyncSession = Depends(get_db),
) -> schemas.ImportJobResponse:
    """Получить статус и прогресс конкретной задачи импорта CSV."""
    job = await db.get(models.ImportJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Import job not found")
    return schemas.ImportJobResponse.model_validate(job)


@router.get("/clients", response_model=List[schemas.ClientSummary])
async def list_clients(
    segment_code: Optional[str] = Query(
        None,
        description="Фильтр по коду сегмента (например: 'low', 'middle', 'high', 'premium')",
    ),
    min_income: Optional[float] = Query(
        None,
        description="Фильтр: предсказанный доход >= min_income",
    ),
    max_income: Optional[float] = Query(
        None,
        description="Фильтр: предсказанный доход <= max_income",
    ),
    limit: int = Query(50, gt=0, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> List[schemas.ClientSummary]:
    """Вернуть постраничный список клиентов с фильтрами по сегменту и доходу."""
    from sqlalchemy.orm import aliased

    seg_alias = aliased(models.Segment)

    # Основной запрос: клиент + его текущее состояние + вычисленный сегмент
    stmt = (
        select(
            models.Client.external_id,
            models.Client.display_name,
            models.ClientState.income_pred,
            seg_alias.code.label("segment_code"),
            seg_alias.name.label("segment_name"),
            models.ClientState.updated_at,
        )
        .join(models.ClientState, models.Client.id == models.ClientState.client_id)
        .join(
            seg_alias,
            (models.ClientState.income_pred >= seg_alias.min_income)
            & (
                (seg_alias.max_income.is_(None))
                | (models.ClientState.income_pred < seg_alias.max_income)
            ),
        )
    )

    # Фильтр по коду сегмента
    if segment_code:
        stmt = stmt.where(seg_alias.code == segment_code)

    # Фильтр по доходу
    if min_income is not None:
        stmt = stmt.where(models.ClientState.income_pred >= min_income)
    if max_income is not None:
        stmt = stmt.where(models.ClientState.income_pred <= max_income)

    stmt = stmt.order_by(models.ClientState.updated_at.desc()).limit(limit).offset(offset)

    result = await db.execute(stmt)
    records = result.all()

    clients = [
        schemas.ClientSummary(
            external_id=row[0],
            display_name=row[1],
            income_pred=float(row[2]),
            segment_code=row[3],
            segment_name=row[4],
            updated_at=row[5],
        )
        for row in records
    ]
    return clients


@router.get("/clients/{external_id}/card", response_model=schemas.CardResponse)
async def get_client_card(
    external_id: str,
    db: AsyncSession = Depends(get_db),
) -> schemas.CardResponse:
    """Карточка клиента: прогноз дохода, сегмент, объяснение и рекомендованные продукты."""
    # Ищем клиента по external_id
    result = await db.execute(
        select(models.Client).where(models.Client.external_id == external_id)
    )
    client: models.Client | None = result.scalars().first()
    if not client:
        raise HTTPException(status_code=404, detail="Client not found")

    state = client.state
    if not state:
        raise HTTPException(status_code=404, detail="Client state not found")

    income_pred = float(state.income_pred)

    # Вычисляем сегмент через таблицу segments
    seg_result = await db.execute(
        select(models.Segment)
        .where(
            models.Segment.min_income <= income_pred,
            (models.Segment.max_income == None)  # noqa: E711
            | (income_pred < models.Segment.max_income),
        )
        .order_by(models.Segment.sort_order)
    )
    segment = seg_result.scalars().first()
    if not segment:
        raise HTTPException(status_code=400, detail="No segment defined for this income")

    # Берём последнее боевое предсказание (не симуляцию)
    pred_result = await db.execute(
        select(models.PredictionLog)
        .where(
            models.PredictionLog.client_id == client.id,
            models.PredictionLog.is_simulation.is_(False),
        )
        .order_by(models.PredictionLog.created_at.desc())
        .limit(1)
    )
    pred_log = pred_result.scalars().first()

    explanation_json: dict | None = pred_log.explanation_json if pred_log else None
    text_explanation: str | None = pred_log.text_explanation if pred_log else None

    # Если объяснения нет (например, лог ещё пустой) — считаем его на лету
    if not explanation_json:
        model = IncomeModel()
        _, explanation_json, text_explanation = await model.predict(state.features)

    explanation_json = explanation_json or {}
    text_explanation = text_explanation or ""

    # Схема объяснения для фронта
    exp = schemas.Explanation(
        baseline_income=explanation_json.get("baseline_income", 0.0),
        prediction=explanation_json.get("prediction", income_pred),
        top_features=[
            schemas.ExplanationItem(
                feature=item.get("feature"),
                title=item.get("title"),
                value=item.get("value"),
                shap_value=item.get("shap_value"),
            )
            for item in explanation_json.get("top_features", [])
        ],
        text=text_explanation,
    )

    # Динамически считаем рекомендации продуктов
    recs = await compute_recommendations(db, income_pred, state.features)
    rec_schemas = [
        schemas.ProductRecommendation(
            code=r["code"],
            name=r["name"],
            description=r.get("description"),
            reason=r["reason"],
        )
        for r in recs
    ]

    # Формируем ответ карточки
    card = schemas.CardResponse(
        client=schemas.CardClientInfo(
            external_id=client.external_id,
            display_name=client.display_name,
            created_at=client.created_at,
        ),
        prediction=schemas.CardPrediction(
            income_pred=income_pred,
            segment_code=segment.code,
            segment_name=segment.name,
            model_version=state.model_version,
            updated_at=state.updated_at,
        ),
        explanation=exp,
        products=rec_schemas,
    )
    return card


@router.post("/simulate", response_model=schemas.SimulationResponse)
async def simulate(
    payload: schemas.SimulationRequest,
    db: AsyncSession = Depends(get_db),
) -> schemas.SimulationResponse:
    """Симуляция: подставить альтернативные признаки и посмотреть результат.

    Поведение:
    - если передан ``external_id``, берём текущие признаки клиента как базу
      и поверх накладываем ``features_override``;
    - если ``external_id`` не передан, используем только ``features_override``.

    Результат:
    - новый прогноз дохода,
    - сегмент,
    - объяснение модели,
    - рекомендованные продукты.
    Также в ``prediction_log`` сохраняется запись с пометкой ``is_simulation = true``.
    """
    base_features: Dict[str, Any] = {}
    client_id: Optional[int] = None

    # Если указан клиент — подгружаем его текущее состояние
    if payload.external_id:
        result = await db.execute(
            select(models.Client).where(models.Client.external_id == payload.external_id)
        )
        client = result.scalars().first()
        if not client or not client.state:
            raise HTTPException(status_code=404, detail="Client state not found for simulation")
        base_features = client.state.features.copy()
        client_id = client.id

    # Применяем overrides
    overrides = payload.features_override or {}
    features: Dict[str, Any] = base_features.copy()
    for k, v in overrides.items():
        features[k] = v

    # Нормализуем признаки (числа, строки и т.п.)
    features = canonicalize_features(features)

    # Прогоняем через модель
    model = IncomeModel()
    income_pred, explanation_json, text_explanation = await model.predict(features)
    income_pred = float(income_pred)
    explanation_json = explanation_json or {}
    text_explanation = text_explanation or ""

    # Находим сегмент для смоделированного дохода
    seg_result = await db.execute(
        select(models.Segment)
        .where(
            models.Segment.min_income <= income_pred,
            (models.Segment.max_income == None)  # noqa: E711
            | (income_pred < models.Segment.max_income),
        )
        .order_by(models.Segment.sort_order)
    )
    segment = seg_result.scalars().first()
    if not segment:
        raise HTTPException(status_code=400, detail="No segment defined for this income")

    # Считаем рекомендации продуктов по смоделированным признакам
    recs = await compute_recommendations(db, income_pred, features)
    rec_schemas = [
        schemas.ProductRecommendation(
            code=r["code"],
            name=r["name"],
            description=r.get("description"),
            reason=r["reason"],
        )
        for r in recs
    ]

    # Логируем симуляцию в prediction_log
    sim_log = models.PredictionLog(
        client_id=client_id,
        is_simulation=True,
        features=features,
        income_pred=income_pred,
        explanation_json=explanation_json,
        text_explanation=text_explanation,
        recommendations=[r.model_dump() for r in rec_schemas],
        model_version="v1",
        request_source="frontend_simulator",
    )
    db.add(sim_log)
    await db.commit()

    # Формируем схемы для ответа
    exp_schema = schemas.Explanation(
        baseline_income=explanation_json.get("baseline_income", 0.0),
        prediction=explanation_json.get("prediction", income_pred),
        top_features=[
            schemas.ExplanationItem(
                feature=item.get("feature"),
                title=item.get("title"),
                value=item.get("value"),
                shap_value=item.get("shap_value"),
            )
            for item in explanation_json.get("top_features", [])
        ],
        text=text_explanation,
    )

    pred_schema = schemas.CardPrediction(
        income_pred=income_pred,
        segment_code=segment.code,
        segment_name=segment.name,
        model_version="v1",
        updated_at=datetime.datetime.now(datetime.UTC),
    )

    return schemas.SimulationResponse(
        prediction=pred_schema,
        explanation=exp_schema,
        products=rec_schemas,
    )