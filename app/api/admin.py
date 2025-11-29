"""Административные маршруты (управление продуктами, сегментами, признаками и правилами)."""

from __future__ import annotations

import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db
from .. import models, schemas
from ..seed_data import seed_initial_data

router = APIRouter(prefix="/api/admin", tags=["admin"])

@router.post("/seed/start")
async def run_seeds(db: AsyncSession = Depends(get_db)) -> dict:
    """Запустить сидирование сегментов, признаков, продуктов и правил.

    Можно дернуть один раз после старта приложения или при развёртывании стенда.
    Эндпоинт идемпотентный: повторный вызов не создаёт дубликатов.
    """
    await seed_initial_data(db)
    return {"status": "ok"}

@router.get("/products", response_model=List[schemas.ProductResponse])
async def list_products(db: AsyncSession = Depends(get_db)) -> List[schemas.ProductResponse]:
    """Список всех продуктов."""
    result = await db.execute(select(models.Product))
    products = result.scalars().all()
    return [schemas.ProductResponse.model_validate(p) for p in products]


@router.post("/products", response_model=schemas.ProductResponse)
async def create_product(
    payload: schemas.ProductCreate,
    db: AsyncSession = Depends(get_db),
) -> schemas.ProductResponse:
    """Создать новый продукт."""
    # Проверяем уникальность кода
    result = await db.execute(
        select(models.Product).where(models.Product.code == payload.code)
    )
    existing = result.scalars().first()
    if existing:
        raise HTTPException(status_code=400, detail="Product code already exists")

    product = models.Product(
        code=payload.code,
        name=payload.name,
        description=payload.description,
        active=payload.active,
        min_income=payload.min_income,
        max_income=payload.max_income,
    )
    db.add(product)
    await db.commit()
    await db.refresh(product)
    return schemas.ProductResponse.model_validate(product)


@router.patch("/products/{product_id}", response_model=schemas.ProductResponse)
async def update_product(
    product_id: int,
    payload: schemas.ProductUpdate,
    db: AsyncSession = Depends(get_db),
) -> schemas.ProductResponse:
    """Обновить параметры продукта (название, описание, активность, пороги дохода)."""
    product = await db.get(models.Product, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    update_data = payload.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(product, key, value)

    product.updated_at = datetime.datetime.now(datetime.UTC)

    await db.commit()
    await db.refresh(product)
    return schemas.ProductResponse.model_validate(product)


@router.get("/features", response_model=List[schemas.FeatureDefinitionResponse])
async def list_features(
    db: AsyncSession = Depends(get_db),
) -> List[schemas.FeatureDefinitionResponse]:
    """Список всех определений признаков, по которым можно задавать правила продуктов."""
    result = await db.execute(select(models.FeatureDefinition))
    features = result.scalars().all()
    return [schemas.FeatureDefinitionResponse.model_validate(f) for f in features]


@router.post("/features", response_model=schemas.FeatureDefinitionResponse)
async def create_feature(
    payload: schemas.FeatureDefinitionCreate,
    db: AsyncSession = Depends(get_db),
) -> schemas.FeatureDefinitionResponse:
    """Создать новое определение признака."""
    # Проверка уникальности имени признака
    result = await db.execute(
        select(models.FeatureDefinition).where(
            models.FeatureDefinition.name == payload.name
        )
    )
    existing = result.scalars().first()
    if existing:
        raise HTTPException(status_code=400, detail="Feature name already exists")

    feat = models.FeatureDefinition(
        name=payload.name,
        title=payload.title,
        description=payload.description,
        data_type=payload.data_type,
    )
    db.add(feat)
    await db.commit()
    await db.refresh(feat)
    return schemas.FeatureDefinitionResponse.model_validate(feat)


@router.get("/segments", response_model=List[schemas.SegmentResponse])
async def list_segments(
    db: AsyncSession = Depends(get_db),
) -> List[schemas.SegmentResponse]:
    """Список всех сегментов клиентов."""
    result = await db.execute(select(models.Segment).order_by(models.Segment.sort_order))
    segments = result.scalars().all()
    return [schemas.SegmentResponse.model_validate(s) for s in segments]


@router.post("/segments", response_model=schemas.SegmentResponse)
async def create_segment(
    payload: schemas.SegmentCreate,
    db: AsyncSession = Depends(get_db),
) -> schemas.SegmentResponse:
    """Создать новый сегмент клиентов."""
    # Проверка уникальности кода сегмента
    result = await db.execute(
        select(models.Segment).where(models.Segment.code == payload.code)
    )
    existing = result.scalars().first()
    if existing:
        raise HTTPException(status_code=400, detail="Segment code already exists")

    seg = models.Segment(
        code=payload.code,
        name=payload.name,
        description=payload.description,
        min_income=payload.min_income,
        max_income=payload.max_income,
        sort_order=payload.sort_order,
    )
    db.add(seg)
    await db.commit()
    await db.refresh(seg)
    return schemas.SegmentResponse.model_validate(seg)


@router.get(
    "/products/{product_id}/segments",
    response_model=List[schemas.SegmentResponse],
)
async def get_product_segments(
    product_id: int,
    db: AsyncSession = Depends(get_db),
) -> List[schemas.SegmentResponse]:
    """Получить список сегментов, привязанных к продукту."""
    product = await db.get(models.Product, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Явно выбираем сегменты через JOIN, без ленивой загрузки отношений
    result = await db.execute(
        select(models.Segment)
        .join(
            models.ProductSegment,
            models.ProductSegment.segment_id == models.Segment.id,
        )
        .where(models.ProductSegment.product_id == product_id)
        .order_by(models.Segment.sort_order)
    )
    segments = result.scalars().all()

    return [schemas.SegmentResponse.model_validate(s) for s in segments]


@router.post(
    "/products/{product_id}/segments",
    response_model=List[schemas.SegmentResponse],
)
async def update_product_segments(
    product_id: int,
    payload: schemas.ProductSegmentUpdate,
    db: AsyncSession = Depends(get_db),
) -> List[schemas.SegmentResponse]:
    """Обновить список сегментов, к которым привязан продукт."""
    product = await db.get(models.Product, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Находим сегменты по их кодам
    segs_result = await db.execute(
        select(models.Segment).where(models.Segment.code.in_(payload.segment_codes))
    )
    segments = segs_result.scalars().all()

    # Проверяем, что все переданные коды существуют
    if len(segments) != len(payload.segment_codes):
        raise HTTPException(status_code=400, detail="One or more segment codes are invalid")

    # Удаляем старые связи продукт ↔ сегмент
    await db.refresh(product, attribute_names=["product_segments"])
    for ps in list(product.product_segments):
        await db.delete(ps)

    # Создаём новые связи
    for seg in segments:
        new_link = models.ProductSegment(product_id=product.id, segment_id=seg.id)
        db.add(new_link)

    await db.commit()

    # Возвращаем актуальный список сегментов для продукта (мы его уже знаем)
    return [schemas.SegmentResponse.model_validate(seg) for seg in segments]


@router.get(
    "/products/{product_id}/rules", response_model=List[schemas.ProductRuleResponse]
)
async def get_product_rules(
    product_id: int,
    db: AsyncSession = Depends(get_db),
) -> List[schemas.ProductRuleResponse]:
    """Получить список правил отбора клиентов для продукта."""
    product = await db.get(models.Product, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    await db.refresh(product, attribute_names=["rules"])
    rules = product.rules

    # Гарантируем, что у правил подгружены данные признаков
    for r in rules:
        await db.refresh(r, attribute_names=["feature"])

    return [
        schemas.ProductRuleResponse(
            id=r.id,
            product_id=r.product_id,
            feature_id=r.feature_id,
            operator=r.operator,
            value_text=r.value_text,
            feature=schemas.FeatureDefinitionResponse.model_validate(r.feature),
        )
        for r in rules
    ]


@router.post(
    "/products/{product_id}/rules", response_model=schemas.ProductRuleResponse
)
async def add_product_rule(
    product_id: int,
    payload: schemas.ProductRuleCreate,
    db: AsyncSession = Depends(get_db),
) -> schemas.ProductRuleResponse:
    """Добавить новое правило отбора клиентов для продукта."""
    product = await db.get(models.Product, product_id)
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Находим описание признака по имени
    feat_result = await db.execute(
        select(models.FeatureDefinition).where(
            models.FeatureDefinition.name == payload.feature_name
        )
    )
    feature = feat_result.scalars().first()
    if not feature:
        raise HTTPException(status_code=404, detail="Feature definition not found")

    # Проверяем, что такого правила ещё нет
    dup_result = await db.execute(
        select(models.ProductFeatureRule).where(
            models.ProductFeatureRule.product_id == product.id,
            models.ProductFeatureRule.feature_id == feature.id,
            models.ProductFeatureRule.operator == payload.operator,
            models.ProductFeatureRule.value_text == payload.value_text,
        )
    )
    duplicate = dup_result.scalars().first()
    if duplicate:
        raise HTTPException(status_code=400, detail="Rule already exists")

    rule = models.ProductFeatureRule(
        product_id=product.id,
        feature_id=feature.id,
        operator=payload.operator,
        value_text=payload.value_text,
    )
    db.add(rule)
    await db.commit()
    await db.refresh(rule, attribute_names=["feature"])

    return schemas.ProductRuleResponse(
        id=rule.id,
        product_id=rule.product_id,
        feature_id=rule.feature_id,
        operator=rule.operator,
        value_text=rule.value_text,
        feature=schemas.FeatureDefinitionResponse.model_validate(rule.feature),
    )