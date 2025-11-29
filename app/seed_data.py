from __future__ import annotations

import csv
import os
import datetime
from typing import Optional, Dict, Any, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from . import models

FEATURES_DESCRIPTION_PATH = os.environ.get(
    "FEATURES_DESCRIPTION_PATH",
    "/data/features_description.csv",
)


async def seed_segments(db: AsyncSession) -> None:
    """Посеять базовые сегменты клиентов по уровню дохода."""
    segments_def = [
        {
            "code": "low",
            "name": "Низкий доход",
            "description": "Доход ниже базового уровня",
            "min_income": 0.0,
            "max_income": 50_000.0,
            "sort_order": 1,
        },
        {
            "code": "middle",
            "name": "Средний доход",
            "description": "Доход в диапазоне массового рынка",
            "min_income": 50_000.0,
            "max_income": 120_000.0,
            "sort_order": 2,
        },
        {
            "code": "high",
            "name": "Высокий доход",
            "description": "Доход выше среднего",
            "min_income": 120_000.0,
            "max_income": 300_000.0,
            "sort_order": 3,
        },
        {
            "code": "premium",
            "name": "Премиальный доход",
            "description": "Состоятельные клиенты",
            "min_income": 300_000.0,
            "max_income": None,
            "sort_order": 4,
        },
    ]

    for seg in segments_def:
        existing = await db.execute(
            select(models.Segment).where(models.Segment.code == seg["code"])
        )
        if existing.scalars().first():
            continue

        obj = models.Segment(
            code=seg["code"],
            name=seg["name"],
            description=seg["description"],
            min_income=seg["min_income"],
            max_income=seg["max_income"],
            sort_order=seg["sort_order"],
        )
        db.add(obj)

    await db.commit()


def _detect_feature_columns(fieldnames: List[str]) -> Dict[str, Optional[str]]:
    """Определить, какие столбцы в CSV соответствуют имени/описанию/типу."""
    lower = [f.lower() for f in fieldnames]

    def find(*candidates: str) -> Optional[str]:
        for cand in candidates:
            for orig, low in zip(fieldnames, lower):
                if cand in low:
                    return orig
        return None

    return {
        "name": find("feature", "name", "признак", "feature_name"),
        "title": find("title", "display", "rus", "описание"),
        "description": find("description", "описание", "comment"),
        "dtype": find("type", "dtype", "format"),
    }


def _infer_data_type(name: str, raw_type: Optional[str]) -> str:
    """Определить data_type ('numeric' или 'string') для FeatureDefinition."""
    if raw_type:
        t = raw_type.strip().lower()
        if any(x in t for x in ("int", "float", "double", "numeric", "num")):
            return "numeric"
        if any(x in t for x in ("str", "cat", "enum", "category")):
            return "string"

    n = name.lower()
    if n in ("gender", "sex", "region", "city", "education", "marital_status"):
        return "string"
    if n.endswith("_flg") or n.startswith("flg_"):
        return "numeric"

    # По умолчанию считаем числом, так как у нас скоринговый датасет
    return "numeric"


async def seed_feature_definitions(db: AsyncSession) -> None:
    """Посеять FeatureDefinition из файла features_description.csv."""
    if not os.path.exists(FEATURES_DESCRIPTION_PATH):
        # Ничего страшного: просто не сидируем признаки
        return

    with open(FEATURES_DESCRIPTION_PATH, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        except csv.Error:
            dialect = csv.excel

        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            return

        cols = _detect_feature_columns(reader.fieldnames)

        for row in reader:
            name = (row.get(cols["name"]) or "").strip()
            if not name:
                continue

            existing = await db.execute(
                select(models.FeatureDefinition).where(
                    models.FeatureDefinition.name == name
                )
            )
            if existing.scalars().first():
                continue

            title = (row.get(cols["title"]) or name).strip()
            description = (row.get(cols["description"]) or "").strip() or None
            raw_type = row.get(cols["dtype"]) if cols["dtype"] else None
            data_type = _infer_data_type(name, raw_type)

            feat = models.FeatureDefinition(
                name=name,
                title=title,
                description=description,
                data_type=data_type,
            )
            db.add(feat)

    await db.commit()


async def _get_segment_map(db: AsyncSession) -> Dict[str, models.Segment]:
    rows = await db.execute(select(models.Segment))
    segs = rows.scalars().all()
    return {s.code: s for s in segs}


async def _get_feature_by_name(db: AsyncSession, name: str) -> Optional[models.FeatureDefinition]:
    result = await db.execute(
        select(models.FeatureDefinition).where(models.FeatureDefinition.name == name)
    )
    return result.scalars().first()


async def seed_products_and_rules(db: AsyncSession) -> None:
    """Посеять банковские продукты и правила отбора клиентов."""

    seg_map = await _get_segment_map(db)

    products_def: List[Dict[str, Any]] = [
        {
            "code": "DEBIT_CARD_EVERYDAY",
            "name": "Дебетовая карта «Повседневная»",
            "description": "Базовая карта без платы за обслуживание.",
            "min_income": 0.0,
            "max_income": 120_000.0,
            "segment_codes": ["low", "middle"],
            "rules": [
                {"feature": "age", "operator": "ge", "value_text": "18"},
            ],
        },
        {
            "code": "CREDIT_CARD_TRAVEL",
            "name": "Кредитная карта «Путешествия»",
            "description": "Кредитка с бонусами за оплату путешествий.",
            "min_income": 80_000.0,
            "max_income": 250_000.0,
            "segment_codes": ["middle", "high"],
            "rules": [
                {"feature": "age", "operator": "ge", "value_text": "21"},
                # Пример правила по гендеру, если такой признак есть
                {"feature": "gender", "operator": "ne", "value_text": "unknown"},
            ],
        },
        {
            "code": "PREMIUM_PACKAGE",
            "name": "Премиальный пакет услуг",
            "description": "Персональный менеджер, повышенный кешбэк, премиальные продукты.",
            "min_income": 250_000.0,
            "max_income": None,
            "segment_codes": ["high", "premium"],
            "rules": [
                {"feature": "age", "operator": "ge", "value_text": "25"},
            ],
        },
    ]

    for pdata in products_def:
        # 1. Продукт
        result = await db.execute(
            select(models.Product).where(models.Product.code == pdata["code"])
        )
        product = result.scalars().first()
        if not product:
            product = models.Product(
                code=pdata["code"],
                name=pdata["name"],
                description=pdata["description"],
                active=True,
                min_income=pdata["min_income"],
                max_income=pdata["max_income"],
            )
            db.add(product)
            await db.flush()

        # 2. Связи с сегментами
        for seg_code in pdata["segment_codes"]:
            seg = seg_map.get(seg_code)
            if not seg:
                continue

            # проверяем, есть ли уже такая связь
            link_q = await db.execute(
                select(models.ProductSegment).where(
                    models.ProductSegment.product_id == product.id,
                    models.ProductSegment.segment_id == seg.id,
                )
            )
            if link_q.scalars().first():
                continue

            link = models.ProductSegment(product_id=product.id, segment_id=seg.id)
            db.add(link)

        # 3. Правила по признакам
        for rule_def in pdata["rules"]:
            feat = await _get_feature_by_name(db, rule_def["feature"])
            if not feat:
                # Если в features_description нет такого признака — правило просто не создаём
                continue

            dup_q = await db.execute(
                select(models.ProductFeatureRule).where(
                    models.ProductFeatureRule.product_id == product.id,
                    models.ProductFeatureRule.feature_id == feat.id,
                    models.ProductFeatureRule.operator == rule_def["operator"],
                    models.ProductFeatureRule.value_text == rule_def["value_text"],
                )
            )
            if dup_q.scalars().first():
                continue

            rule = models.ProductFeatureRule(
                product_id=product.id,
                feature_id=feat.id,
                operator=rule_def["operator"],
                value_text=rule_def["value_text"],
            )
            db.add(rule)

    await db.commit()


async def seed_initial_data(db: AsyncSession) -> None:
    """Запустить полный набор сидов: сегменты, признаки, продукты и правила."""
    await seed_segments(db)
    await seed_feature_definitions(db)
    await seed_products_and_rules(db)