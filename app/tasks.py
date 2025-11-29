"""Фоновые задачи для долгих операций.

Модуль содержит асинхронные функции, которые могут выполняться в фоне,
например через BackgroundTasks FastAPI.

Основные задачи:
- обработка импорта CSV построчно,
- вычисление рекомендаций по продуктам для клиента.
"""

from __future__ import annotations

import asyncio
import csv
import datetime
from typing import Dict, Any, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload  # может не использоваться, но полезен

from .database import SessionLocal
from . import models
from .utils import canonicalize_features, compute_features_hash, sanitize_for_json
from .ml.model import IncomeModel


async def compute_recommendations(
    db: AsyncSession,
    income_pred: float,
    features: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Вычислить динамические рекомендации продуктов для клиента.

    Логика:
    1. Определяем сегмент клиента по его предсказанному доходу.
    2. Находим активные продукты, привязанные к этому сегменту и удовлетворяющие
       дополнительным порогам min_income/max_income продукта.
    3. Для каждого продукта проверяем набор правил по признакам клиента.
    4. Возвращаем список подходящих продуктов с текстовым объяснением причины.

    :param db: Активная сессия БД.
    :param income_pred: Предсказанный доход клиента.
    :param features: Нормализованный словарь признаков клиента.
    :return: Список словарей: code, name, description, reason.
    """
    # Определяем сегмент по доходу
    segment_row = await db.execute(
        select(models.Segment)
        .where(
            models.Segment.min_income <= income_pred,
            (models.Segment.max_income == None)  # noqa: E711
            | (income_pred < models.Segment.max_income),
        )
        .order_by(models.Segment.sort_order)
    )
    segment: models.Segment | None = segment_row.scalars().first()
    if not segment:
        return []

    # Ищем активные продукты, привязанные к сегменту и подходящие по порогам дохода
    prod_rows = await db.execute(
        select(models.Product)
        .join(models.ProductSegment)
        .where(
            models.Product.active.is_(True),
            models.ProductSegment.segment_id == segment.id,
            (models.Product.min_income == None)  # noqa: E711
            | (income_pred >= models.Product.min_income),
            (models.Product.max_income == None)  # noqa: E711
            | (income_pred <= models.Product.max_income),
        )
    )
    products: List[models.Product] = prod_rows.scalars().all()

    recommendations: List[Dict[str, Any]] = []

    for product in products:
        # Подгружаем правила продукта
        await db.refresh(product, attribute_names=["rules"])

        passes = True

        for rule in product.rules:
            # Подгружаем описание признака, чтобы знать тип и title
            await db.refresh(rule, attribute_names=["feature"])
            feature_def = rule.feature
            if feature_def is None:
                passes = False
                break

            feat_name = feature_def.name
            feat_value = features.get(feat_name)

            # Если признака нет в данных клиента — правило не выполняется
            if feat_value is None:
                passes = False
                break

            try:
                if feature_def.data_type in ("numeric", "number"):
                    # Числовое сравнение
                    feat_value_num = float(feat_value)
                    compare_value = float(rule.value_text)

                    if rule.operator == "eq" and not (feat_value_num == compare_value):
                        passes = False
                        break
                    elif rule.operator == "ne" and not (feat_value_num != compare_value):
                        passes = False
                        break
                    elif rule.operator == "lt" and not (feat_value_num < compare_value):
                        passes = False
                        break
                    elif rule.operator == "le" and not (feat_value_num <= compare_value):
                        passes = False
                        break
                    elif rule.operator == "gt" and not (feat_value_num > compare_value):
                        passes = False
                        break
                    elif rule.operator == "ge" and not (feat_value_num >= compare_value):
                        passes = False
                        break
                else:
                    # Строковое сравнение (case-insensitive)
                    feat_value_str = str(feat_value).strip().lower()
                    compare_value_str = rule.value_text.strip().lower()

                    if rule.operator == "eq" and not (feat_value_str == compare_value_str):
                        passes = False
                        break
                    elif rule.operator == "ne" and not (feat_value_str != compare_value_str):
                        passes = False
                        break
                    elif rule.operator in ("lt", "le", "gt", "ge"):
                        # Для нечисловых — лексикографическое сравнение
                        if rule.operator == "lt" and not (feat_value_str < compare_value_str):
                            passes = False
                            break
                        if rule.operator == "le" and not (feat_value_str <= compare_value_str):
                            passes = False
                            break
                        if rule.operator == "gt" and not (feat_value_str > compare_value_str):
                            passes = False
                            break
                        if rule.operator == "ge" and not (feat_value_str >= compare_value_str):
                            passes = False
                            break
            except Exception:
                # Любая ошибка при приведении типов — считаем, что правило не прошло
                passes = False
                break

        if passes:
            reason = (
                f"Сегмент {segment.code} и доход {income_pred:,.0f} ₽ "
                f"удовлетворяют условиям продукта"
            )
            recommendations.append(
                {
                    "code": product.code,
                    "name": product.name,
                    "description": product.description,
                    "reason": reason,
                }
            )

    # На всякий случай прогоняем через санитайзер перед сохранением в JSONB
    return sanitize_for_json(recommendations)  # type: ignore[return-value]


async def process_import_job(job_id: int) -> None:
    """Обработать задачу импорта CSV в фоне.

    Шаги:
    1. Загружаем запись ImportJob по ID
    2. Обновляем статус на ``running``
    3. Считаем общее число строк в файле (для прогресса)
    4. Читаем CSV построчно, нормализуем признаки, считаем хэш
    5. Если признаки клиента изменились:
       - вызываем модель,
       - пишем запись в prediction_log,
       - обновляем/создаём ClientState
    6. Обновляем счётчик обработанных строк
    7. По завершении помечаем задачу как ``completed`` или ``failed``
    """
    async with SessionLocal() as db:
        # Загружаем задачу
        job = await db.get(models.ImportJob, job_id)
        if not job:
            return

        # Обновляем статус на "running"
        job.status = "running"
        job.processed_rows = 0
        await db.commit()

        file_path = job.file_path

        # Пытаемся посчитать количество строк (для отображения прогресса)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                job.total_rows = sum(1 for _ in f) - 1
        except Exception:
            job.total_rows = None
        await db.commit()

        model = IncomeModel()

        try:
            # Read all rows synchronously first, outside of async context
            with open(file_path, "r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile, delimiter=";")
                rows = list(reader)

            # Now process rows asynchronously
            for row in rows:
                # Идентификатор клиента берём из external_id или client_id
                external_id = (
                    row.pop("external_id", None)
                    or row.pop("client_id", None)
                    or row.pop("id", None)  # <-- забираем id и не пускаем его в features
                )
                if not external_id:
                    job.processed_rows = (job.processed_rows or 0) + 1
                    await db.commit()
                    continue

                # Нормализуем признаки
                features_raw = {k: v for k, v in row.items() if k}
                features = canonicalize_features(features_raw)

                # Чистим NaN/Inf, чтобы Postgres принял JSONB
                features = sanitize_for_json(features)  # type: ignore[assignment]

                # Хэш считаем уже по очищенным признакам
                features_hash = compute_features_hash(features)

                # Получаем или создаём клиента
                client = await _get_or_create_client(db, external_id)

                # Явно подгружаем state, чтобы не было lazy-load в async-сессии
                await db.refresh(client, attribute_names=["state"])
                state = client.state

                state_changed = True
                if state and state.features_hash == features_hash:
                    # Признаки не изменились — можно пропустить перерасчёт
                    state_changed = False

                if state_changed:
                    # Считаем предсказание и объяснение
                    income_pred, explanation_json, text_explanation = await model.predict(
                        features
                    )

                    # Чистим explanation_json от NaN/Inf
                    explanation_json = sanitize_for_json(explanation_json)

                    # Считаем рекомендации (снимок на момент импорта, по желанию)
                    recs = await compute_recommendations(
                        db, float(income_pred), features
                    )

                    # Пишем лог предсказания
                    pred_log = models.PredictionLog(
                        client_id=client.id,
                        is_simulation=False,
                        features=features,
                        income_pred=income_pred,
                        explanation_json=explanation_json,
                        text_explanation=text_explanation,
                        recommendations=recs,
                        model_version="v1",
                        request_source="csv_import",
                    )
                    db.add(pred_log)

                    # Обновляем/создаём актуальное состояние клиента
                    if state is None:
                        new_state = models.ClientState(
                            client_id=client.id,
                            features=features,
                            features_hash=features_hash,
                            income_pred=income_pred,
                            model_version="v1",
                        )
                        db.add(new_state)
                    else:
                        state.features = features
                        state.features_hash = features_hash
                        state.income_pred = income_pred
                        state.model_version = "v1"
                        state.updated_at = datetime.datetime.now(datetime.UTC)

                    await db.flush()

                # Обновляем прогресс
                job.processed_rows = (job.processed_rows or 0) + 1
                await db.commit()

        except Exception as exc:
            # Откатываем текущую транзакцию, т.к. flush/commit уже могли упасть
            await db.rollback()

            # Фиксируем текст ошибки и статус failed
            job.status = "failed"
            job.error = str(exc)

            # Сохраняем статус задачи уже в новой транзакции
            await db.commit()
        else:
            # Успешное завершение
            job.status = "completed"
            job.finished_at = datetime.datetime.now(datetime.UTC)
            await db.commit()


async def _get_or_create_client(db: AsyncSession, external_id: str) -> models.Client:
    """Получить клиента по external_id или создать нового.

    :param db: Активная сессия БД
    :param external_id: Внешний идентификатор клиента из CSV/API
    :return: Объект Client, существующий или только что созданный
    """
    result = await db.execute(
        select(models.Client).where(models.Client.external_id == external_id)
    )
    client = result.scalars().first()
    if client:
        return client

    client = models.Client(external_id=external_id, display_name=external_id)
    db.add(client)
    await db.flush()
    return client