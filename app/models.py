"""Определения ORM-моделей

Модуль содержит SQLAlchemy ORM-модели, соответствующие схеме БД
Все модели наследуются от базового класса ``Base`` из ``database.py``.
"""

from __future__ import annotations

import datetime
from typing import Optional, List

from sqlalchemy import (
    String,
    Text,
    DateTime,
    Boolean,
    Integer,
    BigInteger,
    Numeric,
    ForeignKey,
    UniqueConstraint,
)

# Унифицированный тип для JSON-полей в БД:
# - в PostgreSQL используем JSONB
# - в других СУБД — обобщённый JSON SQLAlchemy
try:
    from sqlalchemy.dialects.postgresql import JSONB
    JSONType = JSONB
except ImportError:
    from sqlalchemy import JSON
    JSONType = JSON

from sqlalchemy.orm import relationship, Mapped, mapped_column

from .database import Base


class Segment(Base):
    """Сегмент клиента по уровню дохода (low/middle/high/premium и т.п)"""

    __tablename__ = "segments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    code: Mapped[str] = mapped_column(String(50), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    min_income: Mapped[float] = mapped_column(Numeric, nullable=False)
    max_income: Mapped[Optional[float]] = mapped_column(Numeric, nullable=True)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.datetime.now(datetime.UTC),
    )

    # Связь с таблицей связей продукт - сегмент
    product_segments: Mapped[List["ProductSegment"]] = relationship(
        back_populates="segment",
        cascade="all, delete-orphan",
    )


class Client(Base):
    """Клиент (идентифицируется external_id из внешней системы/CSV)"""

    __tablename__ = "clients"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    external_id: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    display_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.datetime.now(datetime.UTC),
    )

    # Текущее состояние (последний рассчитанный доход и признаки)
    state: Mapped[Optional["ClientState"]] = relationship(
        back_populates="client",
        uselist=False,
        cascade="all, delete-orphan",
    )
    # История всех предсказаний (боевых и симуляций)
    predictions: Mapped[List["PredictionLog"]] = relationship(
        back_populates="client",
        cascade="all, delete-orphan",
    )


class ClientState(Base):
    """Актуальное состояние клиента: признаки + последний прогноз дохода"""

    __tablename__ = "client_state"

    client_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("clients.id", ondelete="CASCADE"),
        primary_key=True,
    )
    # Сырые признаки клиента в виде JSON
    features: Mapped[dict] = mapped_column(JSONType, nullable=False)
    # Хэш нормализованных признаков (для детекта изменений)
    features_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    income_pred: Mapped[float] = mapped_column(Numeric, nullable=False)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.datetime.now(datetime.UTC),
    )

    client: Mapped["Client"] = relationship(
        back_populates="state",
    )


class PredictionLog(Base):
    """Журнал всех предсказаний модели (боевых и симуляций)"""

    __tablename__ = "prediction_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    client_id: Mapped[Optional[int]] = mapped_column(
        BigInteger,
        ForeignKey("clients.id", ondelete="SET NULL"),
        nullable=True,
    )
    is_simulation: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Набор признаков, который реально ушёл в модель
    features: Mapped[dict] = mapped_column(JSONType, nullable=False)
    income_pred: Mapped[float] = mapped_column(Numeric, nullable=False)
    # Объяснение работы модели в виде JSON (например, SHAP-значения)
    explanation_json: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    # Текстовое объяснение для человека
    text_explanation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Слепок рекомендованных продуктов для этого предсказания (опционально)
    recommendations: Mapped[Optional[dict]] = mapped_column(JSONType, nullable=True)
    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    request_source: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.datetime.now(datetime.UTC),
    )

    client: Mapped[Optional["Client"]] = relationship(
        back_populates="predictions",
    )


class Product(Base):
    """Финансовый продукт, который можно рекомендовать клиенту"""

    __tablename__ = "products"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    code: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    # Дополнительные пороги по доходу (помимо сегментов)
    min_income: Mapped[Optional[float]] = mapped_column(Numeric, nullable=True)
    max_income: Mapped[Optional[float]] = mapped_column(Numeric, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.datetime.now(datetime.UTC),
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.datetime.now(datetime.UTC),
        onupdate=lambda: datetime.datetime.now(datetime.UTC),
    )

    # Связь продукт - сегменты
    product_segments: Mapped[List["ProductSegment"]] = relationship(
        back_populates="product",
        cascade="all, delete-orphan",
    )
    # Правила отбора по признакам клиента
    rules: Mapped[List["ProductFeatureRule"]] = relationship(
        back_populates="product",
        cascade="all, delete-orphan",
    )


class ProductSegment(Base):
    """Связь продукта с сегментами (many-to-many)"""

    __tablename__ = "product_segments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    product_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
    )
    segment_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("segments.id", ondelete="CASCADE"),
        nullable=False,
    )

    product: Mapped["Product"] = relationship(
        back_populates="product_segments",
    )
    segment: Mapped["Segment"] = relationship(
        back_populates="product_segments",
    )

    __table_args__ = (
        UniqueConstraint("product_id", "segment_id", name="uq_product_segment"),
    )


class FeatureDefinition(Base):
    """Справочник признаков, по которым можно строить правила к продуктам"""

    __tablename__ = "feature_definitions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    # Ключ в JSON-объекте с признаками клиента (например, "age", "region")
    name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    # Человекочитаемое название признака
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Тип данных признака (numeric/text/boolean/date/categorical)
    data_type: Mapped[str] = mapped_column(String(50), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.datetime.now(datetime.UTC),
    )

    rules: Mapped[List["ProductFeatureRule"]] = relationship(
        back_populates="feature",
        cascade="all, delete-orphan",
    )


class ProductFeatureRule(Base):
    """Правило отбора продукта по одному признаку клиента"""

    __tablename__ = "product_feature_rules"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    product_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("products.id", ondelete="CASCADE"),
        nullable=False,
    )
    feature_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("feature_definitions.id", ondelete="CASCADE"),
        nullable=False,
    )
    # Оператор сравнения: eq/ne/lt/le/gt/ge
    operator: Mapped[str] = mapped_column(String(10), nullable=False)
    # Значение для сравнения в виде строки, парсится по data_type признака
    value_text: Mapped[str] = mapped_column(String(255), nullable=False)

    product: Mapped["Product"] = relationship(
        back_populates="rules",
    )
    feature: Mapped["FeatureDefinition"] = relationship(
        back_populates="rules",
    )

    __table_args__ = (
        UniqueConstraint(
            "product_id",
            "feature_id",
            "operator",
            "value_text",
            name="uq_product_feature_rule",
        ),
    )


class ImportJob(Base):
    """Задание на импорт CSV (фоновая обработка)"""

    __tablename__ = "import_jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, index=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    total_rows: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    processed_rows: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.datetime.now(datetime.UTC),
    )
    finished_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )