"""Pydantic-схемы для запросов и ответов API"""

from __future__ import annotations

import datetime
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, ConfigDict


class ORMModel(BaseModel):
    """Базовая модель для всех схем"""

    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=(),
    )


class ImportJobResponse(ORMModel):
    id: int
    status: str
    total_rows: Optional[int] = None
    processed_rows: Optional[int] = None
    error: Optional[str] = None
    created_at: datetime.datetime
    finished_at: Optional[datetime.datetime] = None


class ClientSummary(ORMModel):
    external_id: str
    display_name: Optional[str] = None
    income_pred: float
    segment_code: str
    segment_name: str
    updated_at: datetime.datetime


class ExplanationItem(ORMModel):
    feature: str
    title: str
    value: Any
    shap_value: float


class Explanation(ORMModel):
    baseline_income: float
    prediction: float
    top_features: List[ExplanationItem]
    text: str


class ProductRecommendation(ORMModel):
    code: str
    name: str
    description: Optional[str]
    reason: str


class CardClientInfo(ORMModel):
    external_id: str
    display_name: Optional[str] = None
    created_at: datetime.datetime


class CardPrediction(ORMModel):
    income_pred: float
    segment_code: str
    segment_name: str
    model_version: str
    updated_at: datetime.datetime


class CardResponse(ORMModel):
    client: CardClientInfo
    prediction: CardPrediction
    explanation: Explanation
    products: List[ProductRecommendation]


class SimulationRequest(ORMModel):
    external_id: Optional[str] = Field(
        None,
        description=(
            "ID клиента, от которого строим симуляцию. "
            "Если не указан — используем только features_override."
        ),
    )
    features_override: Dict[str, Any] = Field(
        ...,
        description="Частичный или полный словарь признаков для симуляции.",
    )


class SimulationResponse(ORMModel):
    prediction: CardPrediction
    explanation: Explanation
    products: List[ProductRecommendation]


class ProductBase(ORMModel):
    code: str
    name: str
    description: Optional[str] = None
    min_income: Optional[float] = None
    max_income: Optional[float] = None


class ProductCreate(ProductBase):
    active: bool = True


class ProductUpdate(ORMModel):
    name: Optional[str] = None
    description: Optional[str] = None
    min_income: Optional[float] = None
    max_income: Optional[float] = None
    active: Optional[bool] = None


class ProductResponse(ProductBase):
    id: int
    active: bool
    created_at: datetime.datetime
    updated_at: datetime.datetime


class FeatureDefinitionBase(ORMModel):
    name: str
    title: str
    description: Optional[str] = None
    data_type: str


class FeatureDefinitionCreate(FeatureDefinitionBase):
    pass


class FeatureDefinitionResponse(FeatureDefinitionBase):
    id: int
    created_at: datetime.datetime


class SegmentBase(ORMModel):
    code: str
    name: str
    description: Optional[str] = None
    min_income: float
    max_income: Optional[float] = None
    sort_order: int = 0


class SegmentCreate(SegmentBase):
    pass


class SegmentResponse(SegmentBase):
    id: int
    created_at: datetime.datetime


class ProductSegmentUpdate(ORMModel):
    segment_codes: List[str] = Field(
        ...,
        description="Список кодов сегментов, к которым должен относиться продукт.",
    )


class ProductRuleCreate(ORMModel):
    feature_name: str
    operator: str
    value_text: str


class ProductRuleResponse(ORMModel):
    id: int
    product_id: int
    feature_id: int
    operator: str
    value_text: str
    feature: FeatureDefinitionResponse
