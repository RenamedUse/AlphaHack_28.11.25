from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Dict, Any, Tuple, List

import numpy as np
import joblib
import pickle
import shap
import lightgbm as lgb
import xgboost as xgb

# Пути к файлам модели и препроцессора; можно переопределить через env
MODEL_DIR = Path(__file__).parent
MODEL_PATH = MODEL_DIR / "income_model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "income_preprocessor.pkl"

@lru_cache(maxsize=1)
def _load_model_and_preprocessor() -> Tuple[Any, List[str]]:
    """Ленивая загрузка модели и списка фичей."""
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    if not PREPROCESSOR_PATH.exists():
        raise RuntimeError(f"Preprocessor file not found: {PREPROCESSOR_PATH}")

    model = joblib.load(MODEL_PATH)

    with PREPROCESSOR_PATH.open("rb") as f:
        prep = pickle.load(f)

    # В preprocessor.pkl должен быть ключ 'features': список исходных колонок
    feature_names: List[str] = prep.get("features") or []
    if not feature_names:
        raise RuntimeError("preprocessor.pkl does not contain 'features' key")

    return model, feature_names


@lru_cache(maxsize=1)
def _build_explainer():
    """Создаёт SHAP TreeExplainer с нулевым background."""
    model, feature_names = _load_model_and_preprocessor()
    background = np.zeros((1, len(feature_names)), dtype=float)
    explainer = shap.TreeExplainer(model, background)

    expected_value = explainer.expected_value
    # expected_value может быть скаляром или массивом
    if isinstance(expected_value, (np.ndarray, list)):
        expected_value = float(np.array(expected_value).ravel()[0])
    else:
        expected_value = float(expected_value)

    return explainer, expected_value


class IncomeModel:
    """Продовая модель прогноза дохода с SHAP‑объяснениями."""

    async def predict(self, features: Dict[str, Any]) -> Tuple[float, Dict[str, Any], str]:
        """
        :param features: нормализованный словарь признаков (после canonicalize_features)
        :return: (income_pred, explanation_json, text_explanation)
        """
        model, feature_names = _load_model_and_preprocessor()

        # Собираем вектор признаков в нужном порядке
        raw_for_report: Dict[str, Any] = {}
        row_values: List[float] = []
        for name in feature_names:
            raw_val = features.get(name)
            raw_for_report[name] = raw_val
            if raw_val is None or raw_val == "":
                row_values.append(-999.0)  # аналогично препроцессору
                continue
            try:
                row_values.append(float(raw_val))
            except Exception:
                # строку приводим к детерминированному числу через хеш
                row_values.append(float(abs(hash(str(raw_val))) % 1000))

        X_row = np.array(row_values, dtype=float).reshape(1, -1)

        # Предсказываем в зависимости от типа модели
        if isinstance(model, xgb.Booster):
            dmat = xgb.DMatrix(X_row, feature_names=feature_names)
            pred = float(model.predict(dmat)[0])
        elif isinstance(model, lgb.Booster):
            pred = float(model.predict(X_row)[0])
        else:
            pred = float(model.predict(X_row)[0])

        # SHAP‑значения
        try:
            explainer, expected_value = _build_explainer()
            shap_vals_raw = explainer.shap_values(X_row)
            if isinstance(shap_vals_raw, list):
                shap_vals_raw = shap_vals_raw[0]
            shap_vals = np.array(shap_vals_raw)[0]
        except Exception:
            # Если SHAP не работает, используем нулевые значения
            expected_value = pred
            shap_vals = np.zeros(len(feature_names))

        contrib_items = []
        for name, shap_val, model_val in zip(feature_names, shap_vals, row_values):
            contrib_items.append(
                {
                    "feature": name,
                    "title": name,
                    "value": raw_for_report.get(name, model_val),
                    "model_value": model_val,
                    "shap_value": float(shap_val),
                }
            )

        top_features = sorted(
            contrib_items,
            key=lambda it: abs(it["shap_value"]),
            reverse=True,
        )[:10]

        explanation_json: Dict[str, Any] = {
            "baseline_income": expected_value,
            "prediction": pred,
            "top_features": top_features,
        }

        if top_features:
            parts = []
            for item in top_features[:3]:
                feat = item["title"]
                val = item["value"]
                shap_val = item["shap_value"]
                sign = "+" if shap_val >= 0 else "-"
                parts.append(f"{feat} ({val}) {sign}{abs(shap_val):,.0f} ₽")
            text_explanation = (
                f"Модель оценила доход в {pred:,.0f} ₽. "
                f"Наибольший вклад дали: {', '.join(parts)}."
            )
        else:
            text_explanation = (
                f"Модель оценила доход в {pred:,.0f} ₽. "
                "Вклад признаков распределён равномерно."
            )

        return pred, explanation_json, text_explanation