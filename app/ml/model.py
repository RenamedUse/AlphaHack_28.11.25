from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib
import pickle
import shap
import lightgbm as lgb
import xgboost as xgb

# Директория, где лежат артефакты модели по умолчанию
MODEL_DIR = Path(__file__).parent

_default_model_path = MODEL_DIR / "income_model.pkl"
_default_preproc_path = MODEL_DIR / "income_preprocessor.pkl"

# Пути можно переопределить через переменные окружения, если нужно
MODEL_PATH = Path(os.getenv("INCOME_MODEL_PATH", str(_default_model_path)))
PREPROCESSOR_PATH = Path(os.getenv("INCOME_PREPROCESSOR_PATH", str(_default_preproc_path)))


@lru_cache(maxsize=1)
def _load_model_and_preprocessor() -> Tuple[Any, List[str], Any]:
    """
    Ленивая загрузка модели и препроцессора.

    Ожидаемый формат preprocessor.pkl (рекомендуемый):

        {
            "preprocessor": <IncomePreprocessor>,
            "features": ["col1", "col2", ...]  # порядок колонок X из обучения
        }

    Если формат другой, выбрасываем осмысленную ошибку, чтобы
    не запускать модель на неправильном порядке признаков.
    """
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    if not PREPROCESSOR_PATH.exists():
        raise RuntimeError(f"Preprocessor file not found: {PREPROCESSOR_PATH}")

    # Модель XGBoost / LightGBM / любая sklearn-совместимая
    model = joblib.load(MODEL_PATH)

    with PREPROCESSOR_PATH.open("rb") as f:
        prep_obj = pickle.load(f)

    pre = None
    feature_names: List[str] = []

    if isinstance(prep_obj, dict):
        pre = prep_obj.get("preprocessor")
        feature_names = list(prep_obj.get("features") or [])
    else:
        # Старый/нестандартный формат — можно попытаться вытащить поля из объекта,
        # но корректность порядка колонок при этом не гарантируется.
        pre = prep_obj
        feature_names = list(getattr(pre, "features", []))

    if pre is None:
        raise RuntimeError("preprocessor artifact does not contain 'preprocessor'")

    if not feature_names:
        raise RuntimeError(
            "Preprocessor artifact does not contain 'features'. "
            "При обучении задайте pre.features = list(X.columns) и сохраните их в income_preprocessor.pkl"
        )

    return model, feature_names, pre


@lru_cache(maxsize=1)
def _build_explainer():
    """
    Создаёт SHAP TreeExplainer.

    Для простоты используем background из нулевого вектора
    нужной размерности. При желании можно заменить на
    небольшой сэмпл из train-данных и пересохранить артефакт.
    """
    model, feature_names, _ = _load_model_and_preprocessor()

    # background: одна "нулевая" строка размерности len(features)
    background = np.zeros((1, len(feature_names)), dtype=float)

    explainer = shap.TreeExplainer(model, background)

    expected_value = explainer.expected_value
    if isinstance(expected_value, (np.ndarray, list)):
        expected_value = float(np.array(expected_value).ravel()[0])
    else:
        expected_value = float(expected_value)

    return explainer, expected_value


class IncomeModel:
    """
    Продовая обёртка над моделью прогноза дохода с SHAP-объяснениями.

    ВАЖНО:
    - На вход подаём уже нормализованный словарь признаков (после canonicalize_features),
      но порядок и препроцессинг фич делаются через сохранённый IncomePreprocessor.
    """

    async def predict(self, features: Dict[str, Any]) -> Tuple[float, Dict[str, Any], str]:
        """
        :param features: словарь признаков клиента (после canonicalize_features)
                         ключи должны совпадать с исходными колонками train.csv
        :return:
            income_pred        – предсказанный доход
            explanation_json   – JSON-структура с топ-фичами и baseline
            text_explanation   – короткое текстовое объяснение для UI
        """
        model, feature_names, pre = _load_model_and_preprocessor()

        # raw_for_report: значения признаков до препроцессинга (для отображения)
        raw_for_report: Dict[str, Any] = {name: features.get(name) for name in feature_names}

        # Собираем DataFrame в том же наборе и порядке колонок, что был на обучении
        # (feature_names должен быть равен X.columns из обучения)
        row_df = pd.DataFrame([{name: raw_for_report.get(name) for name in feature_names}])

        # Применяем ровно тот же препроцессор, что использовался при обучении
        # (IncomePreprocessor.transform: числовые + OrdinalEncoder для категорий)
        X_row_df = pre.transform(row_df)
        # Список числовых значений признаков после препроцессинга (то, что реально видит модель)
        row_values = X_row_df.iloc[0].tolist()

        # Предсказание модели
        if isinstance(model, xgb.Booster):
            dmat = xgb.DMatrix(X_row_df, feature_names=feature_names)
            pred_arr = model.predict(dmat)
            pred = float(np.array(pred_arr).ravel()[0])
        elif isinstance(model, lgb.Booster):
            pred_arr = model.predict(X_row_df)
            pred = float(np.array(pred_arr).ravel()[0])
        else:
            pred_arr = model.predict(X_row_df)
            pred = float(np.array(pred_arr).ravel()[0])

        # SHAP-значения (если что-то пойдёт не так — не ломаем сервис)
        try:
            explainer, expected_value = _build_explainer()
            shap_vals_raw = explainer.shap_values(X_row_df)

            # Для регрессии в XGBoost/LightGBM shap_values может быть массивом или списком
            if isinstance(shap_vals_raw, list):
                shap_vals_raw = shap_vals_raw[0]
            shap_vals = np.array(shap_vals_raw)[0]
        except Exception:
            expected_value = pred
            shap_vals = np.zeros(len(feature_names), dtype=float)

        # Собираем вклад по фичам
        contrib_items: List[Dict[str, Any]] = []
        for name, shap_val, model_val in zip(feature_names, shap_vals, row_values):
            contrib_items.append(
                {
                    "feature": name,
                    "title": name,
                    "value": raw_for_report.get(name),
                    "model_value": model_val,
                    "shap_value": float(shap_val),
                }
            )

        # Топ-10 по модулю SHAP
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

        # Короткий текст для интерфейса
        if top_features:
            parts: List[str] = []
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