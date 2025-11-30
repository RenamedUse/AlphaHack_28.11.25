"""
Income prediction model integration with ensemble model support.

This module provides an asynchronous `IncomeModel` class that can be used
to generate income predictions and SHAP explanations for a given set of
client features.  It transparently supports two types of persisted
models:

* A simple pickled model (e.g. LightGBM or XGBoost) saved at
  ``income_prediction_model.pkl`` alongside a corresponding
  ``preprocessor.pkl`` that defines the expected feature order.  This
  format was used in earlier versions of the project.

* A bundled ensemble model saved at ``income_ensemble_model.pkl`` that
  contains multiple base models (LightGBM, CatBoost and a neural
  network), a meta model (XGBoost) and all of the artefacts required
  for preprocessing (scaler, medians, categorical mappings, PCA
  transformer, etc.).  A corresponding ``shap_explainer.pkl``
  accompanies this bundle and should be used for SHAP explanations.  The
  ``preprocessor.pkl`` for the ensemble still contains the ordered list
  of raw feature names, which is used to populate missing features with
  zeros.

The integration logic automatically detects which format is present and
adapts the prediction and explanation pipeline accordingly.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import joblib
import pickle
try:
    import shap  # type: ignore[import]
except Exception:
    # SHAP is optional.  If it's not available, explanations will be
    # degraded and zero contributions will be returned.
    shap = None  # type: ignore[assignment]
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

try:
    import lightgbm as lgb  # noqa: F401
except Exception:
    # LightGBM is optional; it will only be used if present in the model
    lgb = None  # type: ignore

try:
    import xgboost as xgb  # noqa: F401
except Exception:
    xgb = None  # type: ignore


# Paths to model artefacts; can be overridden via environment variables.
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = THIS_DIR / "income_prediction_model.pkl"
DEFAULT_ENSEMBLE_PATH = THIS_DIR / "income_ensemble_model.pkl"
DEFAULT_EXPLAINER_PATH = THIS_DIR / "shap_explainer.pkl"
DEFAULT_PREPROCESSOR_PATH = THIS_DIR / "preprocessor.pkl"


@lru_cache(maxsize=1)
def _load_model_and_preprocessor() -> Tuple[Any, List[str]]:
    """Load the persisted model/bundle and the list of feature names.

    This function tries to load the newer ensemble bundle first.  If
    ``income_ensemble_model.pkl`` exists in the model directory, it
    returns that bundle along with the feature list from ``preprocessor.pkl``.
    Otherwise it falls back to the legacy single-model format and
    returns the model object and feature list.

    :returns: tuple of (model_or_bundle, feature_names)
    :raises RuntimeError: if neither model file is found or the
        preprocessor is missing/invalid.
    """
    # Determine which model file to use
    model_path: Path
    if DEFAULT_ENSEMBLE_PATH.exists():
        model_path = DEFAULT_ENSEMBLE_PATH
    elif DEFAULT_MODEL_PATH.exists():
        model_path = DEFAULT_MODEL_PATH
    else:
        raise RuntimeError(
            f"No model file found. Expected either {DEFAULT_ENSEMBLE_PATH} "
            f"or {DEFAULT_MODEL_PATH}."
        )

    # Load the model or model bundle
    model_obj = joblib.load(model_path)

    # Load the list of feature names from the preprocessor
    if not DEFAULT_PREPROCESSOR_PATH.exists():
        raise RuntimeError(f"Preprocessor file not found: {DEFAULT_PREPROCESSOR_PATH}")
    with DEFAULT_PREPROCESSOR_PATH.open("rb") as f:
        prep = pickle.load(f)
    feature_names: List[str] = prep.get("features") or []
    if not feature_names:
        # Some older preprocessors may not store 'features' key; try to
        # fall back to attribute on the model bundle if available.
        if isinstance(model_obj, dict) and "features" in model_obj:
            feature_names = model_obj["features"]
        else:
            raise RuntimeError("preprocessor.pkl does not contain 'features' key")

    return model_obj, feature_names


@lru_cache(maxsize=1)
def _build_explainer() -> Tuple[Any, float]:
    """Create and return a SHAP explainer and its expected baseline value.

    If a standalone ``shap_explainer.pkl`` exists in the model directory,
    it will be loaded and returned.  Otherwise the explainer is built
    using ``shap.TreeExplainer`` on the loaded model with a zero
    background.  The returned baseline is a single float value.

    :returns: (explainer, expected_value)
    """
    # If shap is not available, return a dummy explainer and baseline 0.0.
    if shap is None:
        return None, 0.0

    # If a prebuilt explainer is available, use it
    if DEFAULT_EXPLAINER_PATH.exists():
        try:
            explainer = joblib.load(DEFAULT_EXPLAINER_PATH)
            # Compute a scalar expected value from the explainer's base values
            base_vals = getattr(explainer, "base_values", None)
            if base_vals is None:
                # shap.Explainer may expose expected_value instead
                expected_value: Union[float, np.ndarray] = getattr(explainer, "expected_value", 0.0)
            else:
                expected_value = base_vals
            # Collapse to a scalar
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = float(np.array(expected_value).ravel()[0])
            else:
                expected_value = float(expected_value)  # type: ignore[assignment]
            return explainer, expected_value  # type: ignore[return-value]
        except Exception:
            # Fall back to building explainer below
            pass

    # No standalone explainer; build one from the model if possible
    model_obj, feature_names = _load_model_and_preprocessor()
    # Create a background of zeros (1 x num_features)
    background = np.zeros((1, len(feature_names)), dtype=float)
    try:
        explainer = shap.TreeExplainer(model_obj, background)
    except Exception:
        # Fallback: use KernelExplainer if TreeExplainer fails.  This is
        # slower but ensures SHAP explanations are available for non-tree
        # models.  We wrap this in a try/except because KernelExplainer
        # can still fail if the model does not expose predict.
        try:
            explainer = shap.KernelExplainer(model_obj.predict, background)  # type: ignore[attr-defined]
        except Exception:
            # If both explainers fail, return dummy values
            return None, 0.0
    # Determine expected value
    expected = getattr(explainer, "expected_value", 0.0)
    if isinstance(expected, (list, np.ndarray)):
        expected = float(np.array(expected).ravel()[0])
    else:
        expected = float(expected)
    return explainer, expected


class IncomeModel:
    """Production model for income prediction with SHAP explanations.

    The ``predict`` method accepts a normalized dictionary of features
    (typically produced by ``canonicalize_features``) and returns a
    predicted income along with a JSON explanation and a human-readable
    summary.  Internally it supports both legacy single-model
    predictions and the newer ensemble bundle format.
    """

    async def predict(self, features: Dict[str, Any]) -> Tuple[float, Dict[str, Any], str]:
        """Predict the client's income and return explanations.

        :param features: normalized feature dictionary (numbers or strings)
        :return: a tuple of (income_pred, explanation_json, text_explanation)
        """
        model_obj, feature_names = _load_model_and_preprocessor()

        # If model_obj is a dict, treat it as an ensemble bundle
        if isinstance(model_obj, dict):
            return await self._predict_with_ensemble(model_obj, features)
        else:
            # Fallback to legacy single-model prediction
            return await self._predict_with_single(model_obj, feature_names, features)

    async def _predict_with_single(self, model: Any, feature_names: List[str], features: Dict[str, Any]) -> Tuple[float, Dict[str, Any], str]:
        """Predict using a legacy single model (LightGBM/XGBoost/sklearn).

        This logic corresponds to the original implementation in the
        baseline project.  It constructs a numeric feature vector from
        ``features`` in the order specified by ``feature_names``, using
        ``-999.0`` for missing values or hashed strings for categorical
        values.  SHAP values are computed via TreeExplainer or a
        fallback.
        """
        # Prepare row values and raw record for reporting
        raw_for_report: Dict[str, Any] = {}
        row_values: List[float] = []
        for name in feature_names:
            raw_val = features.get(name)
            raw_for_report[name] = raw_val
            if raw_val is None or raw_val == "":
                row_values.append(-999.0)
                continue
            try:
                row_values.append(float(raw_val))
            except Exception:
                # For non-numeric strings, map to a deterministic pseudo-number
                row_values.append(float(abs(hash(str(raw_val))) % 1000))

        X_row = np.array(row_values, dtype=float).reshape(1, -1)

        # Compute the prediction depending on model type
        if xgb is not None and isinstance(model, xgb.Booster):  # type: ignore[attr-defined]
            dmat = xgb.DMatrix(X_row, feature_names=feature_names)  # type: ignore[attr-defined]
            pred = float(model.predict(dmat)[0])  # type: ignore[operator]
        elif lgb is not None and isinstance(model, lgb.Booster):  # type: ignore[attr-defined]
            pred = float(model.predict(X_row)[0])  # type: ignore[index]
        else:
            # Generic scikit-learn model
            pred = float(model.predict(X_row)[0])  # type: ignore[no-any-return]

        # Compute SHAP values if possible
        explainer, expected_value = _build_explainer()
        if shap is not None and explainer is not None:
            try:
                shap_vals_raw = explainer.shap_values(X_row)  # type: ignore[operator]
                # SHAP may return a list for multi-output; take the first element
                if isinstance(shap_vals_raw, list):
                    shap_vals_raw = shap_vals_raw[0]
                shap_vals = np.array(shap_vals_raw)[0]
            except Exception:
                expected_value = pred
                shap_vals = np.zeros(len(feature_names), dtype=float)
        else:
            # SHAP not available; use dummy values
            expected_value = pred
            shap_vals = np.zeros(len(feature_names), dtype=float)

        # Build contribution items
        contrib_items: List[Dict[str, Any]] = []
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

        # Take top features by absolute SHAP magnitude
        top_features = sorted(contrib_items, key=lambda it: abs(it["shap_value"]), reverse=True)[:10]

        explanation_json: Dict[str, Any] = {
            "baseline_income": expected_value,
            "prediction": pred,
            "top_features": top_features,
        }

        # Construct a human-readable explanation
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

    async def _predict_with_ensemble(self, bundle: Dict[str, Any], features: Dict[str, Any]) -> Tuple[float, Dict[str, Any], str]:
        """Predict using an ensemble model bundle.

        The bundle is expected to contain the keys ``lgb_models``,
        ``cat_models``, ``nn_models``, ``meta_model``, ``scaler``,
        ``features``, ``medians``, ``upper_limits``, ``pca``, and
        ``maps``.  It replicates the preprocessing steps from the training
        pipeline before computing base model predictions, stacking them,
        and applying the meta model.  SHAP values for the ensemble are
        computed using the prebuilt explainer if available.
        """
        # Unpack bundle components
        lgb_models = bundle.get("lgb_models", [])
        cat_models = bundle.get("cat_models", [])
        nn_models = bundle.get("nn_models", [])
        meta_model = bundle.get("meta_model")
        scaler: RobustScaler = bundle.get("scaler")  # type: ignore[assignment]
        features_list: List[str] = bundle.get("features", [])
        medians: Dict[str, float] = bundle.get("medians", {})  # type: ignore[assignment]
        upper_limits: Dict[str, float] = bundle.get("upper_limits", {})  # type: ignore[assignment]
        pca: Union[PCA, None] = bundle.get("pca")  # type: ignore[assignment]
        maps: Dict[str, Dict[str, int]] = bundle.get("maps", {})  # type: ignore[assignment]

        # Defensive checks
        if meta_model is None:
            raise RuntimeError("Ensemble bundle is missing 'meta_model'")
        if not features_list:
            raise RuntimeError("Ensemble bundle is missing 'features' list")

        # Create a DataFrame with a single row for processing
        df = pd.DataFrame([features])

        # Fill numeric columns using medians; convert to numeric
        orig_num_cols = list(medians.keys())
        for col in orig_num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(medians[col])
            else:
                # Column missing in input; fill with median directly
                df[col] = medians[col]

        # Map categorical columns using training maps
        cat_cols = list(maps.keys())
        for col in cat_cols:
            if col in df.columns:
                df[col] = df[col].fillna("MISSING")
                df[col] = df[col].apply(lambda v: maps[col].get(str(v), maps[col].get("MISSING", -1)))
            else:
                # If the categorical column is missing, fill with the code for "MISSING"
                missing_code = maps[col].get("MISSING", -1)
                df[col] = missing_code

        # Feature engineering
        # Some features may not exist in df; use get with default
        income_value = df.get('incomeValue', pd.Series([np.nan])).astype(float)
        age_value = df.get('age', pd.Series([np.nan])).astype(float)
        turn_cur_cr_avg_v2 = df.get('turn_cur_cr_avg_v2', pd.Series([np.nan])).astype(float)
        turn_cur_db_avg_v2 = df.get('turn_cur_db_avg_v2', pd.Series([np.nan])).astype(float)
        hdb_bki_total_max_limit = df.get('hdb_bki_total_max_limit', pd.Series([np.nan])).astype(float)
        dp_ils_salary_ratio_1y3y = df.get('dp_ils_salary_ratio_1y3y', pd.Series([np.nan])).astype(float)
        turn_cur_cr_sum_v2 = df.get('turn_cur_cr_sum_v2', pd.Series([np.nan])).astype(float)
        turn_cur_db_sum_v2 = df.get('turn_cur_db_sum_v2', pd.Series([np.nan])).astype(float)
        # Construct new columns
        new_features = {
            'income_to_age_ratio': income_value / (age_value + 1e-5),
            'turnover_ratio_cr_db': turn_cur_cr_avg_v2 / (turn_cur_db_avg_v2 + 1e-5),
            'bki_limit_to_income': hdb_bki_total_max_limit / (income_value + 1e-5),
            'salary_growth_1y3y': dp_ils_salary_ratio_1y3y,
            'total_turnover': turn_cur_cr_sum_v2 + turn_cur_db_sum_v2,
        }
        df = pd.concat([df, pd.DataFrame(new_features)], axis=1)

        new_num_cols = ['income_to_age_ratio', 'turnover_ratio_cr_db', 'bki_limit_to_income', 'salary_growth_1y3y', 'total_turnover']
        all_num_cols = orig_num_cols + new_num_cols

        # Clip numeric values by 99th percentile upper limits
        for col in all_num_cols:
            if col not in df.columns:
                continue
            if col in upper_limits:
                df[col] = np.clip(df[col], None, upper_limits[col])

        # Apply PCA on turnover-related columns if PCA is available
        turnover_cols = [col for col in df.columns if 'turn_' in col]
        if pca is not None and turnover_cols:
            try:
                pca_features = pca.transform(df[turnover_cols].values)
                pca_df = pd.DataFrame(pca_features, columns=[f'pca_turn_{i}' for i in range(pca.n_components_)])
                df = pd.concat([df, pca_df], axis=1)
                df.drop(columns=turnover_cols, inplace=True)
            except Exception:
                # If PCA fails (e.g. missing columns), drop turnover columns without PCA
                df.drop(columns=turnover_cols, inplace=True)
        # Update list of numeric columns after PCA
        pca_cols = [col for col in df.columns if col.startswith('pca_turn_')]
        all_num_cols = [c for c in all_num_cols if c not in turnover_cols] + pca_cols

        # Scale numeric features using RobustScaler.  In case the
        # pre-trained scaler expects a different number of columns than
        # provided (e.g. due to dropped PCA/turnover features), catch
        # errors and fall back to identity scaling.
        try:
            df[all_num_cols] = scaler.transform(df[all_num_cols])
        except Exception:
            # Attempt to add missing columns to match the scaler's expected
            # number of features.  Missing columns are filled with zeros.
            try:
                for c in all_num_cols:
                    if c not in df.columns:
                        df[c] = 0.0
                df[all_num_cols] = scaler.transform(df[all_num_cols])
            except Exception:
                # If still failing, skip scaling entirely
                pass

        # Ensure all expected features are present in the row; fill missing with zero
        for col in features_list:
            if col not in df.columns:
                df[col] = 0

        # Arrange columns in the exact order used during training
        X_client = df[features_list]

        # Generate base model predictions (in original scale, not log)
        # LightGBM models
        if lgb_models:
            lgb_preds = np.mean([np.expm1(m.predict(X_client)) for m in lgb_models], axis=0)
        else:
            lgb_preds = np.zeros(X_client.shape[0])
        # CatBoost models
        if cat_models:
            cat_preds = np.mean([np.expm1(m.predict(X_client)) for m in cat_models], axis=0)
        else:
            cat_preds = np.zeros(X_client.shape[0])
        # Keras models (NN)
        if nn_models:
            # Keras models expect numpy arrays
            nn_preds = np.mean([
                np.expm1(m.predict(X_client.values, verbose=0).flatten()) for m in nn_models
            ], axis=0)
        else:
            nn_preds = np.zeros(X_client.shape[0])

        # Stack predictions as features for meta model
        stack_preds = np.column_stack([lgb_preds, cat_preds, nn_preds])
        # Predict using meta model (outputs log1p of income)
        try:
            meta_output = meta_model.predict(stack_preds)
        except Exception:
            # Some XGBoost models require DMatrix
            if xgb is not None and isinstance(meta_model, xgb.Booster):  # type: ignore[isinstance]
                dmat = xgb.DMatrix(stack_preds, feature_names=['LGB', 'CAT', 'NN'])  # type: ignore[attr-defined]
                meta_output = meta_model.predict(dmat)
            else:
                raise
        # Convert back from log1p to original scale
        pred_income = float(np.expm1(meta_output)[0])

        # Obtain SHAP values using the prebuilt explainer (if available)
        explainer, expected_val = _build_explainer()
        feature_names_shap = ['LGB Prediction', 'CatBoost Prediction', 'NN Prediction']
        if shap is not None and explainer is not None:
            try:
                shap_values = explainer(stack_preds)  # type: ignore[operator]
                base_value = float(shap_values.base_values[0]) if hasattr(shap_values, 'base_values') else float(expected_val)
                shap_vals = np.array(shap_values.values[0])
            except Exception:
                base_value = pred_income
                shap_vals = np.zeros(stack_preds.shape[1])
        else:
            # SHAP not available; fallback to zero contributions
            base_value = pred_income
            shap_vals = np.zeros(stack_preds.shape[1])

        # Build top features list (only three in the ensemble)
        contrib_items: List[Dict[str, Any]] = []
        for name, shap_val, model_val in zip(feature_names_shap, shap_vals, stack_preds[0]):
            contrib_items.append(
                {
                    "feature": name,
                    "title": name,
                    "value": float(model_val),
                    "model_value": float(model_val),
                    "shap_value": float(shap_val),
                }
            )

        top_features = sorted(contrib_items, key=lambda it: abs(it["shap_value"]), reverse=True)

        explanation_json: Dict[str, Any] = {
            "baseline_income": base_value,
            "prediction": pred_income,
            "top_features": top_features,
        }

        # Construct human-readable explanation from top contributions
        if top_features:
            parts = []
            for item in top_features[:3]:
                feat = item["title"]
                val = item["value"]
                shap_val = item["shap_value"]
                sign = "+" if shap_val >= 0 else "-"
                parts.append(f"{feat} ({val:,.0f}) {sign}{abs(shap_val):,.0f} ₽")
            text_explanation = (
                f"Модель оценила доход в {pred_income:,.0f} ₽. "
                f"Наибольший вклад дали: {', '.join(parts)}."
            )
        else:
            text_explanation = (
                f"Модель оценила доход в {pred_income:,.0f} ₽. "
                "Вклад признаков распределён равномерно."
            )

        return pred_income, explanation_json, text_explanation