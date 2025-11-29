"""Мок-реализация модели прогноза дохода

Модуль содержит простую детерминированную модель для демонстрационных целей
Она выдаёт синтетический прогноз дохода на основе числовых признаков и
возвращает SHAP-подобное объяснение вклада признаков

В боевой системе эту реализацию нужно заменить на настоящую модель,
загружаемую с диска или вызываемую через отдельный микросервис
"""

from __future__ import annotations

from typing import Dict, Any, Tuple


class MockIncomeModel:
    """Простая детерминированная модель оценки дохода по признакам

    Модель специально сделана простой: она считает базовый доход и добавляет
    вклад от числовых признаков. Также возвращается SHAP-подобное объяснение
    с топовыми признаками
    """

    def __init__(self, baseline: float = 50_000.0) -> None:
        self.baseline = baseline

    async def predict(self, features: Dict[str, Any]) -> Tuple[float, Dict[str, Any], str]:
        """Посчитать прогноз дохода по словарю признаков

        :param features: Нормализованный словарь признаков
        :return: Кортеж (income_pred, explanation_json, text_explanation)
        """
        contributions: Dict[str, Dict[str, Any]] = {}
        income_pred = self.baseline

        # Простое правило: числовые признаки вносят пропорциональный вклад
        for key, value in features.items():
            shap_value = 0.0
            if isinstance(value, (int, float)):
                # Вклад масштабируется по числовому значению
                shap_value = float(value) * 100.0
            elif isinstance(value, str):
                # Для категориальных признаков даём небольшой фиксированный вклад
                shap_value = 500.0

            income_pred += shap_value
            contributions[key] = {"value": value, "shap_value": shap_value}

        # Формируем структуру объяснения
        # Сортируем признаки по модулю вклада и берём топ-10
        top_items = sorted(
            contributions.items(),
            key=lambda item: abs(item[1]["shap_value"]),
            reverse=True,
        )[:10]

        top_features = []
        for feature_name, info in top_items:
            top_features.append(
                {
                    "feature": feature_name,
                    "title": feature_name,  # В реальности здесь можно подставить человекочитаемый заголовок
                    "value": info["value"],
                    "shap_value": info["shap_value"],
                }
            )

        explanation_json: Dict[str, Any] = {
            "baseline_income": self.baseline,
            "prediction": income_pred,
            "top_features": top_features,
        }

        # Строим текстовое объяснение
        if top_features:
            parts = []
            for item in top_features[:3]:
                feat = item["title"]
                val = item["value"]
                shap_val = item["shap_value"]
                sign = "+" if shap_val >= 0 else "-"
                parts.append(f"{feat} ({val}) {sign}{abs(shap_val):,.0f} ₽")
            text_explanation = (
                f"Модель оценила доход в {income_pred:,.0f} ₽. Наибольший вклад дали: "
                + ", ".join(parts)
                + "."
            )
        else:
            text_explanation = (
                f"Модель оценила доход в {income_pred:,.0f} ₽. Вклад признаков равномерный."
            )

        return income_pred, explanation_json, text_explanation