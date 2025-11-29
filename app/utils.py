from __future__ import annotations

import json
import hashlib
import math
from typing import Dict, Any


def canonicalize_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Нормализовать словарь признаков.

    Логика:
    - значения ``None`` и пустые строки отбрасываются
    - строковые значения чистим от пробелов и неразрывных пробелов
    - для строк заменяем десятичную запятую на точку и пробуем int/float
    - если привести к числу не удалось, оставляем строку (обрезанную)
    """
    normalized: Dict[str, Any] = {}

    for key, value in raw.items():
        if value is None:
            continue

        if isinstance(value, str):
            text = value.strip().replace("\u00a0", "")
            if not text:
                continue

            # Пробуем распарсить как число с учётом десятичной запятой
            numeric_candidate = text.replace(",", ".")

            # Сначала int (если нет точки)
            try:
                if "." not in numeric_candidate:
                    int_val = int(numeric_candidate)
                    normalized[key] = int_val
                    continue
            except (ValueError, TypeError):
                pass

            # Потом float
            try:
                float_val = float(numeric_candidate)
                normalized[key] = float_val
                continue
            except (ValueError, TypeError):
                pass

            # Не число — оставляем строку
            normalized[key] = text
        else:
            normalized[key] = value

    return normalized


def sanitize_for_json(value: Any) -> Any:
    """Рекурсивно заменяет NaN/Infinity на None, чтобы JSON был валидным для PostgreSQL."""
    # Числа с NaN/Inf
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    # Словари
    if isinstance(value, dict):
        return {k: sanitize_for_json(v) for k, v in value.items()}

    # Списки / кортежи
    if isinstance(value, (list, tuple)):
        return [sanitize_for_json(v) for v in value]

    # Остальное (int, str, bool, None и т.д.) не трогаем
    return value


def compute_features_hash(features: Dict[str, Any]) -> str:
    """Посчитать детерминированный хэш для словаря признаков.

    Предполагается, что features уже прошли через sanitize_for_json.
    """
    payload = json.dumps(features, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest