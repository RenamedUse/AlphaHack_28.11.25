# TunTunDohod (AlphaHack_28.11.25)

Прототип AI-сервиса для Альфа-Банка по прогнозу дохода клиентов и
формированию персональных финансовых рекомендаций.

Репозиторий команды «Тунтунпобедил».

## 1. Идея и бизнес-ценность

Задача сервиса --- помочь банку точнее оценивать доход клиентов и
использовать эту оценку в операционных процессах:

-   снизить риск при кредитовании за счет более точной оценки
    платёжеспособности;
-   повысить конверсию продуктовых предложений за счет персонализации;
-   улучшить клиентский опыт: меньше нерелевантных офферов, более
    справедливые лимиты и условия.

Ключевая мысль: модель дохода интегрирована в прототип фронта и
backend-API, которые показывают, как предсказание превращается в
конкретные действия: сегментация, рекомендации и мониторинг качества.

## 2. Функциональность прототипа

### Для сотрудника банка (frontend)

-   Поиск клиента.
-   Просмотр карточки:
    -   прогноз дохода;
    -   сегмент (низкий / средний / высокий);
    -   рекомендованные продукты.
-   Объяснение прогноза:
    -   ключевые признаки;
    -   текстовая интерпретация.
-   Экран мониторинга:
    -   количество запросов;
    -   доля симуляций;
    -   метрики по логам.

### Для аналитика / администратора (backend)

-   Импорт CSV.
-   Добавление сегментов пользователей.
-   Логирование запросов:
    -   параметры запроса;
    -   предсказания модели;
    -   статусы и ошибки.

## 3. Архитектура

-   **ml/** --- обучение модели, сохранение артефактов
    (`income_prediction_model.pkl`, `preprocessor.pkl`).
-   **app/** --- backend на FastAPI:
    -   API (публичные и административные);
    -   SQLAlchemy async;
    -   интеграция ML.
-   **front/** --- React-приложение для сотрудников банка.
-   **docker-compose.yml** --- запуск всех компонентов.

## 4. Структура репозитория

    AlphaHack_28.11.25/
    ├── app/
    ├── front/
    ├── ml/
    ├── tuntundohod/
    ├── docker-compose.yml
    ├── requirements.txt
    └── README.md

## 5. Технологии

-   Python, pandas, numpy
-   XGBoost
-   FastAPI, SQLAlchemy async, PostgreSQL
-   React
-   Docker, docker-compose

## 6. Быстрый старт

### Docker Compose

    git clone https://github.com/RenamedUse/AlphaHack_28.11.25.git
    cd AlphaHack_28.11.25
    docker compose up --build

### Локальный запуск backend

    cd app
    python -m venv .venv
    source .venv/bin/activate
    pip install -r ../requirements.txt
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Swagger: http://localhost:8000/docs

## 7. Основные API

### Публичная зона

-   `GET /api/clients/{id}/card`
-   `POST /api/predictions`
-   `GET /api/clients/{id}/offers`

### Админская зона

-   `POST /api/admin/import/start`
-   `GET /api/admin/import/{id}`
-   `POST /api/admin/seed/start`

### Мониторинг

-   `GET /api/monitoring/model-health`

## 8. ML-модель

-   Dataset: train/test CSV.
-   Target: `target`.
-   Metric: WMAE.
-   Модель: XGBoost.
-   Артефакты:
    -   `income_prediction_model.pkl`
    -   `preprocessor.pkl`

## 9. Мониторинг и логирование

Backend собирает prediction_log: \* client_id; \* входные данные; \*
предсказание; \* is_simulation; \* timestamp.

## 10. Сценарий демонстрации

1.  Выбор клиента.
2.  Показ прогноза и сегмента.
3.  Рекомендации.
4.  Объяснение.
5.  Экран мониторинга.

## 11. Дальнейшее развитие

-   Улучшение модели;
-   Расширение сегментации;
-   Промышленный мониторинг;
-   Алёрты и отслеживание дрифта.

## 12. Так же мы обучили новую модель, но не успели интегрировать ее на беке

Ссылка на новую модель: https://drive.google.com/drive/folders/1OOcO7YtH1kMiKILyoNOVKLwQCqBHRQ8T?usp=sharing

## 13. Дополнительно про обучение

Модели и обучение:
train_model.py — обучение ансамбля моделей.
income_ensemble_model.pkl — сохранённая обученная итоговая модель - очень тяжёлая, на гитхаб не получилось залить из-за ограничения
preprocessor.pkl — сохранённый препроцессор.

Инференс:
inference.py — скрипт получения прогноза: загрузка препроцессора и модели, предикт.
generate_json.py — генерация тестовых JSON-входов для примеров пользователей.
user_id_2.json, user_id_4.json, user_id_46.json — примеры разных пользователей.

Данные:
hackathon_income_train.csv — тренировочный датасет.
hackathon_income_test.csv — тестовый датасет.
submission_ensemble.csv — результат предсказаний, поданный на платформу - аналогично income_ensemble_model.pkl - проблема с размером файлов

Интерпретация модели (SHAP):
shap_explainer.pkl — обученный SHAP-объект.
shap_visualization.py — генерация визуализаций SHAP.
shap_summary.png, shap_waterfall.png — примеры графиков важности признаков.

Тестовый фронтенд:
fastapi_app.py — backend API на FastAPI (обработка запросов, вызов модели).
front.html — простая веб-страница для ввода параметров клиента и получения предсказания.

Служебные директории:
catboost_info/ — служебные файлы CatBoost.
__pycache__/ — автоматически генерируемые Python-кэши.
requirements.txt — список зависимостей проекта.

Старт:
1. Установка окружения
pip install -r requirements.txt
2. Обучение модели
python train_model.py
После выполнения появятся income_ensemble_model.pkl и preprocessor.pkl.
3. Предсказание дохода
python inference.py --input user_id_2.json
4. Генерация сабмишена
(если внутри inference есть батч-режим)
python inference.py --csv hackathon_income_test.csv --out submission_ensemble.csv

Веб-демо - вотч-дэмо(!): 
Запуск API
uvicorn frontend.fastapi_app:app --reload
Использование
Открыть в браузере:
http://localhost:8000/front.html

Интерпретация модели:
Генерация SHAP-графиков:
python shap_visualization.py
Результаты сохраняются в PNG-файлы.
Что я юзаю:
Python
CatBoost / LightGBM / NN Keras
FastAPI для сервера
SHAP для анализа признаков
Pandas / NumPy для обработки данных
