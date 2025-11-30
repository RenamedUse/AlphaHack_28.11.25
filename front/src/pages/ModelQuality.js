import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import './ModelQuality.css';

function ModelQuality() {
  const navigate = useNavigate();
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadModelMetrics();
  }, []);

  const loadModelMetrics = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getModelHealth();
      setMetrics(data);
    } catch (err) {
      setError('Ошибка загрузки метрик: ' + err.message);
      console.error('Ошибка загрузки метрик:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="model-quality-page">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Загрузка метрик модели...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="model-quality-page">
        <div className="error-container">
          <h2>Ошибка загрузки данных</h2>
          <p>{error}</p>
          <button onClick={loadModelMetrics}>Повторить попытку</button>
        </div>
      </div>
    );
  }

  return (
    <div className="model-quality-page">
      {/* Хедер */}
      <div className="monitoring-header">
        <div className="header-content">
          <h1>Мониторинг качества модели</h1>
          <button 
            className="back-button"
            onClick={() => navigate('/main')}
          >
            Назад на главную
          </button>
        </div>
      </div>

      {metrics && (
        <div className="metrics-container">
          {/* Статус модели */}
          <div className="model-status-section">
            <h2>Статус модели</h2>
            <div className="status-grid">
              <div className="status-card">
                <div className="status-indicator">
                  <div className={`status-dot ${metrics.model.loaded ? 'loaded' : 'not-loaded'}`}></div>
                  <span className="status-text">
                    {metrics.model.loaded ? 'Модель загружена' : 'Модель не загружена'}
                  </span>
                </div>
                <div className="feature-count">
                  Количество признаков: <strong>{metrics.model.features_count}</strong>
                </div>
              </div>
            </div>
          </div>

          {/* Трафик за 24 часа */}
          <div className="traffic-section">
            <h2>Трафик за 24 часа</h2>
            <div className="traffic-grid">
              <div className="traffic-card">
                <div className="traffic-icon">📈</div>
                <div className="traffic-content">
                  <h3>Всего предсказаний</h3>
                  <div className="traffic-value">{metrics.traffic_24h.total.toLocaleString()}</div>
                </div>
              </div>

              <div className="traffic-card">
                <div className="traffic-icon">🎯</div>
                <div className="traffic-content">
                  <h3>Симуляции</h3>
                  <div className="traffic-value">{metrics.traffic_24h.simulations.toLocaleString()}</div>
                </div>
              </div>

              <div className="traffic-card">
                <div className="traffic-icon">👥</div>
                <div className="traffic-content">
                  <h3>Уникальные клиенты</h3>
                  <div className="traffic-value">{metrics.traffic_24h.unique_clients.toLocaleString()}</div>
                </div>
              </div>

              {metrics.traffic_24h.avg_latency && (
                <div className="traffic-card">
                  <div className="traffic-icon">⚡</div>
                  <div className="traffic-content">
                    <h3>Средняя задержка</h3>
                    <div className="traffic-value">{metrics.traffic_24h.avg_latency} мс</div>
                  </div>
                </div>
              )}
            </div>

            {/* Распределение по версиям */}
            {metrics.traffic_24h.versions && metrics.traffic_24h.versions.length > 0 && (
              <div className="versions-distribution">
                <h3>Распределение по версиям модели</h3>
                <div className="versions-list">
                  {metrics.traffic_24h.versions.map((version, index) => (
                    <div key={index} className="version-item">
                      <span className="version-name">{version.model_version}</span>
                      <div className="version-bar">
                        <div 
                          className="version-progress" 
                          style={{ 
                            width: `${(version.count / metrics.traffic_24h.total) * 100}%` 
                          }}
                        ></div>
                      </div>
                      <span className="version-count">{version.count.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Источники запросов */}
            {metrics.traffic_24h.top_request_sources && metrics.traffic_24h.top_request_sources.length > 0 && (
              <div className="sources-distribution">
                <h3>Источники запросов</h3>
                <div className="sources-list">
                  {metrics.traffic_24h.top_request_sources.map((source, index) => (
                    <div key={index} className="source-item">
                      <span className="source-name">{formatSourceName(source.request_source)}</span>
                      <div className="source-bar">
                        <div 
                          className="source-progress" 
                          style={{ 
                            width: `${(source.count / metrics.traffic_24h.total) * 100}%` 
                          }}
                        ></div>
                      </div>
                      <span className="source-count">{source.count.toLocaleString()}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Ошибки */}
          <div className="errors-section">
            <h2>Ошибки за 24 часа</h2>
            <div className="errors-summary">
              <div className="error-card">
                <div className="error-icon">📊</div>
                <div className="error-content">
                  <h3>Всего заданий импорта</h3>
                  <div className="error-value">{metrics.errors_24h.import_jobs_total}</div>
                </div>
              </div>

              <div className="error-card">
                <div className="error-icon">❌</div>
                <div className="error-content">
                  <h3>Неудачных заданий</h3>
                  <div className="error-value error-failed">{metrics.errors_24h.import_jobs_failed}</div>
                </div>
              </div>

              <div className="error-card">
                <div className="error-icon">📈</div>
                <div className="error-content">
                  <h3>Успешных заданий</h3>
                  <div className="error-value error-success">
                    {metrics.errors_24h.import_jobs_total - metrics.errors_24h.import_jobs_failed}
                  </div>
                </div>
              </div>
            </div>

            {/* Топ ошибок импорта */}
            {metrics.errors_24h.top_import_errors && metrics.errors_24h.top_import_errors.length > 0 && (
              <div className="top-errors">
                <h3>Частые ошибки импорта</h3>
                <div className="errors-list">
                  {metrics.errors_24h.top_import_errors.map((error, index) => (
                    <div key={index} className="error-item">
                      <div className="error-message">
                        {formatErrorMessage(error.error)}
                      </div>
                      <div className="error-count">
                        {error.count} раз
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Клиенты */}
          <div className="clients-section">
            <h2>Клиенты</h2>
            <div className="clients-grid">
              <div className="client-card">
                <div className="client-icon">👥</div>
                <div className="client-content">
                  <h3>Всего клиентов</h3>
                  <div className="client-value">{metrics.clients.total.toLocaleString()}</div>
                </div>
              </div>

              <div className="client-card">
                <div className="client-icon">🔄</div>
                <div className="client-content">
                  <h3>Активных за 24 часа</h3>
                  <div className="client-value">{metrics.clients.active_24h.toLocaleString()}</div>
                </div>
              </div>
            </div>

            {!metrics.clients.segments_available && (
              <div className="segments-notice">
                Информация о сегментах клиентов недоступна
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Вспомогательные функции для форматирования
function formatSourceName(source) {
  const sourceNames = {
    'csv_import': 'Импорт CSV',
    'api_request': 'API запрос',
    'simulation': 'Симуляция'
  };
  return sourceNames[source] || source;
}

function formatErrorMessage(error) {
  // Укорачиваем длинные сообщения об ошибках
  if (error.length > 100) {
    return error.substring(0, 100) + '...';
  }
  return error;
}

export default ModelQuality;