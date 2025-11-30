import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import { FEATURES_TRANSLATIONS } from '../services/featuresTranslations';
import './Main.css';

function Main() {
  const navigate = useNavigate();
  
  // Состояния
  const [clients, setClients] = useState([]);
  const [selectedClient, setSelectedClient] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filters, setFilters] = useState({
    segment_code: '',
    min_income: '',
    max_income: ''
  });

  // Функция для получения перевода названия фичи
  const getFeatureTitle = (featureName) => {
    return FEATURES_TRANSLATIONS[featureName] || featureName;
  };

  // Функция для перевода текстового объяснения
  const translateExplanationText = (text) => {
    if (!text) return text;
    
    // Ищем в тексте названия фич в скобках и заменяем их
    let translatedText = text;
    
    // Регулярное выражение для поиска фич в формате: featureName (значение)
    const featureRegex = /(\w+)\s*\(([^)]+)\)/g;
    
    translatedText = translatedText.replace(featureRegex, (match, featureName, value) => {
      const translatedFeature = getFeatureTitle(featureName);
      return `${translatedFeature} (${value})`;
    });
    
    // Дополнительно ищем фичи без значений (только названия)
    Object.keys(FEATURES_TRANSLATIONS).forEach(featureName => {
      const regex = new RegExp(`\\b${featureName}\\b`, 'g');
      translatedText = translatedText.replace(regex, FEATURES_TRANSLATIONS[featureName]);
    });
    
    return translatedText;
  };

  // Функция загрузки клиентов с useCallback
  const loadClients = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getClients(filters);
      setClients(data);
      
      // Если есть клиенты, автоматически выбираем первого
      if (data.length > 0 && !selectedClient) {
        handleClientSelect(data[0]);
      }
    } catch (err) {
      setError('Ошибка загрузки клиентов: ' + err.message);
      console.error('Ошибка загрузки клиентов:', err);
    } finally {
      setLoading(false);
    }
  }, [filters, selectedClient]);

  // Загрузка клиентов при монтировании и изменении фильтров
  useEffect(() => {
    loadClients();
  }, [loadClients]);

  // Функция выбора клиента
  const handleClientSelect = async (client) => {
    setLoading(true);
    setError(null);
    try {
      const cardData = await api.getClientCard(client.external_id);
      
      // Упорядочиваем top_features по убыванию абсолютного значения shap_value
      // И заменяем названия фич на человеческие
      if (cardData.explanation && cardData.explanation.top_features) {
        cardData.explanation.top_features = cardData.explanation.top_features
          .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
          .map(feature => ({
            ...feature,
            title: getFeatureTitle(feature.title) // Заменяем название
          }));
      }
      
      // Переводим текстовое объяснение
      if (cardData.explanation && cardData.explanation.text) {
        cardData.explanation.text = translateExplanationText(cardData.explanation.text);
      }
      
      setSelectedClient(cardData);
    } catch (err) {
      setError('Ошибка загрузки карточки клиента: ' + err.message);
      console.error('Ошибка загрузки карточки клиента:', err);
    } finally {
      setLoading(false);
    }
  };

  // Остальной код без изменений...
  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    setError(null);
    try {
      const result = await api.importCSV(file);
      
      if (result.id) {
        pollImportStatus(result.id);
      }
    } catch (err) {
      setError('Ошибка импорта файла: ' + err.message);
      setLoading(false);
    }
  };

  const pollImportStatus = async (jobId) => {
    const interval = setInterval(async () => {
      try {
        const status = await api.getImportStatus(jobId);
        
        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(interval);
          if (status.status === 'completed') {
            loadClients();
          }
          setLoading(false);
        }
      } catch (err) {
        console.error('Ошибка проверки статуса импорта:', err);
        clearInterval(interval);
        setLoading(false);
      }
    }, 2000);
  };

  const applyFilters = (newFilters) => {
    setFilters(newFilters);
  };

  return (
    <div className='page main-page'>
      {/* Хедер */}
      <div className='header'>
        <input 
          type="text" 
          placeholder="Поиск клиентов..."
          className='search-input'
          onChange={(e) => applyFilters({...filters, search: e.target.value})}
        />
        <div className='buttons-container'>
            <label className='upload-btn'>
                Загрузить файл
                <input 
                  type="file" 
                  accept=".csv" 
                  style={{ display: 'none' }}
                  onChange={handleFileUpload}
                />
            </label>
            <button 
            className='monitoring-btn'
            onClick={() => navigate('/model-quality')}
            >
                Мониторинг
            </button>
        </div>
      </div>

      {/* Основной контент - два столбца */}
      <div className='content-container'>
        {/* Левая колонка - список клиентов (25%) */}
        <div className='user-list'>
          <div className='user-list-header'>
            <h3>Клиенты</h3>
            {loading && <span>Загрузка...</span>}
          </div>
          <div className='users-container'>
            {error && (
              <div className="error-message" style={{color: '#ef3124', padding: '10px', textAlign: 'center'}}>
                {error}
              </div>
            )}
            
            {clients.map(client => (
              <div
                key={client.external_id}
                className={`user-item ${selectedClient && selectedClient.client.external_id === client.external_id ? 'active' : ''}`}
                onClick={() => handleClientSelect(client)}
              >
                <p className='user-name'>{client.display_name || client.external_id}</p>
                <p className='user-details'>
                  Доход: {client.income_pred.toLocaleString()} ₽ • {client.segment_name}
                </p>
              </div>
            ))}
            
            {clients.length === 0 && !loading && (
              <div style={{textAlign: 'center', padding: '20px', color: '#666'}}>
                Клиенты не найдены
              </div>
            )}
          </div>
        </div>

        {/* Правая колонка - основная информация (75%) */}
        <div className='main-info'>
          {/* Сетка из 3 колонок */}
          <div className='main-info-grid'>
            
            {/* Карточка с основной информацией о клиенте */}
            {selectedClient && (
              <>
                <div className='main-info-card'>
                  <div className='card-header'>
                    <h3>{selectedClient.client.display_name || selectedClient.client.external_id}</h3>
                  </div>
                  <div className='card-content'>
                    <div className='income-section'>
                      <div className='income-amount'>{selectedClient.prediction.income_pred.toLocaleString()} ₽</div>
                      <div className={`income-segment ${selectedClient.prediction.segment_code}`}>
                        {selectedClient.prediction.segment_name}
                      </div>
                    </div>
                    <div className='client-meta'>
                      <p><strong>ID:</strong> {selectedClient.client.external_id}</p>
                      <p><strong>Модель:</strong> {selectedClient.prediction.model_version}</p>
                      <p><strong>Обновлено:</strong> {new Date(selectedClient.prediction.updated_at).toLocaleDateString()}</p>
                    </div>
                  </div>
                </div>

                {/* Карточка с объяснением предсказания */}
                <div className='main-info-card card-w'>
                  <div className='card-header'>
                    <h4>Объяснение предсказания</h4>
                  </div>
                  <div className='card-content'>
                    <p>{selectedClient.explanation.text}</p>
                    <div className='features-list'>
                      {selectedClient.explanation.top_features.map((feature, index) => (
                        <div key={index} className='feature-item'>
                          <span className='feature-title' title={feature.title}>
                            {feature.title}
                          </span>
                          <span className={`feature-impact ${feature.shap_value > 0 ? 'positive' : 'negative'}`}>
                            {feature.shap_value > 0 ? '+' : ''}{feature.shap_value.toLocaleString()} ₽
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Карточка с рекомендованными продуктами */}
                <div className='main-info-card card-w'>
                  <div className='card-header'>
                    <h4>Рекомендуемые продукты</h4>
                  </div>
                  <div className='card-content'>
                    {selectedClient.products.map((product, index) => (
                      <div key={index} className='product-item'>
                        <strong>{product.name}</strong>
                        <p>{product.description}</p>
                        <p><em>{product.reason}</em></p>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            )}

            {/* Состояние когда клиент не выбран */}
            {!selectedClient && !loading && (
              <div className='main-info-card card-uw' style={{textAlign: 'center', padding: '40px'}}>
                <h3>Выберите клиента из списка</h3>
                <p>Для просмотра детальной информации выберите клиента из списка слева</p>
              </div>
            )}

            {/* Состояние загрузки */}
            {loading && (
              <div className='main-info-card card-uw' style={{textAlign: 'center', padding: '40px'}}>
                <h3>Загрузка...</h3>
                <p>Пожалуйста, подождите</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default Main;