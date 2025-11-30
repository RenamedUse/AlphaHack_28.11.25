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
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);

  // Функция для получения перевода названия фичи
  const getFeatureTitle = (featureName) => {
    return FEATURES_TRANSLATIONS[featureName] || featureName;
  };

  // Функция для перевода текстового объяснения
  const translateExplanationText = (text) => {
    if (!text) return text;
    
    let translatedText = text;
    const featureRegex = /(\w+)\s*\(([^)]+)\)/g;
    
    translatedText = translatedText.replace(featureRegex, (match, featureName, value) => {
      const translatedFeature = getFeatureTitle(featureName);
      return `${translatedFeature} (${value})`;
    });
    
    Object.keys(FEATURES_TRANSLATIONS).forEach(featureName => {
      const regex = new RegExp(`\\b${featureName}\\b`, 'g');
      translatedText = translatedText.replace(regex, FEATURES_TRANSLATIONS[featureName]);
    });
    
    return translatedText;
  };

  // Функция загрузки клиентов
  const loadClients = useCallback(async (reset = false) => {
    if (reset) {
      setOffset(0);
      setHasMore(true);
    }

    const currentOffset = reset ? 0 : offset;
    
    if (currentOffset === 0) {
      setLoading(true);
    } else {
      setLoadingMore(true);
    }
    
    setError(null);
    try {
      const data = await api.getClients({
        offset: currentOffset,
        limit: 50
      });
      
      if (reset || currentOffset === 0) {
        setClients(data);
        // Автоматически выбираем первого клиента при первой загрузке
        if (data.length > 0 && !selectedClient && reset) {
          handleClientSelect(data[0]);
        }
      } else {
        setClients(prev => [...prev, ...data]);
      }
      
      // Если получено меньше клиентов, чем лимит, значит это последняя страница
      if (data.length < 50) {
        setHasMore(false);
      }
      
      if (!reset) {
        setOffset(currentOffset + data.length);
      }
    } catch (err) {
      setError('Ошибка загрузки клиентов: ' + err.message);
      console.error('Ошибка загрузки клиентов:', err);
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [offset, selectedClient]);

  // Загрузка клиентов при монтировании
  useEffect(() => {
    loadClients(true);
  }, []);

  // Функция выбора клиента
  const handleClientSelect = async (client) => {
    setLoading(true);
    setError(null);
    try {
      const cardData = await api.getClientCard(client.external_id);
      
      // Упорядочиваем top_features по убыванию абсолютного значения shap_value
      if (cardData.explanation && cardData.explanation.top_features) {
        cardData.explanation.top_features = cardData.explanation.top_features
          .sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value))
          .map(feature => ({
            ...feature,
            title: getFeatureTitle(feature.title)
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

  // Функция обработки загрузки файла
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

  // Функция опроса статуса импорта
  const pollImportStatus = async (jobId) => {
    const interval = setInterval(async () => {
      try {
        const status = await api.getImportStatus(jobId);
        
        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(interval);
          if (status.status === 'completed') {
            // Перезагружаем список после импорта
            loadClients(true);
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

  // Функция обработки скролла для бесконечной подгрузки
  const handleScroll = (e) => {
    const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;
    
    // Если прокрутили до конца и есть еще данные для загрузки
    if (scrollHeight - scrollTop <= clientHeight + 100 && hasMore && !loadingMore) {
      loadClients();
    }
  };

  // Функция поиска клиентов
  const handleSearch = (query) => {
    setSearchQuery(query);
  };

  // Фильтрация клиентов по поисковому запросу
  const filteredClients = clients.filter(client => 
    client.external_id.toLowerCase().includes(searchQuery.toLowerCase()) ||
    (client.display_name && client.display_name.toLowerCase().includes(searchQuery.toLowerCase()))
  );

  return (
    <div className='page main-page'>
      {/* Хедер */}
      <div className='header'>
        <input 
          type="text" 
          placeholder="Поиск по ID или имени..."
          className='search-input'
          value={searchQuery}
          onChange={(e) => handleSearch(e.target.value)}
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
          </div>
          <div 
            className='users-container'
            onScroll={handleScroll}
          >
            {error && (
              <div className="error-message">
                {error}
              </div>
            )}
            
            {filteredClients.map(client => (
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
            
            {filteredClients.length === 0 && !loading && (
              <div className="no-clients">
                Клиенты не найдены
              </div>
            )}
            
            {loadingMore && (
              <div className="loading-more">
                Загрузка...
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
                {/* Карточка с объяснением предсказания */}
                <div className='main-info-card card-uw'>
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
                <div className='main-info-card card-uw'>
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
          </div>
        </div>
      </div>
    </div>
  );
}

export default Main;