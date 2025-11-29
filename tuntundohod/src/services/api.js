const API_BASE = process.env.REACT_APP_API_URL || 'http://89.208.14.227:8000/api';

// Вспомогательная функция для обработки ответов
const handleResponse = async (response) => {
  if (!response.ok) {
    const error = await response.text();
    throw new Error(error || `HTTP error! status: ${response.status}`);
  }
  return response.json();
};

export const api = {
  // Импорт CSV
  importCSV: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${API_BASE}/imports/income-csv`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse(response);
  },

  getImportStatus: async (jobId) => {
    const response = await fetch(`${API_BASE}/imports/${jobId}`);
    return handleResponse(response);
  },

  // Получение списка клиентов
  getClients: async (filters = {}) => {
    const params = new URLSearchParams();
    
    // Добавляем фильтры согласно OpenAPI
    const allowedFilters = ['segment_code', 'min_income', 'max_income', 'limit', 'offset'];
    allowedFilters.forEach(key => {
      if (filters[key] !== undefined && filters[key] !== null && filters[key] !== '') {
        params.append(key, filters[key]);
      }
    });
    
    const url = `${API_BASE}/clients${params.toString() ? `?${params.toString()}` : ''}`;
    const response = await fetch(url);
    return handleResponse(response);
  },

  // Получение карточки клиента
  getClientCard: async (externalId) => {
    const response = await fetch(`${API_BASE}/clients/${externalId}/card`);
    return handleResponse(response);
  },

  // Симуляция
  simulate: async (data) => {
    const response = await fetch(`${API_BASE}/simulate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });
    return handleResponse(response);
  },

  // Мониторинг
  getModelHealth: async () => {
    const response = await fetch(`${API_BASE}/monitoring/model-health`);
    return handleResponse(response);
  }
};