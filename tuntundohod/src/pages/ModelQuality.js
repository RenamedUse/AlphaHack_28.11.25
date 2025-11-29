import React from 'react';
import { useNavigate } from 'react-router-dom';

function ModelQuality() {
  const navigate = useNavigate();

  return (
    <div className="page">
      <h1>Качество работы модели</h1>
      <p>Здесь будет визуализация качества работы модели</p>
      <button onClick={() => navigate('/main')}>
        Назад на главную
      </button>
    </div>
  );
}

export default ModelQuality;