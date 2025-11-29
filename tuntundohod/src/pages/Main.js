import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Main.css';

function Main() {
  const navigate = useNavigate();

  return (
    <div className='page main-page'>
      {/* Хедер */}
      <div className='header'>
        <input 
          type="text" 
          placeholder="Поиск клиентов..."
          className='search-input'
        />
        <div className='buttons-container'>
            <button className='upload-btn'>
                Загрузить файл
            </button>
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
          <div className='users-container'>
            <div className='user-item active'>
              <p className='user-name'>Иван Иванов</p>
              <p className='user-details'>Доход: 120 000 ₽ • High</p>
            </div>
            <div className='user-item'>
              <p className='user-name'>Петр Петров</p>
              <p className='user-details'>Доход: 85 000 ₽ • Middle</p>
            </div>
            <div className='user-item'>
              <p className='user-name'>Мария Сидорова</p>
              <p className='user-details'>Доход: 65 000 ₽ • Middle</p>
            </div>
            <div className='user-item'>
              <p className='user-name'>Алексей Козлов</p>
              <p className='user-details'>Доход: 45 000 ₽ • Low</p>
            </div>
            <div className='user-item'>
              <p className='user-name'>Елена Новикова</p>
              <p className='user-details'>Доход: 150 000 ₽ • Premium</p>
            </div>
          </div>
        </div>

        {/* Правая колонка - основная информация (75%) */}
        <div className='main-info'>
          {/* Сетка из 3 колонок */}
          <div className='main-info-grid'>
            
            {/* Карточка 1 - занимает 2 колонки (card-w) */}
            <div className='main-info-card'>
              <div className='card-header'>
                <h3>Иван Иванов</h3>
              </div>
              <div className='card-content'>
                <div className='income-section'>
                  <div className='income-amount'>120 000 ₽</div>
                  <div className='income-segment high'>High</div>
                </div>
                <div className='client-meta'>
                  <p><strong>Возраст:</strong> 35 лет</p>
                  <p><strong>Стаж:</strong> 7 лет</p>
                  <p><strong>Образование:</strong> Высшее</p>
                </div>
              </div>
            </div>

            {/* Карточка 3 - занимает 3 колонки (card-uw) */}
            <div className='main-info-card card-w'>
              <div className='card-header'>
                <h4>Объяснение предсказания</h4>
              </div>
              <div className='card-content'>
                <p>Модель оценила доход в 120 000 ₽. Наибольший вклад дали работа в IT (+15 000 ₽) и стаж 7 лет (+10 000 ₽).</p>
                <div className='features-list'>
                  <div className='feature-item'>
                    <span>Работа в IT</span>
                    <span className='feature-impact'>+15 000 ₽</span>
                  </div>
                  <div className='feature-item'>
                    <span>Стаж работы</span>
                    <span className='feature-impact'>+10 000 ₽</span>
                  </div>
                  <div className='feature-item'>
                    <span>Высшее образование</span>
                    <span className='feature-impact'>+8 000 ₽</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Карточка 4 - занимает 2 колонки (card-w) */}
            <div className='main-info-card card-w'>
              <div className='card-header'>
                <h4>Рекомендуемые продукты</h4>
              </div>
              <div className='card-content'>
                <div className='product-item'>
                  <strong>Золотая кредитная карта</strong>
                  <p>Для клиентов high/premium с доходом выше 100 000 ₽</p>
                </div>
                <div className='product-item'>
                  <strong>Вклад на 12 месяцев</strong>
                  <p>Для клиентов со стабильным доходом</p>
                </div>
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>
  );
}

export default Main;