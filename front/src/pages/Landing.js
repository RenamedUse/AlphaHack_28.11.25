import React from 'react';
import { useNavigate } from 'react-router-dom';
import './Landing.css';

function Landing() {
  const navigate = useNavigate();

  return (
    <div className="landing-page">
      <div className="hero-section">
        
        {/* Красные акцентные элементы */}
        <div className="accent-elements">
          <div className="accent-line line-1"></div>
          <div className="accent-line line-2"></div>
          <div className="accent-circle circle-1"></div>
          <div className="accent-circle circle-2"></div>
        </div>

        {/* Основной контент */}
        <div className="hero-content">
          <div className="title-block">
            <h1 className="main-title">
              Команда <span className="team-name">Тунтунпобедил</span>
            </h1>
            <div className="presents-text">представляет</div>
            
            <div className="subtitle-stack">
              <div className="strikethrough-container">
                <h2 className="strikethrough-text">обученную модель</h2>
                <div className="strikethrough-line"></div>
              </div>
              <h2 className="main-accent">РЕВОЛЮЦИОННЫЙ ИНСТРУМЕНТ</h2>
              <h2 className="sub-accent">для анализа дохода</h2>
            </div>
          </div>

          {/* CTA кнопка */}
          <button 
            className="cta-button"
            onClick={() => navigate('/main')}
          >
            <span className="button-text">ПОПРОБОВАТЬ СЕЙЧАС</span>
            <div className="button-hover"></div>
          </button>

          {/* Блок авторов */}
          <div className="authors-section">
            <div className="authors-title">Авторы проекта</div>
            <div className="authors-grid">
              <div className="author-card">
                <div className="author-name">Александр Савин</div>
              </div>
              <div className="author-card">
                <div className="author-name">Игорь Бенберин</div>
              </div>
              <div className="author-card">
                <div className="author-name">Антон Сысоев</div>
              </div>
            </div>
          </div>
        </div>

        {/* Декоративные элементы в стиле Альфа */}
        <div className="alpha-elements">
          <div className="alpha-dot dot-1"></div>
          <div className="alpha-dot dot-2"></div>
          <div className="alpha-dot dot-3"></div>
          <div className="alpha-dot dot-4"></div>
        </div>
      </div>
    </div>
  );
}

export default Landing;