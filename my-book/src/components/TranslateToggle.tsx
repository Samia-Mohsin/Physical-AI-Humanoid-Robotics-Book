import React from 'react';
import { useLanguage } from '../contexts/LanguageContext';

const TranslateToggle: React.FC = () => {
  const { language, setLanguage } = useLanguage();

  const languages = [
    { code: 'en', name: 'English' },
    { code: 'ur', name: 'Urdu' },
    { code: 'roman-ur', name: 'Roman Urdu' }
  ];

  return (
    <div className="translate-toggle" style={{ display: 'flex', gap: '4px' }}>
      {languages.map((lang) => (
        <button
          key={lang.code}
          className={`lang-btn ${language === lang.code ? 'active' : ''}`}
          onClick={() => setLanguage(lang.code)}
          style={{
            backgroundColor: language === lang.code ? '#4f46e5' : '#e5e7eb',
            color: language === lang.code ? 'white' : '#374151',
            border: 'none',
            borderRadius: '4px',
            padding: '4px 8px',
            cursor: 'pointer',
            fontSize: '12px',
            transition: 'background-color 0.2s'
          }}
        >
          {lang.name}
        </button>
      ))}
    </div>
  );
};

export default TranslateToggle;