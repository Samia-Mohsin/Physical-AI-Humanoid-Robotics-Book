import React, { useState } from 'react';
import { useUser } from '../contexts/UserContext';

const PersonalizeButton: React.FC = () => {
  const { user, updateUserPreferences } = useUser();
  const [isOpen, setIsOpen] = useState(false);

  const handlePreferenceChange = (preference: string, value: string) => {
    if (updateUserPreferences) {
      updateUserPreferences({ [preference]: value });
    }
  };

  if (!user) {
    return (
      <button
        className="personalize-btn"
        onClick={() => alert('Please log in to access personalization features')}
        style={{
          backgroundColor: '#10b981',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          padding: '8px 12px',
          cursor: 'pointer',
          fontSize: '14px'
        }}
      >
        Personalize
      </button>
    );
  }

  return (
    <div className="personalize-container" style={{ position: 'relative', display: 'inline-block' }}>
      <button
        className="personalize-btn"
        onClick={() => setIsOpen(!isOpen)}
        style={{
          backgroundColor: '#10b981',
          color: 'white',
          border: 'none',
          borderRadius: '6px',
          padding: '8px 12px',
          cursor: 'pointer',
          fontSize: '14px'
        }}
      >
        Personalize
      </button>

      {isOpen && (
        <div
          className="personalize-dropdown"
          style={{
            position: 'absolute',
            top: '100%',
            left: '0',
            backgroundColor: 'white',
            border: '1px solid #e5e7eb',
            borderRadius: '6px',
            padding: '12px',
            boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
            zIndex: 1000,
            minWidth: '200px'
          }}
        >
          <div className="preference-section" style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px', fontWeight: 'bold' }}>
              Learning Path:
            </label>
            <select
              value={user.preferences.learningPath}
              onChange={(e) => handlePreferenceChange('learningPath', e.target.value)}
              style={{
                width: '100%',
                padding: '4px',
                border: '1px solid #d1d5db',
                borderRadius: '4px'
              }}
            >
              <option value="beginner">Beginner</option>
              <option value="intermediate">Intermediate</option>
              <option value="advanced">Advanced</option>
            </select>
          </div>

          <div className="preference-section" style={{ marginBottom: '12px' }}>
            <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px', fontWeight: 'bold' }}>
              Experience Level:
            </label>
            <select
              value={user.preferences.experienceLevel}
              onChange={(e) => handlePreferenceChange('experienceLevel', e.target.value)}
              style={{
                width: '100%',
                padding: '4px',
                border: '1px solid #d1d5db',
                borderRadius: '4px'
              }}
            >
              <option value="beginner">Beginner</option>
              <option value="intermediate">Intermediate</option>
              <option value="expert">Expert</option>
            </select>
          </div>

          <div className="preference-section">
            <label style={{ display: 'block', marginBottom: '4px', fontSize: '12px', fontWeight: 'bold' }}>
              Preferred Language:
            </label>
            <select
              value={user.preferences.language}
              onChange={(e) => handlePreferenceChange('language', e.target.value)}
              style={{
                width: '100%',
                padding: '4px',
                border: '1px solid #d1d5db',
                borderRadius: '4px'
              }}
            >
              <option value="en">English</option>
              <option value="ur">Urdu</option>
              <option value="roman-ur">Roman Urdu</option>
            </select>
          </div>
        </div>
      )}
    </div>
  );
};

export default PersonalizeButton;