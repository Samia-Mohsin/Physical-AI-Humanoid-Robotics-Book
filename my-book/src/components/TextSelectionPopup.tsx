import React, { useState, useEffect, useRef } from 'react';

interface TextSelectionPopupProps {
  onSelectAction: (action: 'explain' | 'translate' | 'save', selectedText: string) => void;
}

const TextSelectionPopup: React.FC<TextSelectionPopupProps> = ({ onSelectAction }) => {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [selectedText, setSelectedText] = useState('');
  const popupRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleSelection = () => {
      const selection = window.getSelection();
      const text = selection?.toString().trim() || '';

      if (text.length > 0 && text.length < 500) { // Limit selection length
        const range = selection?.getRangeAt(0);
        const rect = range?.getBoundingClientRect();

        if (rect) {
          // Position the popup above the selection
          setPosition({
            x: rect.left + window.scrollX,
            y: rect.top + window.scrollY - 40 // 40px above the selection
          });
          setSelectedText(text);
          setIsVisible(true);
        }
      } else {
        setIsVisible(false);
      }
    };

    const handleClickOutside = (event: MouseEvent) => {
      if (popupRef.current && !(popupRef.current as HTMLElement).contains(event.target as Node)) {
        setIsVisible(false);
      }
    };

    document.addEventListener('mouseup', handleSelection);
    document.addEventListener('click', handleClickOutside);

    return () => {
      document.removeEventListener('mouseup', handleSelection);
      document.removeEventListener('click', handleClickOutside);
    };
  }, []);

  const handleAction = (action: 'explain' | 'translate' | 'save') => {
    onSelectAction(action, selectedText);
    setIsVisible(false);
  };

  if (!isVisible || !selectedText) {
    return null;
  }

  return (
    <div
      ref={popupRef}
      className="text-selection-popup"
      style={{
        position: 'absolute',
        left: position.x,
        top: position.y,
        zIndex: 10000,
        backgroundColor: '#4f46e5',
        color: 'white',
        borderRadius: '6px',
        padding: '4px',
        fontSize: '14px',
        boxShadow: '0 4px 6px rgba(0, 0, 0, 0.1)',
        display: 'flex',
        gap: '2px'
      }}
    >
      <button
        className="popup-btn"
        onClick={() => handleAction('explain')}
        title="Explain this text"
        style={{
          background: 'none',
          border: 'none',
          color: 'white',
          padding: '6px 8px',
          cursor: 'pointer',
          borderRadius: '4px',
          fontSize: '12px'
        }}
      >
        Explain
      </button>
      <button
        className="popup-btn"
        onClick={() => handleAction('translate')}
        title="Translate this text"
        style={{
          background: 'none',
          border: 'none',
          color: 'white',
          padding: '6px 8px',
          cursor: 'pointer',
          borderRadius: '4px',
          fontSize: '12px'
        }}
      >
        Translate
      </button>
      <button
        className="popup-btn"
        onClick={() => handleAction('save')}
        title="Save this text"
        style={{
          background: 'none',
          border: 'none',
          color: 'white',
          padding: '6px 8px',
          cursor: 'pointer',
          borderRadius: '4px',
          fontSize: '12px'
        }}
      >
        Save
      </button>
    </div>
  );
};

export default TextSelectionPopup;