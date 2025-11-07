import React from 'react';
import './TranslationDisplay.css';

function TranslationDisplay({ translation = "Translation will appear here..." }) {
  return (
    <div className="translation-container">
      <h3 className="translation-title">Translation Output</h3>
      <div className="translation-display">
        <p className="translation-text">{translation}</p>
      </div>
    </div>
  );
}

export default TranslationDisplay;