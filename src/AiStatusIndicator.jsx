import React from 'react';
import './AiStatusIndicator.css';

function AiStatusIndicator({ status = 'offline' }) {
  // Determine the status text and color based on the status prop
  const getStatusInfo = () => {
    switch(status.toLowerCase()) {
      case 'offline':
        return { text: 'Offline', color: '#e74c3c' }; // Red
      case 'connecting':
        return { text: 'Connecting...', color: '#f39c12' }; // Yellow/Orange
      case 'ready':
        return { text: 'Ready', color: '#2ecc71' }; // Green
      default:
        return { text: status, color: '#95a5a6' }; // Gray for unknown status
    }
  };

  const statusInfo = getStatusInfo();

  return (
    <div className="ai-status-container">
      <span className="ai-status-label">AI Status:</span>
      <div className="ai-status-indicator">
        <div 
          className="status-dot" 
          style={{ backgroundColor: statusInfo.color }}
        ></div>
        <span className="status-text">{statusInfo.text}</span>
      </div>
    </div>
  );
}

export default AiStatusIndicator;