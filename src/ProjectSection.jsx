import React, { useState, useEffect } from 'react';
import CameraFeed from './CameraFeed';
import TranslationDisplay from './TranslationDisplay';
import AiStatusIndicator from './AiStatusIndicator';
import './ProjectSection.css';

function ProjectSection() {
  // State to manage the simulated AI status
  const [aiStatus, setAiStatus] = useState('connecting'); // Start with connecting status

  // Simulate AI connection status changes
  useEffect(() => {
    // After a short delay, change status to ready
    const timer = setTimeout(() => {
      setAiStatus('ready');
    }, 3000); // Change to ready after 3 seconds

    // Cleanup the timer
    return () => clearTimeout(timer);
  }, []);

  return (
    <section className="project-section">
      <h2 className="section-title">Real-Time ISL Translation</h2>
      
      {/* Container for the main content */}
      <div className="project-content">
        {/* Camera feed on the left column (or top on mobile) */}
        <div className="camera-column">
          <CameraFeed />
        </div>
        
        {/* Translation display and AI status on the right column (or bottom on mobile) */}
        <div className="translation-column">
          <TranslationDisplay />
          <AiStatusIndicator status={aiStatus} />
        </div>
      </div>
    </section>
  );
}

export default ProjectSection;