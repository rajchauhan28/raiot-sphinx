import React from 'react';
import './AboutSection.css';

function AboutSection() {
  return (
    <section className="about-section">
      {/* Title */}
      <h2 className="section-title">Real-Time ISL Translator</h2>
      
      {/* Subtitle */}
      <h3 className="section-subtitle">Bridging Communication Gaps</h3>
      
      {/* Description */}
      <p className="section-description">
        This project aims to develop a real-time Indian Sign Language (ISL) to multilingual translation system. 
        Our goal is to empower the deaf community in India by breaking down communication barriers in education, 
        employment, healthcare, and social interactions.
      </p>
      
      {/* Hackathon Context */}
      <div className="hackathon-context">
        <p>
          Developed for [Hackathon Name/Date] - A step towards social inclusion and accessibility.
        </p>
      </div>
      
      {/* AI Status */}
      <div className="ai-status">
        <p>
          <strong>Note:</strong> The AI backend for real-time sign recognition and translation is currently under development by our team.
        </p>
      </div>
    </section>
  );
}

export default AboutSection;