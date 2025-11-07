import React from 'react';
import './Header.css';

function Header({ activeSection, setActiveSection }) {
  return (
    <header className="header">
      {/* Project title */}
      <div className="header-title">
        <h1>Real-Time ISL Translator</h1>
      </div>
      
      {/* Navigation buttons/tabs */}
      <nav className="navigation">
        <button 
          className={`nav-button ${activeSection === 'about' ? 'active' : ''}`}
          onClick={() => setActiveSection('about')}
        >
          About
        </button>
        <button 
          className={`nav-button ${activeSection === 'project' ? 'active' : ''}`}
          onClick={() => setActiveSection('project')}
        >
          Actual Project
        </button>
      </nav>
    </header>
  );
}

export default Header;