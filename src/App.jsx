import React, { useState } from 'react';
import Header from './Header';
import AboutSection from './AboutSection';
import ProjectSection from './ProjectSection';
import './App.css';

function App() {
  // State to manage which section is currently active
  const [activeSection, setActiveSection] = useState('about'); // Start with About section

  return (
    <div className="app">
      {/* Header component with navigation */}
      <Header activeSection={activeSection} setActiveSection={setActiveSection} />
      
      {/* Main content area */}
      <main className="main-content">
        {/* Render the appropriate section based on state */}
        {activeSection === 'about' ? <AboutSection /> : <ProjectSection />}
      </main>
    </div>
  );
}

export default App;