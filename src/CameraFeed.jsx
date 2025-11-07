import React, { useEffect, useRef, useState } from 'react';
import './CameraFeed.css';

function CameraFeed() {
  const videoRef = useRef(null);
  const [hasPermission, setHasPermission] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Function to access the user's webcam
    const getVideo = async () => {
      try {
        // Request access to the user's camera
        const stream = await navigator.mediaDevices.getUserMedia({ 
          video: { 
            width: { ideal: 1280 },
            height: { ideal: 720 }
          } 
        });
        
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setHasPermission(true);
        setError(null);
      } catch (err) {
        console.error('Error accessing camera:', err);
        setError(`Camera access denied: ${err.message || 'Unknown error'}`);
        setHasPermission(false);
      }
    };

    getVideo();

    // Clean up the video stream when component unmounts
    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="camera-container">
      <h3 className="camera-title">Live Camera Feed</h3>
      {error ? (
        <div className="camera-error">
          {error}
        </div>
      ) : hasPermission ? (
        <video 
          ref={videoRef} 
          autoPlay 
          playsInline 
          muted
          className="camera-video"
        />
      ) : (
        <div className="camera-loading">
          Requesting camera access...
        </div>
      )}
    </div>
  );
}

export default CameraFeed;