import { useState, useRef, useEffect } from 'react';
import { Upload, Image as ImageIcon, RefreshCw, Calculator, AlertCircle, CheckCircle } from 'lucide-react';
import { apiService } from '../services/api';

function Calibration() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [imageDimensions, setImageDimensions] = useState({ width: 1, height: 1 });
  const [points, setPoints] = useState([]); // Original-image coordinates
  const [distance, setDistance] = useState('');
  const [isCalculating, setIsCalculating] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  
  const fileInputRef = useRef(null);
  const imageRef = useRef(null);

  const handleFileSelect = (file) => {
    if (!file) return;
    
    // Reset all state
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setImageDimensions({ width: 1, height: 1 });
    setPoints([]);
    setDistance('');
    setResults(null);
    setError(null);
  };

  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  const handleImageClick = (e) => {
    if (points.length >= 2 || !imageRef.current) return;

    const rect = imageRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Calculate original image coordinates
    const scaleX = imageRef.current.naturalWidth / rect.width;
    const scaleY = imageRef.current.naturalHeight / rect.height;
    
    const originalX = x * scaleX;
    const originalY = y * scaleY;

    setPoints(prev => [...prev, {
      x: originalX, 
      y: originalY
    }]);
    
    setError(null);
  };

  const handleResetPoints = () => {
    setPoints([]);
    setResults(null);
    setError(null);
  };

  const handleCalculate = async () => {
    if (points.length !== 2) {
      setError("Please select exactly two points on the image.");
      return;
    }
    
    const distVal = parseFloat(distance);
    if (isNaN(distVal) || distVal <= 0) {
      setError("Please enter a valid positive distance in millimetres.");
      return;
    }

    setIsCalculating(true);
    setError(null);

    try {
      const payload = {
        point_1_x: points[0].x,
        point_1_y: points[0].y,
        point_2_x: points[1].x,
        point_2_y: points[1].y,
        real_world_distance_mm: distVal,
        image_filename: selectedFile.name
      };
      
      const res = await apiService.calculateCalibration(payload);
      setResults(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsCalculating(false);
    }
  };

  return (
    <div className="page-calibration">
      <h1 className="page-title">Scale Calibration</h1>
      <p className="page-description">Manually calibrate the pixel-to-millimetre scale by selecting two points of a known distance.</p>

      {error && (
        <div style={{ padding: '16px', backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', borderRadius: '8px', color: '#ef4444', marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {results && (
        <div style={{ padding: '16px', backgroundColor: 'rgba(16, 185, 129, 0.1)', border: '1px solid rgba(16, 185, 129, 0.2)', borderRadius: '8px', color: '#10b981', marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <CheckCircle size={20} />
          <span>Calibration successful! Metadata saved successfully.</span>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '24px', alignItems: 'start' }}>
        
        {/* Left Column: Image Viewer */}
        <div className="card" style={{ padding: '16px', minHeight: '500px', display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: '500' }}>Calibration Image</h3>
            <button className="btn btn-outline" onClick={() => fileInputRef.current?.click()} style={{ padding: '6px 12px', fontSize: '0.9rem' }}>
              <Upload size={16} style={{ marginRight: '6px' }} /> Upload Image
            </button>
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={(e) => handleFileSelect(e.target.files[0])} 
              style={{ display: 'none' }} 
              accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff"
            />
          </div>

          <div style={{ flex: 1, backgroundColor: '#000', borderRadius: '8px', overflow: 'hidden', position: 'relative', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {!previewUrl ? (
              <div style={{ color: 'var(--text-secondary)', textAlign: 'center' }}>
                <ImageIcon size={48} style={{ opacity: 0.5, margin: '0 auto 16px auto' }} />
                <p>No image selected.</p>
                <p style={{ fontSize: '0.85rem', opacity: 0.7 }}>Upload an image with a visible reference scale.</p>
              </div>
            ) : (
              <div style={{ position: 'relative', display: 'inline-block' }}>
                <img 
                  ref={imageRef}
                  src={previewUrl} 
                  alt="Calibration reference" 
                  onClick={handleImageClick}
                  onLoad={(event) => setImageDimensions({
                    width: event.currentTarget.naturalWidth,
                    height: event.currentTarget.naturalHeight,
                  })}
                  style={{ 
                    maxWidth: '100%', 
                    maxHeight: '600px', 
                    objectFit: 'contain',
                    cursor: points.length < 2 ? 'crosshair' : 'default',
                    display: 'block'
                  }} 
                  draggable={false}
                />
                
                {/* SVG Overlay for markers and lines */}
                <svg
                  viewBox={`0 0 ${imageDimensions.width} ${imageDimensions.height}`}
                  preserveAspectRatio="none"
                  style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
                >
                  {points.length === 2 && (
                    <line 
                      x1={points[0].x}
                      y1={points[0].y}
                      x2={points[1].x}
                      y2={points[1].y}
                      stroke="#10b981" 
                      strokeWidth="2" 
                      strokeDasharray="4 4"
                    />
                  )}
                  {points.map((p, idx) => (
                    <g key={idx}>
                      <circle cx={p.x} cy={p.y} r="5" fill="#ef4444" vectorEffect="non-scaling-stroke" />
                      <text x={p.x + 10} y={p.y - 10} fill="#ef4444" fontSize="14" fontWeight="bold" style={{ textShadow: '0 1px 3px rgba(0,0,0,0.8)' }}>
                        P{idx + 1}
                      </text>
                    </g>
                  ))}
                </svg>
              </div>
            )}
          </div>
          
          {previewUrl && (
            <div style={{ marginTop: '16px', display: 'flex', justifyContent: 'center', gap: '8px', fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
              <span>Status:</span>
              <span style={{ color: points.length < 2 ? 'var(--accent-primary)' : '#10b981', fontWeight: '500' }}>
                {points.length === 0 && "Click to place Point 1"}
                {points.length === 1 && "Click to place Point 2"}
                {points.length === 2 && "Points selected. Ready to calculate."}
              </span>
            </div>
          )}
        </div>

        {/* Right Column: Controls & Results */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          <div className="card glass" style={{ padding: '24px' }}>
            <h3 style={{ fontSize: '1.1rem', fontWeight: '500', marginBottom: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Calculator size={18} className="text-primary" /> Settings
            </h3>
            
            <div style={{ marginBottom: '20px' }}>
              <label style={{ display: 'block', fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                Selected Points
              </label>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '0.85rem' }}>
                <div style={{ backgroundColor: 'rgba(255,255,255,0.05)', padding: '8px 12px', borderRadius: '4px', borderLeft: '3px solid #ef4444' }}>
                  Point 1: {points[0] ? `(${Math.round(points[0].x)}, ${Math.round(points[0].y)})` : 'Not set'}
                </div>
                <div style={{ backgroundColor: 'rgba(255,255,255,0.05)', padding: '8px 12px', borderRadius: '4px', borderLeft: '3px solid #ef4444' }}>
                  Point 2: {points[1] ? `(${Math.round(points[1].x)}, ${Math.round(points[1].y)})` : 'Not set'}
                </div>
              </div>
              <button 
                className="btn btn-outline" 
                style={{ width: '100%', marginTop: '12px', fontSize: '0.85rem', padding: '6px' }}
                onClick={handleResetPoints}
                disabled={points.length === 0}
              >
                <RefreshCw size={14} style={{ marginRight: '6px' }} /> Reset Points
              </button>
            </div>

            <div style={{ marginBottom: '24px' }}>
              <label style={{ display: 'block', fontSize: '0.9rem', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                Real-World Distance (mm)
              </label>
              <input 
                type="number" 
                value={distance}
                onChange={(e) => setDistance(e.target.value)}
                placeholder="e.g. 10.0"
                style={{ 
                  width: '100%', 
                  padding: '10px 12px', 
                  borderRadius: '6px', 
                  border: '1px solid var(--border-color)', 
                  backgroundColor: 'rgba(0,0,0,0.2)',
                  color: 'white',
                  fontSize: '1rem'
                }}
              />
            </div>

            <button 
              className="btn btn-primary" 
              style={{ width: '100%' }}
              onClick={handleCalculate}
              disabled={points.length !== 2 || !distance || isCalculating}
            >
              {isCalculating ? 'Calculating...' : 'Calculate Calibration'}
            </button>
          </div>

          {results && (
            <div className="card" style={{ padding: '24px', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
              <h3 style={{ fontSize: '1.1rem', fontWeight: '500', marginBottom: '16px', color: '#10b981' }}>Calibration Results</h3>
              
              <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                <div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Pixel Distance</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: '500' }}>{results.pixel_distance.toFixed(2)} px</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Scale (mm/pixel)</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: '500', color: 'var(--accent-primary)' }}>{results.mm_per_pixel.toFixed(6)} mm/px</div>
                </div>
                <div>
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>Density (pixels/mm)</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: '500' }}>{results.pixels_per_mm.toFixed(2)} px/mm</div>
                </div>
              </div>
            </div>
          )}
          
        </div>
      </div>
    </div>
  );
}

export default Calibration;
