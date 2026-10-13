import { useState, useRef, useEffect, useMemo } from 'react';
import { Upload, Loader2, Download, AlertCircle, RefreshCw, X, CheckCircle, FileWarning } from 'lucide-react';
import { useProcessingJob } from '../hooks/useProcessingJob';

function ImageProcessing() {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const job = useProcessingJob('image');
  const batchStatus = job.busy ? 'processing' : job.status?.status === 'FAILED' ? 'error' : job.result ? 'completed' : 'idle';
  const batchResults = job.result?.preprocessing;
  const error = job.error;
  const displayedFiles = job.status?.files || selectedFiles;
  const fileInputRef = useRef(null);

  const previewUrls = useMemo(
    () => Object.fromEntries(
      selectedFiles.map((file) => [file.name, URL.createObjectURL(file)]),
    ),
    [selectedFiles],
  );

  useEffect(
    () => () => Object.values(previewUrls).forEach((url) => URL.revokeObjectURL(url)),
    [previewUrls],
  );

  const handleFileSelect = (files) => {
    if (!files || files.length === 0 || job.busy) return;
    
    const fileArray = Array.from(files);
    
    // Add only new files (prevent duplicates by name)
    setSelectedFiles(existing => {
      const prev = job.jobId ? [] : existing;
      const existingNames = new Set(prev.map(f => f.name));
      const newFiles = fileArray.filter(f => !existingNames.has(f.name));
      return [...prev, ...newFiles];
    });
    
    job.clear();
  };

  const removeFile = (fileName) => {
    setSelectedFiles(prev => prev.filter(f => f.name !== fileName));
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files);
      e.dataTransfer.clearData();
    }
  };

  const handleProcessBatch = async () => {
    if (selectedFiles.length === 0) return;
    await job.start(selectedFiles);
  };

  const resetUpload = () => {
    setSelectedFiles([]);
    job.clear();
  };

  const handleDownload = async (url, filename) => {
    if (!url) return;
    try {
      const response = await fetch(url);
      const blob = await response.blob();
      const localUrl = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = localUrl;
      link.download = `processed_${filename}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(localUrl);
    } catch (err) {
      console.error("Download failed", err);
    }
  };

  return (
    <div className="page-processing">
      <h1 className="page-title">Image Processing</h1>
      <p className="page-description">Module 1: Pre-process cloud chamber images to isolate particle tracks.</p>

      {error && (
        <div style={{ padding: '16px', backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.2)', borderRadius: '8px', color: '#ef4444', marginBottom: '24px', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <AlertCircle size={20} />
          <span>{error}</span>
        </div>
      )}

      {selectedFiles.length === 0 && batchStatus === 'idle' ? (
        // Multi-Upload Zone
        <div 
          className="card glass" 
          style={{ 
            textAlign: 'center', 
            padding: '64px 24px', 
            borderStyle: 'dashed', 
            borderWidth: '2px', 
            borderColor: isDragging ? 'var(--accent-primary)' : 'var(--border-color)', 
            backgroundColor: isDragging ? 'rgba(59, 130, 246, 0.05)' : 'rgba(255, 255, 255, 0.02)',
            transition: 'all 0.2s ease',
            cursor: 'pointer'
          }}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
          onClick={() => fileInputRef.current?.click()}
        >
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={(e) => handleFileSelect(e.target.files)} 
            style={{ display: 'none' }} 
            accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff"
            multiple
          />
          <div style={{ display: 'inline-flex', alignItems: 'center', justifyContent: 'center', width: '64px', height: '64px', borderRadius: '50%', backgroundColor: 'rgba(59, 130, 246, 0.1)', color: 'var(--accent-primary)', marginBottom: '24px' }}>
            <Upload size={32} />
          </div>
          <h3 style={{ fontSize: '1.25rem', fontWeight: '500', marginBottom: '8px' }}>Upload Images for Processing</h3>
          <p style={{ color: 'var(--text-secondary)', marginBottom: '24px', maxWidth: '400px', margin: '0 auto 24px auto' }}>
            Drag and drop multiple raw cloud chamber images here, or click to browse.
          </p>
          <button className="btn btn-primary" onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}>
            Select Files
          </button>
        </div>
      ) : (
        // Workspace Zone (List, Processing, Results)
        <div style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <h2 style={{ fontSize: '1.25rem', fontWeight: '500' }}>
                {batchStatus === 'completed' ? 'Processing Complete' : 'Selected Images'}
              </h2>
              <span style={{ backgroundColor: 'rgba(255,255,255,0.1)', padding: '2px 8px', borderRadius: '12px', fontSize: '0.85rem' }}>
                {displayedFiles.length} {displayedFiles.length === 1 ? 'file' : 'files'}
              </span>
            </div>
            
            <div style={{ display: 'flex', gap: '12px' }}>
              <button className="btn btn-outline" onClick={resetUpload} disabled={batchStatus === 'processing'}>
                <RefreshCw size={16} /> Start Over
              </button>
              
              {batchStatus !== 'completed' && (
                <button className="btn btn-primary" onClick={handleProcessBatch} disabled={batchStatus === 'processing' || selectedFiles.length === 0}>
                  {batchStatus === 'processing' ? (
                    <><Loader2 size={16} className="lucide-spin" style={{ animation: 'spin 2s linear infinite' }} /> Processing {displayedFiles.length || ''} images...</>
                  ) : (
                    `Process All (${selectedFiles.length})`
                  )}
                </button>
              )}
              
              {batchStatus === 'idle' && (
                <button 
                  className="btn btn-outline" 
                  onClick={() => fileInputRef.current?.click()}
                  style={{ borderStyle: 'dashed' }}
                >
                  + Add More
                </button>
              )}
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={(e) => handleFileSelect(e.target.files)} 
                style={{ display: 'none' }} 
                accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff"
                multiple
              />
            </div>
          </div>

          {/* Processing Indicator */}
          {batchStatus === 'processing' && (
             <div className="card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '48px', borderStyle: 'dashed', borderColor: 'var(--accent-primary)', backgroundColor: 'rgba(59, 130, 246, 0.02)' }}>
                <Loader2 size={48} className="lucide-spin" style={{ animation: 'spin 2s linear infinite', color: 'var(--accent-primary)', marginBottom: '16px' }} />
                <h3 style={{ fontSize: '1.2rem', fontWeight: '500', marginBottom: '8px' }}>Processing Batch...</h3>
                <p style={{ color: 'var(--text-secondary)' }}>{job.status ? `${job.status.progress_percent}% · ${job.status.stage} · Image ${job.status.current_frame} / ${job.status.total_frames}` : job.jobId ? 'Reconnecting to saved image job…' : 'Uploading images…'}</p>
             </div>
          )}

          {job.status?.files && <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
            {job.status.files.map((file) => <figure key={file.original_image_url}>
              <img src={file.original_image_url} alt={file.name} style={{ width: '140px', height: '100px', objectFit: 'contain' }} />
              <figcaption>{file.name} · {file.type || 'Image'}</figcaption>
            </figure>)}
          </div>}

          {/* Summary Banner */}
          {batchStatus === 'completed' && batchResults && (
            <div className="card glass" style={{ display: 'flex', gap: '24px', alignItems: 'center', padding: '24px' }}>
              <div style={{ flex: 1 }}>
                <h3 style={{ fontSize: '1.2rem', marginBottom: '8px' }}>Batch Summary</h3>
                <p style={{ color: 'var(--text-secondary)', margin: 0 }}>Processed a total of {batchResults.total_images} images.</p>
              </div>
              <div style={{ display: 'flex', gap: '16px' }}>
                <div style={{ backgroundColor: 'rgba(16, 185, 129, 0.1)', padding: '16px', borderRadius: '8px', minWidth: '120px', textAlign: 'center', border: '1px solid rgba(16, 185, 129, 0.2)' }}>
                  <div style={{ color: '#10b981', fontSize: '1.5rem', fontWeight: 'bold' }}>{batchResults.successful_count}</div>
                  <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Successful</div>
                </div>
                <div style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', padding: '16px', borderRadius: '8px', minWidth: '120px', textAlign: 'center', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                  <div style={{ color: '#ef4444', fontSize: '1.5rem', fontWeight: 'bold' }}>{batchResults.failed_count}</div>
                  <div style={{ color: 'var(--text-secondary)', fontSize: '0.85rem' }}>Failed</div>
                </div>
              </div>
            </div>
          )}

          {/* File List / Results Gallery */}
          {batchStatus !== 'processing' && (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(300px, 1fr))', gap: '16px' }}>
              
              {batchStatus === 'idle' && selectedFiles.map((file, idx) => (
                <div key={idx} className="card" style={{ padding: '16px', display: 'flex', alignItems: 'center', gap: '16px' }}>
                  <div style={{ width: '60px', height: '60px', borderRadius: '8px', overflow: 'hidden', backgroundColor: '#000', flexShrink: 0 }}>
                    <img src={previewUrls[file.name]} alt="preview" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontWeight: '500', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{file.name}</div>
                    <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{(file.size / 1024 / 1024).toFixed(2)} MB</div>
                  </div>
                  <button className="icon-btn" onClick={() => removeFile(file.name)} title="Remove file">
                    <X size={18} />
                  </button>
                </div>
              ))}

              {batchStatus === 'completed' && batchResults && batchResults.results.map((res, idx) => (
                <div key={idx} className="card" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px', border: res.status === 'success' ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(239, 68, 68, 0.3)' }}>
                  
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                    <div style={{ fontWeight: '500', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '80%' }} title={res.original_filename}>
                      {res.original_filename}
                    </div>
                    {res.status === 'success' ? (
                      <CheckCircle size={18} color="#10b981" />
                    ) : (
                      <FileWarning size={18} color="#ef4444" />
                    )}
                  </div>

                  {res.status === 'success' ? (
                    <>
                      <div style={{ width: '100%', height: '160px', borderRadius: '8px', overflow: 'hidden', backgroundColor: '#000' }}>
                        <img src={res.processed_image_url} alt="processed" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
                      </div>
                      <button className="btn btn-outline" style={{ width: '100%', marginTop: 'auto' }} onClick={() => handleDownload(res.processed_image_url, res.original_filename)}>
                        <Download size={16} /> Download
                      </button>
                    </>
                  ) : (
                    <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', backgroundColor: 'rgba(239, 68, 68, 0.05)', padding: '12px', borderRadius: '8px' }}>
                      <div style={{ color: '#ef4444', fontSize: '0.85rem', fontWeight: '500', marginBottom: '4px' }}>Processing Failed</div>
                      <div style={{ color: 'var(--text-secondary)', fontSize: '0.8rem', wordBreak: 'break-word' }}>{res.error_message}</div>
                    </div>
                  )}
                  
                </div>
              ))}

            </div>
          )}

        </div>
      )}
      
      <style dangerouslySetInnerHTML={{__html: `
        @keyframes spin { 100% { transform: rotate(360deg); } }
      `}} />
    </div>
  );
}

export default ImageProcessing;
