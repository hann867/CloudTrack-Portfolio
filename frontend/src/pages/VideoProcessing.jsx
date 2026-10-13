import { useEffect, useRef, useState } from 'react';
import {
  AlertCircle, Check, ChevronLeft, ChevronRight, Download, FileJson,
  FileText, Film, Loader2, Upload, X,
} from 'lucide-react';
import { apiService } from '../services/api';
import { useProcessingJob } from '../hooks/useProcessingJob';
import './VideoProcessing.css';

const SUPPORTED_VIDEO = /\.(mp4|avi|mov|mkv)$/i;
const PIPELINE_STAGES = [
  ['UPLOADING', 'Video uploaded'],
  ['READING_METADATA', 'Metadata read'],
  ['EXTRACTING_FRAMES', 'Frames extracted'],
  ['PREPROCESSING', 'Preprocessing'],
  ['DETECTING', 'Detecting tracks'],
  ['CLASSIFYING', 'Classifying particles'],
  ['BUILDING_VIDEO', 'Building output video'],
  ['FINALISING', 'Finalising results'],
];
const STAGE_LABELS = {
  UPLOADING: 'Uploading Video', READING_METADATA: 'Reading Metadata',
  EXTRACTING_FRAMES: 'Extracting Frames', PREPROCESSING: 'Preprocessing',
  DETECTING: 'Detecting Tracks', CLASSIFYING: 'Classifying Particles',
  BUILDING_VIDEO: 'Encoding Classified Video', FINALISING: 'Finalising Results',
  COMPLETED: 'Completed', FAILED: 'Processing Failed',
};

const formatSize = (bytes) => bytes < 1024 * 1024
  ? `${(bytes / 1024).toFixed(1)} KB`
  : `${(bytes / 1024 / 1024).toFixed(1)} MB`;
const formatFps = (fps) => Number(fps || 0).toLocaleString(undefined, { maximumFractionDigits: 3 });

function VideoProcessing() {
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const job = useProcessingJob('video');
  const { busy: isProcessing, status: progress, result } = job;
  const [selectionError, setError] = useState('');
  const error = selectionError || job.error;
  const displayedFile = progress?.filename ? { name: progress.filename, size: progress.size } : selectedFile;
  const displayedPreview = progress?.original_video_url || previewUrl;

  useEffect(() => () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }, [previewUrl]);

  const selectFile = (file) => {
    if (!file || isProcessing) return;
    if (!SUPPORTED_VIDEO.test(file.name)) {
      setError('Unsupported video format. Please choose an MP4, AVI, MOV, or MKV file.');
      return;
    }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    job.clear();
    setError('');
  };

  const clearFile = () => {
    if (isProcessing) return;
    setSelectedFile(null);
    setPreviewUrl('');
    job.clear();
    setError('');
  };

  const processVideo = () => {
    if (!selectedFile || isProcessing || job.jobId) return;
    setError('');
    void job.start(selectedFile);
  };

  const onDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    selectFile(event.dataTransfer.files?.[0]);
  };

  const openPicker = () => {
    if (!isProcessing) fileInputRef.current?.click();
  };

  return (
    <div className="video-page">
      <h1 className="page-title">Video Processing</h1>
      <p className="page-description">Analyse a 30 FPS presentation sequence with CloudTrack’s existing preprocessing, detection, and classification modules.</p>

      <section className="card glass video-upload-card">
        <div className="video-section-heading">
          <span className="video-heading-icon"><Film size={20} /></span>
          <div><span>New analysis</span><h2>Upload Video</h2></div>
        </div>
        <div
          className={`video-dropzone${isDragging ? ' is-dragging' : ''}${isProcessing ? ' is-disabled' : ''}`}
          onDragOver={(event) => { event.preventDefault(); if (!isProcessing) setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)} onDrop={onDrop} onClick={openPicker}
          onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') openPicker(); }}
          role="button" tabIndex={isProcessing ? -1 : 0} aria-disabled={isProcessing}
        >
          <input ref={fileInputRef} type="file"
            accept=".mp4,.avi,.mov,.mkv,video/mp4,video/quicktime,video/x-msvideo,video/x-matroska"
            disabled={isProcessing} onChange={(event) => { selectFile(event.target.files?.[0]); event.target.value = ''; }} />
          <span className="video-upload-icon"><Upload size={30} /></span>
          <h3>Drag &amp; Drop Video Here</h3>
          <p>Upload one source video. Processing starts only when you confirm below.</p>
          <button className="btn btn-outline" type="button" disabled={isProcessing}
            onClick={(event) => { event.stopPropagation(); openPicker(); }}>Browse File</button>
          <small>MP4, AVI, MOV, MKV</small>
        </div>

        {displayedFile && (
          <div className="video-selection">
            <div className="video-preview"><video src={displayedPreview} controls preload="metadata">Video preview is unavailable.</video></div>
            <div className="video-file-details"><span>Selected File</span><strong title={displayedFile.name}>{displayedFile.name}</strong><span>Size</span><strong>{displayedFile.size == null ? 'Not recorded' : formatSize(displayedFile.size)}</strong></div>
            <button className="video-clear" type="button" onClick={clearFile} disabled={isProcessing} aria-label="Remove selected video"><X size={19} /></button>
          </div>
        )}
        <button className="btn btn-primary video-process-button" type="button" onClick={processVideo} disabled={!selectedFile || isProcessing || !!job.jobId}>
          {isProcessing ? <><Loader2 className="video-spinner" size={18} /> Processing Video...</> : <><Film size={18} /> Process Video</>}
        </button>
      </section>

      {isProcessing && <ProcessingProgress progress={progress || { stage: job.jobId ? 'RESTORING' : 'UPLOADING', message: job.jobId ? 'Reconnecting to saved job…' : 'Uploading video…' }} />}
      {error && <div className="video-error" role="alert"><AlertCircle size={21} /><div><strong>{progress?.status === 'FAILED' ? 'Video processing failed' : 'Video status'}</strong><span>{error}</span></div></div>}
      {result && <VideoResult key={job.jobId} result={result} originalPreviewUrl={displayedPreview} />}
    </div>
  );
}

function ProcessingProgress({ progress }) {
  const completed = new Set(progress.completed_stages || []);
  const percent = Math.max(0, Math.min(100, Number(progress.progress_percent || 0)));
  return (
    <section className="card glass video-progress-card" aria-live="polite">
      <div className="video-progress-header"><div><span>Processing video</span><h2>{STAGE_LABELS[progress.stage] || progress.message}</h2></div><strong>{Math.round(percent)}%</strong></div>
      <div className="video-progress-track"><span style={{ width: `${percent}%` }} /></div>
      <p>{progress.message}</p>
      {progress.total_frames > 0 && <p className="video-frame-progress">Frame {progress.current_frame.toLocaleString()} / {progress.total_frames.toLocaleString()}</p>}
      <div className="video-pipeline-list">
        {PIPELINE_STAGES.map(([stage, label]) => {
          const isComplete = completed.has(stage);
          const isCurrent = progress.stage === stage;
          return <div key={stage} className={isComplete ? 'complete' : isCurrent ? 'current' : ''}>
            <span>{isComplete ? <Check size={14} /> : isCurrent ? <span className="video-current-dot" /> : <span className="video-wait-dot" />}</span>{label}
          </div>;
        })}
      </div>
    </section>
  );
}

function VideoResult({ result, originalPreviewUrl }) {
  const { video, processing, tracks, preview_frames: previewFrames, job_id: jobId } = result;
  const [frameDetail, setFrameDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState('');
  const [reportUrl, setReportUrl] = useState(result.downloads.report);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportError, setReportError] = useState('');
  const [playbackError, setPlaybackError] = useState('');

  const openFrame = async (frameIndex) => {
    setDetailLoading(true); setDetailError('');
    try { setFrameDetail(await apiService.getVideoFrameResult(jobId, frameIndex)); }
    catch (error) { setDetailError(error.message || 'Unable to load frame details.'); }
    finally { setDetailLoading(false); }
  };

  const generateReport = async () => {
    setReportLoading(true); setReportError('');
    try {
      const report = await apiService.generateVideoReport(jobId);
      setReportUrl(report.download_url);
      const link = document.createElement('a');
      link.href = report.download_url;
      link.download = report.filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
    }
    catch (error) { setReportError(error.message || 'Unable to generate report.'); }
    finally { setReportLoading(false); }
  };

  return (
    <div className="video-results">
      <section className="card glass video-analysis-result">
        <div className="video-analysis-header"><div><span>Analysis completed</span><h2>Video Analysis Result</h2></div></div>
        <div className="video-player-grid">
          <figure><video src={originalPreviewUrl || result.original_video_url} controls preload="metadata">Original video unavailable.</video><figcaption>Original Video</figcaption></figure>
          <figure><video key={result.classified_video_url} src={result.classified_video_url} controls preload="metadata" playsInline
            onError={() => setPlaybackError('Unable to play this output. Older MPEG-4 outputs must be encoded as H.264; downloading remains available.')}>
            Classified video unavailable.</video><figcaption>Classified Video · {formatFps(video.target_fps)} FPS</figcaption></figure>
        </div>
        {playbackError && <p className="video-report-error" role="alert">{playbackError}</p>}
      </section>

      <section className="card glass">
        <h2 className="video-result-title">Track Detections</h2>
        <div className="video-particle-grid">
          <div className="alpha"><span>Alpha</span><strong>{tracks.alpha.toLocaleString()}</strong></div>
          <div className="electron"><span>Electron</span><strong>{tracks.electron.toLocaleString()}</strong></div>
          <div><span>Total Track Detections</span><strong>{tracks.total_track_detections.toLocaleString()}</strong></div>
        </div>
        <p className="video-unique-note">Counts use {processing.unique_source_frames_processed.toLocaleString()} unique analysed source frames, so frame-rate conversion duplicates are not double-counted. Frames are analysed independently; these are track detections, not unique physical particles.</p>
      </section>

      <section className="card glass">
        <div className="video-section-heading"><span className="video-heading-icon"><Film size={20} /></span><div><span>Source and output</span><h2>Video Information</h2></div></div>
        <div className="video-info-grid">
          <Info label="Filename" value={video.filename} /><Info label="Video Duration" value={`${Number(video.duration_seconds).toFixed(2)} seconds`} />
          <Info label="Processing Duration" value={video.processing_duration_seconds == null ? 'Not recorded' : `${Number(video.processing_duration_seconds).toFixed(2)} seconds`} />
          <Info label="Original FPS" value={formatFps(video.original_fps)} /><Info label="Processing FPS" value={formatFps(video.target_fps)} />
          <Info label="Original Frames" value={video.original_frames.toLocaleString()} /><Info label="Processed Frames" value={video.output_frames.toLocaleString()} />
          <Info label="Resolution" value={`${video.width} × ${video.height}`} />
        </div>
        <h3 className="video-preview-title">Processing Summary</h3>
        <div className="video-info-grid">
          <Info label="Frames Extracted" value={processing.extracted_frames.toLocaleString()} />
          <Info label="Frames Preprocessed" value={processing.preprocessed_frames.toLocaleString()} />
          <Info label="Frames Analysed" value={processing.detected_frames.toLocaleString()} />
          <Info label="Frames Classified" value={processing.classified_frames.toLocaleString()} />
          <Info label="Failed Frames" value={processing.failed_frames.toLocaleString()} />
          <Info label="Output FPS" value={formatFps(video.target_fps)} />
        </div>
      </section>

      <section className="card glass">
        <h2 className="video-result-title">Classified Frame Preview</h2>
        <p className="video-section-copy">Representative frames from the presentation timeline. Select a frame to inspect its track results.</p>
        <div className="video-frame-grid">
          {previewFrames.map((frame) => {
            const number = frame.output_frame_index + 1;
            return <button key={number} type="button" className="video-frame-card" onClick={() => openFrame(number)}>
              <img src={frame.classified_frame_url} alt={`Classified frame ${number}`} loading="lazy" />
              <span><strong>Frame {number}</strong><small>{Number(frame.timestamp_seconds).toFixed(2)} sec</small><small>Tracks {frame.tracks_detected} · Alpha {frame.alpha_count} · Electron {frame.electron_count}</small></span>
            </button>;
          })}
        </div>
      </section>

      <section className="card glass video-download-card">
        <div><span>Exports</span><h2>Video Analysis Downloads</h2><p>Saved outputs are served directly; downloading does not rerun analysis.</p></div>
        <div className="video-download-actions">
          <a className="btn btn-primary" href={result.downloads.video}><Download size={17} /> Download Classified Video</a>
          <a className="btn btn-outline" href={result.downloads.results}><FileJson size={17} /> Download Results JSON</a>
          {reportUrl
            ? <a className="btn btn-outline" href={reportUrl}><FileText size={17} /> Download Video Analysis Report</a>
            : <button className="btn btn-outline" type="button" onClick={generateReport} disabled={reportLoading}>{reportLoading ? <Loader2 className="video-spinner" size={17} /> : <FileText size={17} />} {reportLoading ? 'Generating report...' : 'Generate Video Analysis Report'}</button>}
        </div>
        {reportError && <p className="video-report-error">{reportError}</p>}
      </section>

      {(detailLoading || detailError || frameDetail) && <FrameDetail detail={frameDetail} loading={detailLoading} error={detailError} onClose={() => { setFrameDetail(null); setDetailError(''); }} onNavigate={openFrame} />}
    </div>
  );
}

function Info({ label, value }) { return <div><span>{label}</span><strong>{value}</strong></div>; }

function FrameDetail({ detail, loading, error, onClose, onNavigate }) {
  return <div className="video-modal-backdrop" role="presentation" onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}>
    <section className="video-frame-modal" role="dialog" aria-modal="true" aria-label="Classified frame details">
      <button className="video-modal-close" type="button" onClick={onClose} aria-label="Close frame details"><X size={20} /></button>
      {loading && <div className="video-detail-loading"><Loader2 className="video-spinner" /> Loading frame details…</div>}
      {error && <div className="video-error"><AlertCircle size={20} /><span>{error}</span></div>}
      {!loading && detail && <>
        <div className="video-detail-heading"><span>Classified frame</span><h2>Frame {detail.frame_index}</h2><p>{Number(detail.timestamp_seconds).toFixed(2)} sec</p></div>
        <img className="video-detail-image" src={detail.classified_frame_url} alt={`Classified frame ${detail.frame_index}`} />
        <div className="video-detail-stats"><Info label="Tracks Detected" value={detail.tracks_detected} /><Info label="Alpha" value={detail.alpha} /><Info label="Electron" value={detail.electron} /></div>
        {detail.tracks.length > 0 && <div className="video-track-table-wrap"><table><thead><tr><th>Track</th><th>Classification</th><th>Length</th><th>Width</th><th>Curvature</th><th>Tortuosity</th></tr></thead><tbody>
          {detail.tracks.map((track) => { const classification = track.classification || {}; const features = classification.features || {}; return <tr key={track.track_id}><td>{track.track_id}</td><td>{classification.predicted_class || 'Failed'}</td><td>{metric(features.track_length_px ?? features.length)}</td><td>{metric(features.mean_width_px ?? features.width)}</td><td>{metric(features.mean_curvature ?? features.curvature)}</td><td>{metric(features.tortuosity)}</td></tr>; })}
        </tbody></table></div>}
        <div className="video-detail-navigation">
          <button className="btn btn-outline" type="button" disabled={!detail.previous_frame} onClick={() => onNavigate(detail.previous_frame)}><ChevronLeft size={17} /> Previous Frame</button>
          <button className="btn btn-outline" type="button" disabled={!detail.next_frame} onClick={() => onNavigate(detail.next_frame)}>Next Frame <ChevronRight size={17} /></button>
        </div>
      </>}
    </section>
  </div>;
}

const metric = (value) => Number.isFinite(Number(value)) ? Number(value).toFixed(3) : '—';

export default VideoProcessing;
