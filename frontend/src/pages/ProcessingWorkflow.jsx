import { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  ArrowDown,
  Check,
  CheckCircle,
  Circle,
  Download,
  FileImage,
  Loader2,
  Maximize2,
  Play,
  RefreshCw,
  Ruler,
  Upload,
  X,
} from 'lucide-react';
import { apiService } from '../services/api';
import { useProcessingJob } from '../hooks/useProcessingJob';
import ImageLightbox from '../components/common/ImageLightbox';
import { classificationView } from './module3Presentation';
import { classifiedAnnotation } from './module4ClassifiedAnnotation';
import { exportProcessingReport } from '../reports/processingReport';
import './ProcessingWorkflow.css';

const SUPPORTED_RAW_IMAGE = /\.(bmp|jpe?g|png|tiff?)$/i;
const EMPTY_FILES = [];
const SCALE_CONSISTENCY_TOLERANCE_PERCENT = 3;
const WORKFLOW_TABS = [
  ['preprocessing', 'Image Pre-processing'],
  ['segmentation', 'Segmentation'],
  ['classification', 'Classification'],
  ['report', 'Report & Analysis'],
];

const stepState = (step, stage, hasFiles, preprocessing, detection, failedStage) => {
  if (step === 'upload') return hasFiles ? 'completed' : 'waiting';
  if (step === 'preprocessing') {
    if (stage === 'preprocessing') return 'active';
    if (failedStage === 'preprocessing') return 'failed';
    return preprocessing ? 'completed' : 'waiting';
  }
  if (step === 'detection') {
    if (stage === 'detecting') return 'active';
    if (failedStage === 'detection') return 'failed';
    return detection ? 'completed' : 'waiting';
  }
  if (stage === 'classifying') return 'active';
  if (stage === 'completed') return 'completed';
  if (failedStage === 'classification') return 'failed';
  return 'waiting';
};

const formatDuration = (milliseconds) => {
  if (milliseconds === null) return '—';
  return milliseconds < 1000 ? `${milliseconds} ms` : `${(milliseconds / 1000).toFixed(1)} s`;
};

const imageResolution = (file) => new Promise((resolve, reject) => {
  const url = URL.createObjectURL(file);
  const image = new Image();
  image.onload = () => {
    URL.revokeObjectURL(url);
    resolve({ width: image.naturalWidth, height: image.naturalHeight });
  };
  image.onerror = () => {
    URL.revokeObjectURL(url);
    reject(new Error(`Could not read the resolution of ${file.name}.`));
  };
  image.src = url;
});

const calibrationSnapshot = async (files, configuration) => {
  if (configuration?.status !== 'calibrated') {
    return Object.fromEntries(files.map((file) => [file.name, { status: 'uncalibrated' }]));
  }
  const expected = configuration.data?.source_image_resolution;
  if (expected) {
    const resolutions = await Promise.all(files.map(imageResolution));
    const mismatch = resolutions.findIndex(({ width, height }) => (
      width !== Number(expected.width) || height !== Number(expected.height)
    ));
    if (mismatch >= 0) {
      throw new Error(
        `${files[mismatch].name} is ${resolutions[mismatch].width} × ${resolutions[mismatch].height} px, `
        + `but calibration “${configuration.data.profile_name}” is for ${expected.width} × ${expected.height} px images.`,
      );
    }
  }
  return Object.fromEntries(files.map((file) => [file.name, {
    status: 'calibrated',
    data: { ...configuration.data },
  }]));
};

function ProcessingWorkflow() {
  const [localFiles, setSelectedFiles] = useState([]);
  const [dragging, setDragging] = useState(false);
  const [localStage, setStage] = useState('idle');
  const [error, setError] = useState('');
  const [localCalibration, setCalibrationConfiguration] = useState({ status: 'uncalibrated' });
  const [calibrationDialog, setCalibrationDialog] = useState('');
  const [savedCalibrationProfiles, setSavedCalibrationProfiles] = useState([]);
  const [profileLoadError, setProfileLoadError] = useState('');
  const [processingStartedAt, setProcessingStartedAt] = useState(null);
  const [activeTab, setActiveTab] = useState('preprocessing');
  const [selectedImageName, setSelectedImageName] = useState('');
  const [imageViewer, setImageViewer] = useState(null);
  const inputRef = useRef(null);
  const job = useProcessingJob('workflow');
  const [submitting, setSubmitting] = useState(false);
  const submissionRef = useRef(false);

  const selectedFiles = job.status?.files || (job.error.includes('could not be found') ? EMPTY_FILES : localFiles);
  const preprocessing = job.result?.preprocessing || null;
  const detection = job.result?.detection || null;
  const classifications = job.result?.classifications || {};
  const classificationSummary = job.result?.classificationSummary || null;
  const calibrations = job.status?.calibrations || {};
  const calibrationConfiguration = job.status
    ? Object.values(calibrations).find((item) => item.status === 'calibrated') || { status: 'uncalibrated' }
    : localCalibration;
  const handoffErrors = job.result?.handoffErrors || [];
  const durationMs = job.result?.durationMs ?? null;
  const warning = job.result?.warning || '';
  const failedStage = job.status?.status === 'FAILED'
    ? ({ DETECTING: 'detection', CLASSIFYING: 'classification' }[job.status.stage] || 'preprocessing') : '';
  const stage = failedStage ? 'failed' : job.result ? 'completed'
    : job.busy ? ({ DETECTING: 'detecting', CLASSIFYING: 'classifying' }[job.status?.stage] || 'preprocessing')
      : job.jobId ? 'preprocessing' : localStage;
  const isRunning = submitting || job.busy;
  const previews = useMemo(
    () => Object.fromEntries(selectedFiles.map((file) => [file.name, file.original_image_url || URL.createObjectURL(file)])),
    [selectedFiles],
  );

  useEffect(
    () => () => Object.values(previews).filter((url) => url.startsWith('blob:')).forEach((url) => URL.revokeObjectURL(url)),
    [previews],
  );

  useEffect(() => {
    let active = true;
    apiService.getCalibrationProfiles()
      .then((profiles) => { if (active) setSavedCalibrationProfiles(profiles); })
      .catch((loadError) => { if (active) setProfileLoadError(loadError.message); });
    return () => { active = false; };
  }, []);

  const clearResults = () => {
    job.clear();
    setProcessingStartedAt(null);
  };

  const addFiles = (incoming) => {
    if (isRunning) return;
    const candidates = Array.from(incoming || []);
    const supported = candidates.filter((file) => SUPPORTED_RAW_IMAGE.test(file.name));
    const rejected = candidates.filter((file) => !SUPPORTED_RAW_IMAGE.test(file.name));
    const hadJob = !!job.jobId;
    setSelectedFiles((existing) => {
      const current = hadJob ? [] : existing;
      const names = new Set(current.map((file) => file.name));
      return [...current, ...supported.filter((file) => !names.has(file.name))];
    });
    if (!selectedImageName && supported.length) setSelectedImageName(supported[0].name);
    clearResults();
    setStage('idle');
    setError(rejected.length
      ? `Unsupported file${rejected.length === 1 ? '' : 's'}: ${rejected.map((file) => file.name).join(', ')}. Use JPG, JPEG, PNG, BMP, TIF, or TIFF.`
      : supported.length ? '' : 'Select at least one supported image.');
    if (inputRef.current) inputRef.current.value = '';
  };

  const reset = () => {
    setSelectedFiles([]);
    setCalibrationConfiguration({ status: 'uncalibrated' });
    setSelectedImageName('');
    setImageViewer(null);
    setActiveTab('preprocessing');
    clearResults();
    setStage('idle');
    setError('');
    if (inputRef.current) inputRef.current.value = '';
  };

  const runFullPipeline = async () => {
    if (!selectedFiles.length || isRunning || submissionRef.current) return;
    submissionRef.current = true;
    setSubmitting(true);
    setError('');
    try {
      // Re-uploading a saved source happens only on an explicit user-requested retry.
      const files = await Promise.all(selectedFiles.map((file) => file instanceof File
        ? file : apiService.imageUrlToFile(file.original_image_url, file.name)));
      const runCalibrations = await calibrationSnapshot(files, calibrationConfiguration);
      setSelectedFiles(files);
      clearResults();
      setProcessingStartedAt(Date.now());
      setActiveTab('preprocessing');
      setStage('preprocessing');
      await job.start(files, runCalibrations);
    } catch (requestError) {
      setError(requestError.message);
      setStage('failed');
    } finally {
      submissionRef.current = false;
      setSubmitting(false);
    }
  };

  const changeCalibration = (configuration) => {
    if (isRunning) return;
    setSelectedFiles(selectedFiles);
    setCalibrationConfiguration(configuration);
    clearResults();
    setStage('idle');
    setError('');
  };

  const calibrationSaved = (profile) => {
    setSavedCalibrationProfiles((current) => [
      profile,
      ...current.filter((item) => item.profile_id !== profile.profile_id),
    ]);
    changeCalibration({ status: 'calibrated', data: profile });
    setCalibrationDialog('');
  };

  const successfulDetections = detection?.results.filter((result) => result.status === 'success') || [];
  const acceptedTracks = successfulDetections.reduce((sum, result) => sum + Number(result.accepted_track_count || 0), 0);
  const module2Failed = (detection?.failed_count || 0) + handoffErrors.length;

  return (
    <div className="workflow-page">
      <div className="workflow-heading">
        <div>
          <h1 className="page-title">Image Processing</h1>
          <p className="page-description">Upload cloud chamber images and run the complete particle analysis pipeline.</p>
        </div>
        {(job.jobId || preprocessing || detection) && (
          <button className="btn btn-outline" onClick={reset} disabled={isRunning}>
            <RefreshCw size={16} /> Start new processing run
          </button>
        )}
      </div>

      <PipelineStepper
        stateFor={(step) => stepState(step, stage, selectedFiles.length > 0, preprocessing, detection, failedStage)}
      />

      {(error || job.error) && <div className="workflow-alert error"><AlertCircle size={20} /><span><strong>{failedStage ? `${failedStage[0].toUpperCase()}${failedStage.slice(1)} failed. ` : ''}</strong>{error || job.error}</span></div>}
      {job.busy && <div className="workflow-alert" role="status">
        {job.status ? `${job.status.progress_percent}% · ${job.status.stage} · Image ${job.status.current_frame} / ${job.status.total_frames}`
          : job.jobId ? 'Reconnecting to saved image job…' : 'Uploading images…'}
      </div>}
      {warning && <div className="workflow-alert warning"><AlertCircle size={20} /><span>{warning}</span></div>}

      <section className="card workflow-section">
        <SectionHeader number="1" title="Upload Raw Images" status={selectedFiles.length ? `${selectedFiles.length} selected` : 'Waiting'} />
        <div
          className={`workflow-dropzone ${dragging ? 'is-dragging' : ''} ${isRunning ? 'is-disabled' : ''}`}
          onDragOver={(event) => { event.preventDefault(); if (!isRunning) setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            addFiles(event.dataTransfer.files);
          }}
          onClick={() => !isRunning && inputRef.current?.click()}
        >
          <div className="workflow-upload-icon"><Upload size={28} /></div>
          <div><h3>Upload raw chamber images</h3><p>Drag and drop images here, or browse from your computer.</p></div>
          <button className="btn btn-outline" type="button" disabled={isRunning} onClick={(event) => {
            event.stopPropagation();
            inputRef.current?.click();
          }}>Select files</button>
        </div>
        <input ref={inputRef} type="file" accept=".jpg,.jpeg,.png,.bmp,.tif,.tiff" multiple hidden onChange={(event) => addFiles(event.target.files)} />

        {!!selectedFiles.length && (
          <div className="workflow-selected">
            {selectedFiles.map((file) => (
              <article key={file.name} className="workflow-file">
                <img src={previews[file.name]} alt="Raw image preview" />
                <div><strong title={file.name}>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(2)} MB</span></div>
                <button type="button" aria-label={`Remove ${file.name}`} disabled={isRunning} onClick={() => {
                  const remaining = selectedFiles.filter((item) => item.name !== file.name);
                  setSelectedFiles(remaining);
                  if (selectedImageName === file.name) setSelectedImageName(remaining[0]?.name || '');
                  clearResults();
                  setStage('idle');
                }}><X size={17} /></button>
              </article>
            ))}
          </div>
        )}

        {!!selectedFiles.length && (
          <section className={`workflow-calibration-setup ${calibrationConfiguration.status}`}>
            <div className="workflow-calibration-setup-icon"><Ruler size={22} /></div>
            <div className="workflow-calibration-setup-copy">
              <span>Optional measurement configuration</span>
              <h3>Spatial Calibration</h3>
              {calibrationConfiguration.status === 'calibrated' ? <>
                <strong><CheckCircle size={17} /> {calibrationConfiguration.data.profile_name}</strong>
                <p>{Number(calibrationConfiguration.data.chamber_width_mm).toFixed(2)} × {Number(calibrationConfiguration.data.chamber_height_mm).toFixed(2)} mm · 4-point measurement calibration</p>
              </> : <>
                <strong>Status: Not calibrated</strong>
                <p>Measurements will remain in pixels.</p>
              </>}
            </div>
            <div className="workflow-calibration-setup-actions">
              {calibrationConfiguration.status === 'calibrated' ? <>
                <button className="btn btn-outline" type="button" disabled={isRunning} onClick={() => setCalibrationDialog('calibrate')}>Change Calibration</button>
                <button className="btn btn-quiet" type="button" disabled={isRunning} onClick={() => changeCalibration({ status: 'uncalibrated' })}>No Calibration</button>
              </> : <>
                <button className="btn btn-outline" type="button" disabled={isRunning} onClick={() => setCalibrationDialog('calibrate')}>Calibrate</button>
                <button className="btn btn-outline" type="button" disabled={isRunning || !savedCalibrationProfiles.length} onClick={() => setCalibrationDialog('saved')}>Use Saved Calibration</button>
              </>}
            </div>
          </section>
        )}
        {profileLoadError && <div className="workflow-calibration-profile-warning">Saved profiles could not be loaded: {profileLoadError}</div>}

        <div className="workflow-primary-action">
          <button className="btn btn-primary" onClick={runFullPipeline} disabled={!selectedFiles.length || isRunning || !!job.jobId}>
            {isRunning ? <><Loader2 className="spin" size={17} /> Processing images…</> : <><Play size={17} /> Process Images</>}
          </button>
          {stage === 'failed' && <button className="btn btn-outline" onClick={runFullPipeline}><RefreshCw size={16} /> Retry analysis</button>}
        </div>
      </section>

      {calibrationDialog === 'calibrate' && <CalibrationDialog
        files={selectedFiles}
        previews={previews}
        initialProfile={calibrationConfiguration.status === 'calibrated' ? calibrationConfiguration.data : null}
        onCancel={() => setCalibrationDialog('')}
        onSaved={calibrationSaved}
      />}
      {calibrationDialog === 'saved' && <SavedCalibrationDialog
        profiles={savedCalibrationProfiles}
        onCancel={() => setCalibrationDialog('')}
        onSelect={(profile) => {
          changeCalibration({ status: 'calibrated', data: profile });
          setCalibrationDialog('');
        }}
      />}

      {(isRunning || preprocessing || detection) && (
        <WorkflowWorkspace
          activeTab={activeTab}
          setActiveTab={setActiveTab}
          selectedImageName={selectedFiles.some((file) => file.name === selectedImageName) ? selectedImageName : selectedFiles[0]?.name || ''}
          setSelectedImageName={setSelectedImageName}
          selectedFiles={selectedFiles}
          stage={stage}
          failedStage={failedStage}
          preprocessing={preprocessing}
          detection={detection}
          classifications={classifications}
          classificationSummary={classificationSummary}
          calibrations={calibrations}
          handoffErrors={handoffErrors}
          previews={previews}
          durationMs={durationMs}
          acceptedTracks={acceptedTracks}
          module2Failed={module2Failed}
          successfulDetections={successfulDetections}
          processingStartedAt={processingStartedAt}
          onOpenImage={(label, url, view = {}) => setImageViewer({ label, url, ...view })}
        />
      )}
      {imageViewer && (
        <ImageLightbox
          label={imageViewer.label}
          url={imageViewer.url}
          onClose={() => setImageViewer(null)}
        >
          {imageViewer.kind === 'classified' ? (
            <ClassifiedImage
              result={imageViewer.result}
              sourceUrl={imageViewer.url}
              classifications={imageViewer.classifications}
            />
          ) : null}
        </ImageLightbox>
      )}
    </div>
  );
}

function CalibrationDialog({ files, previews, initialProfile, onCancel, onSaved }) {
  const [sourceName, setSourceName] = useState(files[0]?.name || '');
  const [profileName, setProfileName] = useState(initialProfile?.profile_name || 'Cloud Chamber Setup A');
  const [chamberWidth, setChamberWidth] = useState(initialProfile?.chamber_width_mm || '');
  const [chamberHeight, setChamberHeight] = useState(initialProfile?.chamber_height_mm || '');
  const [points, setPoints] = useState(() => {
    const corners = initialProfile?.selected_corners;
    if (!corners) return [];
    const ordered = ['A', 'B', 'C', 'D'].map((label) => corners[label]);
    return ordered.every((point) => Number.isFinite(Number(point?.x)) && Number.isFinite(Number(point?.y)))
      ? ordered.map((point) => ({ x: Number(point.x), y: Number(point.y) }))
      : [];
  });
  const [dimensions, setDimensions] = useState({ width: 1, height: 1 });
  const [saving, setSaving] = useState(false);
  const [dialogError, setDialogError] = useState('');
  const imageRef = useRef(null);
  const sourceFile = files.find((file) => file.name === sourceName) || files[0];
  const requiredPoints = 4;
  const pointLabels = ['A', 'B', 'C', 'D'];
  const pointGuidance = ['top-left', 'top-right', 'bottom-right', 'bottom-left'];
  const chamberMetrics = useMemo(() => {
    if (points.length !== 4) return null;
    const distancePx = (first, second) => Math.hypot(second.x - first.x, second.y - first.y);
    const widthPx = (distancePx(points[0], points[1]) + distancePx(points[3], points[2])) / 2;
    const heightPx = (distancePx(points[0], points[3]) + distancePx(points[1], points[2])) / 2;
    const horizontal = Number(chamberWidth) / widthPx;
    const vertical = Number(chamberHeight) / heightPx;
    const mean = (horizontal + vertical) / 2;
    const difference = Math.abs(horizontal - vertical) / mean * 100;
    return { widthPx, heightPx, horizontal, vertical, difference };
  }, [points, chamberWidth, chamberHeight]);

  const selectPoint = (event) => {
    if (points.length >= requiredPoints || !imageRef.current) return;
    const rect = imageRef.current.getBoundingClientRect();
    const x = (event.clientX - rect.left) * (imageRef.current.naturalWidth / rect.width);
    const y = (event.clientY - rect.top) * (imageRef.current.naturalHeight / rect.height);
    if (points.some((point) => Math.hypot(point.x - x, point.y - y) < 3)) {
      setDialogError('Choose a distinct point; calibration points cannot overlap.');
      return;
    }
    setPoints((current) => [...current, { x, y }]);
    setDialogError('');
  };

  const save = async () => {
    if (!profileName.trim()) return setDialogError('Enter a profile name for this camera/chamber setup.');
    if (points.length !== requiredPoints) return setDialogError(`Select all ${requiredPoints} calibration points on the image.`);
    if (!Number.isFinite(Number(chamberWidth)) || Number(chamberWidth) <= 0 || !Number.isFinite(Number(chamberHeight)) || Number(chamberHeight) <= 0) return setDialogError('Enter positive chamber width and height values in millimetres.');
    if (dimensions.width <= 1 || dimensions.height <= 1) return setDialogError('Wait for the calibration image to finish loading.');
    setSaving(true);
    setDialogError('');
    try {
      const profile = await apiService.calculateCalibration({
        point_1_x: points[0].x,
        point_1_y: points[0].y,
        point_2_x: points[1].x,
        point_2_y: points[1].y,
        point_3_x: points[2].x,
        point_3_y: points[2].y,
        point_4_x: points[3].x,
        point_4_y: points[3].y,
        chamber_width_mm: Number(chamberWidth),
        chamber_height_mm: Number(chamberHeight),
        image_filename: sourceFile.name,
        profile_name: profileName.trim(),
        source_image_width: dimensions.width,
        source_image_height: dimensions.height,
      });
      onSaved(profile);
    } catch (saveError) {
      setDialogError(saveError.message);
    } finally {
      setSaving(false);
    }
  };

  return <div className="workflow-dialog-backdrop" role="presentation" onMouseDown={onCancel}>
    <section className="card workflow-dialog calibration-dialog" role="dialog" aria-modal="true" aria-labelledby="calibration-dialog-title" onMouseDown={(event) => event.stopPropagation()}>
      <header><div><span>Measurement configuration</span><h2 id="calibration-dialog-title">Spatial Calibration</h2><p>Select the four inner chamber corners. The resulting perspective mapping is used only for physical measurements; processing pixels are never warped.</p></div><button type="button" aria-label="Close calibration" onClick={onCancel}><X size={20} /></button></header>
      {dialogError && <div className="workflow-inline-error">{dialogError}</div>}
      <div className="calibration-dialog-grid">
        <div className="calibration-dialog-viewer">
          {files.length > 1 && <label>Reference image<select value={sourceName} onChange={(event) => { setSourceName(event.target.value); setPoints([]); setDimensions({ width: 1, height: 1 }); }} disabled={saving}>{files.map((file) => <option key={file.name} value={file.name}>{file.name}</option>)}</select></label>}
          <div className="calibration-image-stage">
            <div>
              <img ref={imageRef} src={previews[sourceFile?.name]} alt="Calibration reference" draggable={false} onClick={selectPoint} onLoad={(event) => setDimensions({ width: event.currentTarget.naturalWidth, height: event.currentTarget.naturalHeight })} />
              <svg viewBox={`0 0 ${dimensions.width} ${dimensions.height}`} preserveAspectRatio="none" aria-hidden="true">
                {points.length > 1 && <polyline points={[...points, ...(points.length === 4 ? [points[0]] : [])].map((point) => `${point.x},${point.y}`).join(' ')} />}
                {points.map((point, index) => <g key={`${point.x}-${point.y}`}><circle cx={point.x} cy={point.y} r="6" /><text x={point.x + 12} y={point.y - 12}>{pointLabels[index]}</text></g>)}
              </svg>
            </div>
          </div>
          <p>{points.length < requiredPoints ? `Point ${pointLabels[points.length]}: select the ${pointGuidance[points.length]} inner chamber corner.` : 'All four corners selected. Review the chamber dimensions and save.'}</p>
        </div>
        <div className="calibration-dialog-fields">
          <label>Profile name<input value={profileName} onChange={(event) => setProfileName(event.target.value)} placeholder="e.g. Cloud Chamber Setup A" disabled={saving} /></label>
          <div className="calibration-dimension-fields"><label>Chamber width (mm)<input type="number" min="0" step="any" value={chamberWidth} onChange={(event) => setChamberWidth(event.target.value)} placeholder="e.g. 300" disabled={saving} /></label><label>Chamber height (mm)<input type="number" min="0" step="any" value={chamberHeight} onChange={(event) => setChamberHeight(event.target.value)} placeholder="e.g. 500" disabled={saving} /></label></div>
          <div className="calibration-point-readout">{pointLabels.map((label, index) => <div key={label}><span>{label} · {pointGuidance[index]}</span><strong>{points[index] ? `${points[index].x.toFixed(1)}, ${points[index].y.toFixed(1)} px` : 'Not selected'}</strong></div>)}</div>
          <div className="calibration-resolution"><span>Source resolution</span><strong>{dimensions.width} × {dimensions.height} px</strong><small>The saved profile can be reused with images at this resolution from the same fixed setup.</small></div>
          {chamberMetrics && Number.isFinite(chamberMetrics.horizontal) && Number.isFinite(chamberMetrics.vertical) && <div className={`calibration-scale-feedback ${chamberMetrics.difference <= SCALE_CONSISTENCY_TOLERANCE_PERCENT ? 'good' : 'warning'}`}><div><span>Selected chamber area</span><strong>{chamberMetrics.widthPx.toFixed(1)} × {chamberMetrics.heightPx.toFixed(1)} px</strong></div><div><span>Average horizontal scale</span><strong>{chamberMetrics.horizontal.toFixed(4)} mm/px</strong></div><div><span>Average vertical scale</span><strong>{chamberMetrics.vertical.toFixed(4)} mm/px</strong></div><small>These averages are a summary only. Track measurements use the full four-point perspective mapping at each path location.</small></div>}
          <button className="btn btn-outline" type="button" disabled={!points.length || saving} onClick={() => setPoints([])}><RefreshCw size={15} /> Reset points</button>
        </div>
      </div>
      <footer><button className="btn btn-quiet" type="button" onClick={onCancel} disabled={saving}>Cancel</button><button className="btn btn-primary" type="button" onClick={save} disabled={saving || points.length !== requiredPoints}>{saving ? <><Loader2 className="spin" size={16} /> Saving…</> : <><Check size={16} /> Save & Use Calibration</>}</button></footer>
    </section>
  </div>;
}

function SavedCalibrationDialog({ profiles, onCancel, onSelect }) {
  return <div className="workflow-dialog-backdrop" role="presentation" onMouseDown={onCancel}>
    <section className="card workflow-dialog saved-calibration-dialog" role="dialog" aria-modal="true" aria-labelledby="saved-calibration-title" onMouseDown={(event) => event.stopPropagation()}>
      <header><div><span>Measurement configuration</span><h2 id="saved-calibration-title">Use Saved Calibration</h2><p>Choose a profile for images captured with the same fixed camera/chamber setup and resolution.</p></div><button type="button" aria-label="Close saved calibrations" onClick={onCancel}><X size={20} /></button></header>
      <div className="saved-calibration-list">{profiles.map((profile) => <button type="button" key={profile.profile_id} onClick={() => onSelect(profile)}>
        <div><strong>{profile.profile_name}</strong><span>{Number(profile.mm_per_pixel).toFixed(4)} mm/px</span></div>
        <dl><div><dt>Calibration</dt><dd>4-point measurement mapping</dd></div><div><dt>Chamber size</dt><dd>{Number(profile.chamber_width_mm).toFixed(2)} × {Number(profile.chamber_height_mm).toFixed(2)} mm</dd></div><div><dt>Resolution</dt><dd>{profile.source_image_resolution ? `${profile.source_image_resolution.width} × ${profile.source_image_resolution.height} px` : 'Unknown'}</dd></div><div><dt>Image processing</dt><dd>Unchanged</dd></div></dl>
      </button>)}</div>
      <footer><button className="btn btn-quiet" type="button" onClick={onCancel}>Cancel</button></footer>
    </section>
  </div>;
}

function WorkflowWorkspace({
  activeTab,
  setActiveTab,
  selectedImageName,
  setSelectedImageName,
  selectedFiles,
  stage,
  failedStage,
  preprocessing,
  detection,
  classifications,
  classificationSummary,
  calibrations,
  handoffErrors,
  previews,
  durationMs,
  acceptedTracks,
  module2Failed,
  successfulDetections,
  processingStartedAt,
  onOpenImage,
}) {
  return <section className="workflow-workspace">
    {stage === 'completed' && <div className="workflow-completion-strip">
      <CheckCircle size={20} />
      <strong>Processing completed</strong>
      <span>{acceptedTracks} particle track{acceptedTracks === 1 ? '' : 's'} detected</span>
      <span>{classificationSummary?.electron_count ?? 0} Electron</span>
      <span>{classificationSummary?.alpha_count ?? 0} Alpha</span>
      <span>{formatDuration(durationMs)}</span>
    </div>}
    <div className="workflow-tabs" role="tablist" aria-label="Processing modules">
      {WORKFLOW_TABS.map(([key, label], index) => (
        <button
          key={key}
          type="button"
          role="tab"
          aria-selected={activeTab === key}
          className={activeTab === key ? 'active' : ''}
          onClick={() => setActiveTab(key)}
        >
          <span>{index + 1}</span>{label}
        </button>
      ))}
    </div>

    <div className="card workflow-module-panel">
      {selectedFiles.length > 1 && (
        <div className="workflow-image-selector">
          <label htmlFor="workflow-image-select">Viewing image</label>
          <select id="workflow-image-select" value={selectedImageName} onChange={(event) => setSelectedImageName(event.target.value)}>
            {selectedFiles.map((file) => <option key={file.name} value={file.name}>{file.name}</option>)}
          </select>
          <span>{selectedFiles.findIndex((file) => file.name === selectedImageName) + 1} of {selectedFiles.length}</span>
        </div>
      )}

      <div role="tabpanel">
        {activeTab === 'preprocessing' && <PreprocessingPanel stage={stage} failedStage={failedStage} preprocessing={preprocessing} selectedImageName={selectedImageName} preview={previews[selectedImageName]} calibration={calibrations[selectedImageName]} onOpenImage={onOpenImage} />}
        {activeTab === 'segmentation' && <DetectionSegmentationSection stage={stage} failedStage={failedStage} detection={detection} preprocessing={preprocessing} handoffErrors={handoffErrors} selectedImageName={selectedImageName} onOpenImage={onOpenImage} />}
        {activeTab === 'classification' && <GeometricClassificationSection stage={stage} failedStage={failedStage} detection={detection} classifications={classifications} selectedImageName={selectedImageName} onOpenImage={onOpenImage} />}
        {activeTab === 'report' && <ReportAnalysisPanel stage={stage} selectedFiles={selectedFiles} selectedImageName={selectedImageName} preprocessing={preprocessing} detection={detection} classifications={classifications} classificationSummary={classificationSummary} calibrations={calibrations} previews={previews} durationMs={durationMs} processingStartedAt={processingStartedAt} acceptedTracks={acceptedTracks} module2Failed={module2Failed} successfulDetections={successfulDetections} onOpenImage={onOpenImage} />}
      </div>
    </div>
  </section>;
}

function ModulePanelHeader({ title, subtitle, status, active = false, failed = false }) {
  return <header className="workflow-module-header">
    <div><h2>{title}</h2><p>{subtitle}</p></div>
    <div className={`workflow-status ${active ? 'active' : ''} ${failed ? 'failed' : ''}`}>
      {active && <Loader2 className="spin" size={14} />}
      {!active && status === 'Completed' && <CheckCircle size={14} />}
      {failed && <AlertCircle size={14} />}
      {status}
    </div>
  </header>;
}

function PreprocessingPanel({ stage, failedStage, preprocessing, selectedImageName, preview, calibration, onOpenImage }) {
  const active = stage === 'preprocessing';
  const failed = failedStage === 'preprocessing';
  const result = preprocessing?.results.find((item) => item.original_filename === selectedImageName);
  const status = active ? 'Processing' : failed ? 'Failed' : result?.status === 'success' ? 'Completed' : result ? 'Failed' : 'Waiting';
  return <section>
    <ModulePanelHeader title="Module 1 — Image Pre-processing and Calibration" subtitle="Prepare the raw chamber image and report the measurement configuration selected before this run." status={status} active={active} failed={failed} />
    {active && <ProgressMessage title="Isolating particle tracks" text="Module 1 is processing the selected raw images." />}
    {!result && !active && <EmptyStage text="Pre-processing results are not available yet." />}
    {result && result.status !== 'success' && <div className="workflow-inline-error">{result.error_message || 'Pre-processing failed for this image.'}</div>}
    {result?.status === 'success' && <div className="workflow-preprocess-focus">
      <ImagePanel label="Raw image" url={preview} onOpen={() => onOpenImage('Raw image', preview)} />
      <ArrowDown className="workflow-panel-arrow" size={24} />
      <ImagePanel label="Preprocessed particle image" url={result.processed_image_url} onOpen={() => onOpenImage('Preprocessed particle image', result.processed_image_url)} />
    </div>}
    {result?.status === 'success' && <CalibrationStatus calibration={calibration} />}
  </section>;
}

function CalibrationStatus({ calibration }) {
  const data = calibration?.data;
  return <section className={`workflow-calibration ${calibration?.status || 'uncalibrated'}`}>
    <div>
      <span>Spatial calibration</span>
      {calibration?.status === 'calibrated'
        ? <strong><CheckCircle size={16} /> {data.profile_name || 'Calibrated'}</strong>
        : calibration?.status === 'loading'
          ? <strong><Loader2 className="spin" size={16} /> Checking calibration…</strong>
          : <strong>Not calibrated — measurements remain in pixels</strong>}
    </div>
    {data && <dl>
      <div><dt>Status</dt><dd>4-point calibration</dd></div>
      <div><dt>Chamber size</dt><dd>{Number(data.chamber_width_mm).toFixed(2)} × {Number(data.chamber_height_mm).toFixed(2)} mm</dd></div>
      <div><dt>Horizontal scale</dt><dd>{Number(data.horizontal_mm_per_pixel).toFixed(4)} mm/px average</dd></div>
      <div><dt>Vertical scale</dt><dd>{Number(data.vertical_mm_per_pixel).toFixed(4)} mm/px average</dd></div>
      <div><dt>Processing image</dt><dd>Unchanged</dd></div>
    </dl>}
    {calibration?.status === 'error' && <p>{calibration.message}</p>}
  </section>;
}

function ReportAnalysisPanel({ stage, selectedFiles, selectedImageName, preprocessing, detection, classifications, classificationSummary, calibrations, previews, durationMs, processingStartedAt, acceptedTracks, module2Failed, successfulDetections, onOpenImage }) {
  const [annotationError, setAnnotationError] = useState('');
  const [annotationStatus, setAnnotationStatus] = useState('idle');
  const [pdfStatus, setPdfStatus] = useState('idle');
  const [pdfMessage, setPdfMessage] = useState('');
  const detectionResult = detection?.results.find((item) => item.source_original_filename === selectedImageName);
  const rows = detectionResult?.status === 'success' ? detectionResult.tracks.map((track) => ({
    track,
    view: classificationView(classifications[`${detectionResult.run_id}:${track.track_id}`] || { status: 'unavailable' }),
  })) : [];
  const sourceUrl = preprocessing?.results.find(
    (item) => item.status === 'success' && item.original_filename === selectedImageName,
  )?.processed_image_url;
  const classifiedCount = rows.filter(({ view }) => view.kind === 'success').length;
  const complete = stage === 'completed';
  const downloadClassifiedImage = async () => {
    if (annotationStatus === 'preparing') return;
    setAnnotationError('');
    setAnnotationStatus('preparing');
    try {
      await exportClassifiedImage(detectionResult, sourceUrl, classifications);
    } catch (downloadError) {
      setAnnotationError(`Could not create the classified image: ${downloadError.message}`);
    } finally {
      setAnnotationStatus('idle');
    }
  };
  const exportPdfReport = async () => {
    if (pdfStatus === 'generating') return;
    setPdfStatus('generating');
    setPdfMessage('');
    try {
      const exported = await exportProcessingReport({
        selectedFiles,
        selectedImageName,
        previews,
        preprocessing,
        detection,
        classifications,
        calibrations,
        durationMs,
        processingStartedAt,
        generatedAt: new Date(),
      });
      setPdfStatus('success');
      setPdfMessage(`PDF report generated and downloaded (${exported.pageCount} pages).`);
    } catch (pdfError) {
      setPdfStatus('error');
      setPdfMessage(`Could not generate the PDF report: ${pdfError.message}`);
    }
  };
  return <section>
    <ModulePanelHeader title="Module 4 — Particle Analytics and Result Management" subtitle="Review run totals, final annotated output, and the measured values already returned by the pipeline." status={complete ? 'Completed' : 'Waiting'} />
    {!preprocessing && <EmptyStage text="Run summary and final outputs will appear after processing begins." />}
    {preprocessing && <>
      <div className="workflow-summary-grid">
        <Metric label="Images selected" value={selectedFiles.length} />
        <Metric label="Preprocessed" value={preprocessing.successful_count} />
        <Metric label="Preprocessing failed" value={preprocessing.failed_count} tone={preprocessing.failed_count ? 'failure' : ''} />
        <Metric label="Detection successful" value={detection?.successful_count ?? '—'} />
        <Metric label="Detection / handoff failed" value={detection ? module2Failed : '—'} tone={module2Failed ? 'failure' : ''} />
        <Metric label="Tracks detected" value={detection ? acceptedTracks : '—'} />
        <Metric label="Alpha particles" value={classificationSummary?.alpha_count ?? '—'} tone="alpha" />
        <Metric label="Electrons" value={classificationSummary?.electron_count ?? '—'} tone="electron" />
        <Metric label="Classification failed" value={classificationSummary?.failed_tracks ?? '—'} tone={classificationSummary?.failed_tracks ? 'failure' : ''} />
        <Metric label="Processing duration" value={formatDuration(durationMs)} />
      </div>
      <CalibrationStatus calibration={calibrations[selectedImageName]} />

      {detectionResult?.status === 'success' && <div className="workflow-report-focus">
        <ClassifiedImage
          result={detectionResult}
          sourceUrl={sourceUrl}
          classifications={classifications}
          onOpen={() => onOpenImage('Final classified annotation', sourceUrl, {
            kind: 'classified',
            result: detectionResult,
            classifications,
          })}
        />
        <div>
          <h3 title={selectedImageName}>{selectedImageName}</h3>
          <p>{detectionResult.accepted_track_count} detected track{detectionResult.accepted_track_count === 1 ? '' : 's'} · Run <code title={detectionResult.run_id}>{detectionResult.run_id}</code></p>
          <div className="workflow-report-actions">
            <button className="btn btn-outline" type="button" disabled={!classifiedCount || annotationStatus === 'preparing'} onClick={downloadClassifiedImage}>
              {annotationStatus === 'preparing' ? <><Loader2 className="spin" size={16} /> Preparing image...</> : <><Download size={16} /> Download Classified Image</>}
            </button>
            <button className="btn btn-primary" type="button" disabled={pdfStatus === 'generating'} onClick={exportPdfReport}>
              {pdfStatus === 'generating' ? <><Loader2 className="spin" size={16} /> Generating PDF...</> : <><Download size={16} /> Export PDF Report</>}
            </button>
          </div>
          {!classifiedCount && <span className="workflow-report-note">The classified annotation becomes available when at least one track is classified.</span>}
          {pdfMessage && <span className={`workflow-report-note pdf-${pdfStatus}`} role={pdfStatus === 'error' ? 'alert' : 'status'}>{pdfMessage}</span>}
        </div>
      </div>}
      {annotationError && <div className="workflow-inline-error">{annotationError}</div>}
      {detectionResult && detectionResult.status !== 'success' && <div className="workflow-inline-error">{detectionResult.error_message || 'No report is available for this image.'}</div>}
      {!detectionResult && <EmptyStage text="Detection and classification results are not available for this image yet." />}

      {!!rows.length && <div className="workflow-report-table-wrap">
        <table className="workflow-report-table">
          <thead><tr><th>Track</th><th>Length</th><th>Mean width</th><th>Mean curvature</th><th>Tortuosity</th><th>Particle</th><th>Confidence</th></tr></thead>
          <tbody>{rows.map(({ track, view }) => {
            const feature = (key) => {
              const row = view.kind === 'success' && view.features.find((item) => item.key === key);
              return row ? `${row.value}${row.physicalValue ? ` · ${row.physicalValue}` : ''}` : '—';
            };
            return <tr key={track.track_id}><td>Track {track.label_id}</td><td>{feature('track_length_px')}</td><td>{feature('mean_width_px')}</td><td>{feature('mean_curvature')}</td><td>{feature('tortuosity')}</td><td>{view.kind === 'success' ? view.title : '—'}</td><td>{view.kind === 'success' ? view.confidence : '—'}</td></tr>;
          })}</tbody>
        </table>
      </div>}
    </>}
    {!!successfulDetections.length && <div className="workflow-run-ids"><span>Backend run IDs</span><div>{successfulDetections.map((result) => <code key={result.run_id} title={result.run_id}>{result.run_id}</code>)}</div></div>}
  </section>;
}

function classificationForTrack(result, track, classifications) {
  return classificationView(
    classifications[`${result.run_id}:${track.track_id}`]
      || { status: 'unavailable' },
  );
}

function annotationForTrack(result, track, classifications) {
  return classifiedAnnotation(
    track,
    classificationForTrack(result, track, classifications),
  );
}

async function loadAnnotationImage(sourceUrl) {
  const response = await fetch(sourceUrl);
  if (!response.ok) throw new Error(`The source image could not be downloaded (${response.status}).`);
  const objectUrl = URL.createObjectURL(await response.blob());
  try {
    return await new Promise((resolve, reject) => {
      const image = new Image();
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error('The source image could not be loaded.'));
      image.src = objectUrl;
    });
  } finally {
    URL.revokeObjectURL(objectUrl);
  }
}

function canvasBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error('The browser could not encode the classified PNG.'));
    }, 'image/png');
  });
}

function drawClassifiedAnnotations(context, result, classifications, imageWidth) {
  const fontSize = Math.max(18, imageWidth * 0.014);
  const plaqueHeight = fontSize * 1.55;
  const strokeWidth = Math.max(2, imageWidth * 0.0015);
  context.font = `700 ${fontSize}px Inter, system-ui, sans-serif`;
  context.textBaseline = 'alphabetic';
  context.lineWidth = strokeWidth;

  result.tracks.forEach((track) => {
    const annotation = annotationForTrack(result, track, classifications);
    if (!annotation) return;
    const x = Number(track.bbox_x);
    const y = Number(track.bbox_y);
    const width = Number(track.bbox_width);
    const height = Number(track.bbox_height);
    if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0) return;

    const plaqueWidth = Math.min(
      context.canvas.width,
      Math.max(width, context.measureText(annotation.text).width + fontSize),
    );
    const plaqueX = Math.max(0, Math.min(x, context.canvas.width - plaqueWidth));
    const plaqueY = y >= plaqueHeight + 3 ? y - plaqueHeight - 3 : y;

    context.strokeStyle = annotation.colour;
    context.strokeRect(x, y, width, height);
    context.fillStyle = annotation.colour;
    context.fillRect(plaqueX, plaqueY, plaqueWidth, plaqueHeight);
    context.fillStyle = '#07090c';
    context.fillText(annotation.text, plaqueX + fontSize * 0.5, plaqueY + fontSize * 1.08);
  });
}

async function exportClassifiedImage(result, sourceUrl, classifications) {
  if (!result || !sourceUrl) throw new Error('The classified source image is unavailable.');
  const image = await loadAnnotationImage(sourceUrl);
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas export is unavailable.');

  context.drawImage(image, 0, 0);
  drawClassifiedAnnotations(context, result, classifications, canvas.width);
  const blob = await canvasBlob(canvas);
  const objectUrl = URL.createObjectURL(blob);
  const sourceStem = String(result.source_original_filename || result.original_filename || 'classified_tracks')
    .replace(/\.[^.]+$/, '')
    .replace(/[^A-Za-z0-9._-]+/g, '_');
  const anchor = document.createElement('a');
  anchor.href = objectUrl;
  anchor.download = `${sourceStem}_classified.png`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
}

function ClassifiedImage({ result, sourceUrl, classifications, onOpen = null }) {
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const classifiedCanvas = <span className="workflow-classified-canvas">
    <img
      src={sourceUrl}
      alt={`Classified particle tracks for ${result.source_original_filename || result.original_filename}`}
      onLoad={(event) => setImageSize({
        width: event.currentTarget.naturalWidth,
        height: event.currentTarget.naturalHeight,
      })}
    />
    {imageSize.width > 0 && imageSize.height > 0 && <svg className="workflow-classified-overlay" viewBox={`0 0 ${imageSize.width} ${imageSize.height}`} aria-hidden="true">
      {result.tracks.map((track) => {
        const annotation = annotationForTrack(result, track, classifications);
        if (!annotation) return null;
        const x = Number(track.bbox_x);
        const y = Number(track.bbox_y);
        const width = Number(track.bbox_width);
        const height = Number(track.bbox_height);
        if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0) return null;
        const fontSize = Math.max(18, imageSize.width * 0.014);
        const plaqueHeight = fontSize * 1.55;
        const plaqueWidth = Math.min(imageSize.width, Math.max(width, annotation.text.length * fontSize * 0.59 + fontSize));
        const plaqueX = Math.max(0, Math.min(x, imageSize.width - plaqueWidth));
        const plaqueY = y >= plaqueHeight + 3 ? y - plaqueHeight - 3 : y;
        return <g key={track.track_id}>
          <rect x={x} y={y} width={width} height={height} fill="none" stroke={annotation.colour} strokeWidth={Math.max(2, imageSize.width * 0.0015)} vectorEffect="non-scaling-stroke" />
          <rect x={plaqueX} y={plaqueY} width={plaqueWidth} height={plaqueHeight} rx={fontSize * 0.18} fill={annotation.colour} />
          <text x={plaqueX + fontSize * 0.5} y={plaqueY + fontSize * 1.08} fill="#07090c" fontSize={fontSize} fontFamily="Inter, system-ui, sans-serif" fontWeight="700">{annotation.text}</text>
        </g>;
      })}
    </svg>}
  </span>;
  return <figure className="workflow-image-panel workflow-classified-image">
    <div>
      {onOpen && sourceUrl
        ? <button type="button" className="workflow-image-button workflow-classified-image-button" aria-label="View final classified annotation full picture" onClick={onOpen}>
          {classifiedCanvas}
          <span className="workflow-expand-indicator"><Maximize2 size={15} /> View full picture</span>
        </button>
        : classifiedCanvas}
    </div>
    <figcaption>Final classified annotation</figcaption>
  </figure>;
}

function PipelineStepper({ stateFor }) {
  const steps = [['upload', 'Upload'], ['preprocessing', 'Preprocessing'], ['detection', 'Detection & Analysis'], ['results', 'Results']];
  return <div className="workflow-stepper">{steps.map(([key, label], index) => {
    const state = stateFor(key);
    return <div className="workflow-step-wrap" key={key}><div className={`workflow-step ${state}`}><span>{state === 'completed' ? <Check size={17} /> : state === 'active' ? <Loader2 className="spin" size={17} /> : state === 'failed' ? <AlertCircle size={17} /> : <Circle size={13} />}</span><strong>{label}</strong></div>{index < steps.length - 1 && <div className={`workflow-step-line ${state === 'completed' ? 'completed' : ''}`} />}</div>;
  })}</div>;
}

function SectionHeader({ number, title, status, active = false }) {
  return <header className="workflow-section-header"><div><span>{number}</span><h2>{title}</h2></div><div className={`workflow-status ${active ? 'active' : ''}`}>{active && <Loader2 className="spin" size={14} />}{status}</div></header>;
}

function ProgressMessage({ title, text }) {
  return <div className="workflow-progress"><Loader2 className="spin" size={36} /><div><h3>{title}</h3><p>{text}</p></div></div>;
}

function EmptyStage({ text }) {
  return <div className="workflow-empty"><Circle size={17} /> {text}</div>;
}

function ImagePanel({ label, url, onOpen = null }) {
  return <figure className="workflow-image-panel">
    <div>{url
      ? onOpen
        ? <button type="button" className="workflow-image-button" aria-label={`View ${label} full picture`} onClick={onOpen}>
          <img src={url} alt={label} />
          <span className="workflow-expand-indicator"><Maximize2 size={15} /> View full picture</span>
        </button>
        : <img src={url} alt={label} />
      : <FileImage size={28} />}</div>
    <figcaption>{label}</figcaption>
  </figure>;
}

function Metric({ label, value, tone = '' }) {
  return <div className={`workflow-metric ${tone}`}><span>{label}</span><strong>{value}</strong></div>;
}

function SubstageHeader({ code, title, subtitle, status, active = false, failed = false }) {
  return <header className="workflow-substage-header"><div><span>{code}</span><div><h3>{title}</h3><p>{subtitle}</p></div></div><div className={`workflow-status ${active ? 'active' : ''} ${failed ? 'failed' : ''}`}>{active && <Loader2 className="spin" size={14} />}{!active && status === 'Completed' && <CheckCircle size={14} />}{failed && <AlertCircle size={14} />}{status}</div></header>;
}

function DetectionSegmentationSection({ stage, failedStage, detection, preprocessing, handoffErrors, selectedImageName, onOpenImage }) {
  const active = stage === 'detecting';
  const failed = failedStage === 'detection';
  const status = active ? 'Processing' : failed ? 'Failed' : detection ? 'Completed' : 'Waiting';
  const selectedResults = detection?.results.filter((result) => result.source_original_filename === selectedImageName) || [];
  const selectedHandoffErrors = handoffErrors.filter((failure) => failure.original_filename === selectedImageName);
  return <section className="workflow-substage detection-substage">
    <SubstageHeader code="2" title="Module 2 — Particle Track Detection and Segmentation" subtitle="Identify particle trajectories, refine segmented regions, and isolate individual tracks." status={status} active={active} failed={failed} />
    {active && <ProgressMessage title="Detecting particle trajectories" text="Module 1 outputs are being passed automatically to Module 2 as lossless PNG files." />}
    {!detection && !active && <EmptyStage text="Detection, segmentation, and isolated track results will appear here automatically." />}
    {!!selectedHandoffErrors.length && selectedHandoffErrors.map((failure) => <div className="workflow-inline-error" key={failure.original_filename}><strong>{failure.original_filename}:</strong> {failure.error_message}</div>)}
    {detection && !selectedResults.length && !selectedHandoffErrors.length && <EmptyStage text="No detection result is available for this image." />}
    {!!selectedResults.length && <div className="workflow-result-list">{selectedResults.map((result, index) => (
      <article className={`workflow-detection-result ${result.status}`} key={`${result.original_filename}-${index}`}>
        <div className="workflow-result-title"><div><span title={result.status === 'success' ? `Run ${result.run_id}` : 'Module 2 error'}>{result.status === 'success' ? `Run ${result.run_id}` : 'Module 2 error'}</span><strong title={result.source_original_filename}>{result.source_original_filename}</strong></div>{result.status === 'success' ? <CheckCircle size={18} /> : <AlertCircle size={18} />}</div>
        {result.status !== 'success' ? <div className="workflow-inline-error">{result.error_message}</div> : <>
          <div className="workflow-metrics">
            <Metric label="Raw components" value={result.raw_component_count} />
            <Metric label="Trajectory groups" value={result.trajectory_group_count} />
            <Metric label="Linked groups" value={result.linked_group_count} />
            <Metric label="Tracks detected" value={result.accepted_track_count} />
          </div>
          <div className="workflow-segmentation-main">
            <ImagePanel label="Main segmentation result" url={result.outputs.annotated_image} onOpen={() => onOpenImage('Main segmentation result', result.outputs.annotated_image)} />
          </div>
          <div className="workflow-output-grid">
            <ImagePanel label="Preprocessed input" url={preprocessing.results.find((item) => item.original_filename === result.source_original_filename)?.processed_image_url} onOpen={() => onOpenImage('Preprocessed input', preprocessing.results.find((item) => item.original_filename === result.source_original_filename)?.processed_image_url)} />
            <ImagePanel label="Thresholded binary mask" url={result.outputs.binary_mask} onOpen={() => onOpenImage('Thresholded binary mask', result.outputs.binary_mask)} />
            <ImagePanel label="Cleaned segmentation mask" url={result.outputs.cleaned_mask} onOpen={() => onOpenImage('Cleaned segmentation mask', result.outputs.cleaned_mask)} />
            <ImagePanel label="Labelled segmentation mask" url={result.outputs.labelled_mask} onOpen={() => onOpenImage('Labelled segmentation mask', result.outputs.labelled_mask)} />
          </div>
          <div className="workflow-download-row"><span>{result.segmentation_mode} · Foreground {Number(result.foreground_percent.final).toFixed(4)}%</span><a className="btn btn-outline" href={result.csv_url} target="_blank" rel="noreferrer"><Download size={16} /> Track CSV</a></div>
          <div className="workflow-track-heading"><h3>Isolated particle tracks</h3><span>{result.tracks.length} ROI{result.tracks.length === 1 ? '' : 's'}</span></div>
          {result.tracks.length ? <div className="workflow-roi-grid">{result.tracks.map((track) => <article className="workflow-roi" key={track.track_id} title={`Track ID: ${track.track_id}`}><button type="button" className="workflow-roi-image-button" aria-label={`View isolated track ${track.label_id} full picture`} onClick={() => onOpenImage(`Isolated track ${track.label_id}`, track.roi_image_url)}><img src={track.roi_image_url} alt={`Isolated track ${track.label_id}`} /><span className="workflow-expand-indicator"><Maximize2 size={13} /></span></button><div><strong>Track {track.label_id}</strong><span>{track.bbox_width} × {track.bbox_height}px · area {track.area_pixels}px²</span></div></article>)}</div> : <div className="workflow-empty"><FileImage size={19} /> No track candidates passed the Module 2 filters.</div>}
        </>}
      </article>
    ))}</div>}
  </section>;
}

function GeometricClassificationSection({ stage, failedStage, detection, classifications, selectedImageName, onOpenImage }) {
  const active = stage === 'classifying';
  const failed = failedStage === 'classification';
  const completed = stage === 'completed';
  const status = active ? 'Processing' : failed ? 'Failed' : completed ? 'Completed' : 'Waiting';
  const selectedResults = detection?.results.filter((result) => result.source_original_filename === selectedImageName) || [];
  const successfulResults = selectedResults.filter((result) => result.status === 'success');
  return <section className="workflow-substage classification-substage">
    <SubstageHeader code="3" title="Module 3 — Geometric Analysis and Particle Classification" subtitle="Extract geometric track features and classify detected particles." status={status} active={active} failed={failed} />
    {active && <ProgressMessage title="Analysing and classifying extracted tracks" text="Module 3 is measuring the existing ROI masks and classifying each detected particle." />}
    {!detection && !active && <EmptyStage text="Geometric features and particle classifications will appear after track detection." />}
    {detection && !successfulResults.length && <EmptyStage text="No successful Module 2 result is available for geometric analysis for this image." />}
    {!!successfulResults.length && <div className="workflow-classification-results">{successfulResults.map((result) => <article className="workflow-analysis-result" key={result.run_id}>
      <div className="workflow-result-title"><div><span title={`Module 3 · Run ${result.run_id}`}>Module 3 · Run {result.run_id}</span><strong title={result.source_original_filename}>{result.source_original_filename}</strong></div>{completed && <CheckCircle size={18} />}</div>
      <div className="workflow-track-heading"><h3>Geometric features and classifications</h3><span>{result.tracks.length} track{result.tracks.length === 1 ? '' : 's'}</span></div>
      {result.tracks.length ? <div className="workflow-track-grid">{result.tracks.map((track) => <TrackResult key={track.track_id} result={result} track={track} classification={classifications[`${result.run_id}:${track.track_id}`]} onOpenImage={onOpenImage} />)}</div> : <div className="workflow-empty"><FileImage size={19} /> No isolated tracks are available for classification.</div>}
    </article>)}</div>}
  </section>;
}

function TrackResult({ result, track, classification, onOpenImage }) {
  const view = classificationView(classification || (track.roi_mask_url ? { status: 'loading' } : { status: 'unavailable' }));
  return <article className="workflow-track">
    <header className="workflow-track-header">
      <div>
        <strong>Track {track.label_id}</strong>
        <span title={`Track ID: ${track.track_id}`}>Track ID: {track.track_id}</span>
      </div>
      <div className={`workflow-track-verdict ${view.kind} ${view.kind === 'success' ? `particle-${view.title.toLowerCase()}` : ''}`}>
        {view.kind === 'loading' && <Loader2 className="spin" size={16} />}
        <strong>{view.title}</strong>
        {view.kind === 'success' && <span>{view.confidence} confidence</span>}
      </div>
    </header>

    <div className="workflow-track-layout">
      <aside className="workflow-track-visual">
        <button type="button" className="workflow-track-image-link" aria-label={`View Track ${track.label_id} ROI full picture`} onClick={() => onOpenImage(`Track ${track.label_id} ROI`, track.roi_image_url)}>
          <img src={track.roi_image_url} alt={`Track ${track.label_id} ROI`} />
          <span className="workflow-expand-indicator"><Maximize2 size={15} /></span>
        </button>
        <div className="workflow-track-dimensions">
          <span><strong>{track.bbox_width} × {track.bbox_height}</strong> px</span>
          <span><strong>{track.area_pixels}</strong> px² area</span>
        </div>
        <div className="workflow-track-actions">
          <button type="button" className="btn btn-outline" onClick={() => onOpenImage(`Track ${track.label_id} ROI`, track.roi_image_url)}>View ROI</button>
          <button type="button" className="btn btn-outline" onClick={() => onOpenImage(`Track ${track.label_id} mask`, track.roi_mask_url)}>View Mask</button>
        </div>
      </aside>

      <div className="workflow-track-details">
        <section className="workflow-features-panel">
          <h4>Geometric Features</h4>
          {view.kind === 'success' && <dl className="workflow-feature-grid">{view.features.map((feature) => <div key={feature.key}><dt>{feature.label}</dt><dd><span>{feature.value}</span>{feature.physicalValue && <small>{feature.physicalValue}</small>}</dd></div>)}</dl>}
          {view.kind !== 'success' && <div className="workflow-feature-pending">Feature values are shown when classification completes.</div>}
        </section>

        <section className={`workflow-classification ${view.kind} ${view.kind === 'success' ? `particle-${view.title.toLowerCase()}` : ''}`}>
          <span className="workflow-classification-label">Particle classification</span>
          {view.kind === 'loading' && <Loader2 className="spin" size={18} />}
          <div>
            <strong>{view.title}</strong>
            {view.kind === 'success' && <b>{view.confidence}</b>}
          </div>
          {view.kind === 'success'
            ? <p><span>Alpha <strong>{view.alpha}</strong></span><span>Electron <strong>{view.electron}</strong></span></p>
            : <p>{view.message}</p>}
        </section>
      </div>
    </div>

    <footer className="workflow-track-footer">
      <span>Track ID <code title={track.track_id}>{track.track_id}</code></span>
      <span>Run ID <code title={result.run_id}>{result.run_id}</code></span>
    </footer>
  </article>;
}

export default ProcessingWorkflow;
