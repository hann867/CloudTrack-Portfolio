import { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertCircle,
  CheckCircle,
  Download,
  FileImage,
  Loader2,
  Maximize2,
  RefreshCw,
  ScanSearch,
  Upload,
  X,
} from 'lucide-react';
import { apiService } from '../services/api';
import ImageLightbox from '../components/common/ImageLightbox';
import './Detection.css';
import { classificationView } from './module3Presentation';


const classificationKey = (result, track) => `${result.run_id}:${track.track_id}`;


function Detection() {
  const [files, setFiles] = useState([]);
  const [status, setStatus] = useState('idle');
  const [batch, setBatch] = useState(null);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const [classifications, setClassifications] = useState({});
  const [module3Summary, setModule3Summary] = useState(null);
  const [imageViewer, setImageViewer] = useState(null);
  const inputRef = useRef(null);
  const classificationGeneration = useRef(0);

  const previews = useMemo(
    () => Object.fromEntries(files.map((file) => [file.name, URL.createObjectURL(file)])),
    [files],
  );

  useEffect(
    () => () => Object.values(previews).forEach(URL.revokeObjectURL),
    [previews],
  );

  const addFiles = (incoming) => {
    const supported = Array.from(incoming || []).filter((file) =>
      /\.(png|jpe?g)$/i.test(file.name));
    setFiles((current) => {
      const names = new Set(current.map((file) => file.name));
      return [...current, ...supported.filter((file) => !names.has(file.name))];
    });
    setBatch(null);
    setClassifications({});
    setModule3Summary(null);
    setError(supported.length ? '' : 'Select a preprocessed PNG or JPEG image.');
    setStatus('idle');
  };

  const reset = () => {
    classificationGeneration.current += 1;
    setFiles([]);
    setBatch(null);
    setClassifications({});
    setModule3Summary(null);
    setImageViewer(null);
    setError('');
    setStatus('idle');
    if (inputRef.current) inputRef.current.value = '';
  };

  const classifyDetectedTracks = async (response, generation) => {
    const initial = {};
    const jobs = [];
    response.results.forEach((result) => {
      if (result.status !== 'success') return;
      result.tracks.forEach((track) => {
        const key = classificationKey(result, track);
        if (!track.roi_mask_url) {
          initial[key] = { status: 'unavailable' };
          return;
        }
        initial[key] = { status: 'loading' };
        jobs.push({ key, track });
      });
    });
    setClassifications(initial);
    setModule3Summary(null);
    if (!jobs.length) return;

    try {
      const module3Batch = await apiService.classifyTrackMasks(jobs.map(({ key, track }) => ({
        maskUrl: track.roi_mask_url,
        provenance: { ...track, request_id: key },
      })));
      if (classificationGeneration.current !== generation) return;
      const completed = { ...initial };
      jobs.forEach(({ key }, index) => {
        const result = module3Batch.results[index];
        completed[key] = result?.processing_status === 'success'
          ? { status: 'success', result }
          : {
            status: 'error',
            errorCode: result?.error_code || 'module3_request_failed',
            message: result?.error_message || 'Classification unavailable',
          };
      });
      setClassifications(completed);
      setModule3Summary(module3Batch.summary);
    } catch (classificationError) {
      if (classificationGeneration.current !== generation) return;
      setClassifications((current) => {
        const failed = { ...current };
        jobs.forEach(({ key }) => {
          failed[key] = {
            status: 'error',
            errorCode: classificationError.errorCode,
            message: classificationError.message,
          };
        });
        return failed;
      });
      setModule3Summary({
        total_tracks: jobs.length,
        successful_tracks: 0,
        failed_tracks: jobs.length,
        alpha_count: 0,
        electron_count: 0,
      });
    }
  };

  const runDetection = async () => {
    if (!files.length) return;
    const generation = classificationGeneration.current + 1;
    classificationGeneration.current = generation;
    setStatus('processing');
    setClassifications({});
    setModule3Summary(null);
    setError('');
    try {
      const response = await apiService.detectTracksBatch(files);
      setBatch(response);
      setStatus('completed');
      classifyDetectedTracks(response, generation);
    } catch (requestError) {
      setError(requestError.message);
      setStatus('error');
    }
  };

  const download = (url) => {
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = '';
    anchor.target = '_blank';
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
  };

  const downloadClassifiedImage = async (result) => {
    try {
      await exportClassifiedImage(
        result,
        previews[result.original_filename] || result.outputs.annotated_image,
        classifications,
      );
    } catch (exportError) {
      setError(`Could not create the classified image: ${exportError.message}`);
    }
  };

  return (
    <div className="detection-page">
      <div className="detection-heading">
        <div>
          <h1 className="page-title">Track Detection</h1>
          <p className="page-description">
            Module 2: detect particle trajectories, create segmentation masks, and extract track ROIs.
          </p>
        </div>
        {(files.length > 0 || batch) && (
          <button className="btn btn-outline" onClick={reset} disabled={status === 'processing'}>
            <RefreshCw size={16} /> Start over
          </button>
        )}
      </div>

      {error && (
        <div className="detection-alert">
          <AlertCircle size={20} /> {error}
        </div>
      )}

      {!files.length && !batch && (
        <div
          className={`card detection-dropzone ${dragging ? 'is-dragging' : ''}`}
          onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            addFiles(event.dataTransfer.files);
          }}
          onClick={() => inputRef.current?.click()}
        >
          <div className="detection-drop-icon"><Upload size={32} /></div>
          <h2>Upload preprocessed images</h2>
          <p>Use the particle-only PNG output from Module 1. JPEG images are also supported.</p>
          <button className="btn btn-primary" onClick={(event) => {
            event.stopPropagation();
            inputRef.current?.click();
          }}>
            Choose images
          </button>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept=".png,.jpg,.jpeg"
        multiple
        hidden
        onChange={(event) => addFiles(event.target.files)}
      />

      {!!files.length && status !== 'completed' && (
        <>
          <div className="detection-actions">
            <div>
              <h2>Selected images</h2>
              <span>{files.length} ready for Module 2</span>
            </div>
            <div className="detection-action-buttons">
              <button className="btn btn-outline" onClick={() => inputRef.current?.click()} disabled={status === 'processing'}>
                Add more
              </button>
              <button className="btn btn-primary" onClick={runDetection} disabled={status === 'processing'}>
                {status === 'processing'
                  ? <><Loader2 className="spin" size={17} /> Detecting tracks…</>
                  : <><ScanSearch size={17} /> Run Module 2</>}
              </button>
            </div>
          </div>

          <div className="detection-file-grid">
            {files.map((file) => (
              <div className="card detection-file" key={file.name}>
                <img src={previews[file.name]} alt="Selected preprocessed file" />
                <div className="detection-file-info">
                  <strong title={file.name}>{file.name}</strong>
                  <span>{(file.size / 1024 / 1024).toFixed(2)} MB</span>
                </div>
                <button
                  className="detection-icon-button"
                  title="Remove"
                  onClick={() => setFiles((current) => current.filter((item) => item.name !== file.name))}
                  disabled={status === 'processing'}
                >
                  <X size={18} />
                </button>
              </div>
            ))}
          </div>
        </>
      )}

      {status === 'processing' && (
        <div className="card detection-progress">
          <Loader2 className="spin" size={42} />
          <h2>Finding particle trajectories</h2>
          <p>Module 2 is segmenting components and extracting track regions. Complex images can take a moment.</p>
        </div>
      )}

      {status === 'completed' && batch && (
        <div className="detection-results">
          <div className="card detection-summary">
            <div><span>Images</span><strong>{batch.total_images}</strong></div>
            <div className="success"><span>Successful</span><strong>{batch.successful_count}</strong></div>
            <div className={batch.failed_count ? 'failure' : ''}><span>Failed</span><strong>{batch.failed_count}</strong></div>
          </div>

          {module3Summary && (
            <div className="card module3-batch-summary" aria-live="polite">
              <div className="module3-batch-title">
                <span>Module 3 batch</span>
                <strong>Classification summary</strong>
              </div>
              <div><span>Total classified</span><strong>{module3Summary.successful_tracks}</strong></div>
              <div className="particle-alpha"><span>Alpha</span><strong>{module3Summary.alpha_count}</strong></div>
              <div className="particle-electron"><span>Electron</span><strong>{module3Summary.electron_count}</strong></div>
              <div className={module3Summary.failed_tracks ? 'failure' : ''}>
                <span>Failed</span><strong>{module3Summary.failed_tracks}</strong>
              </div>
            </div>
          )}

          {batch.results.map((result, index) => (
            <section className={`card detection-result ${result.status}`} key={`${result.original_filename}-${index}`}>
              <header>
                <div>
                  <span className="result-kicker">{result.status === 'success' ? `Run ${result.run_id}` : 'Processing error'}</span>
                  <h2>{result.original_filename}</h2>
                </div>
                {result.status === 'success'
                  ? <span className="result-status success"><CheckCircle size={17} /> Complete</span>
                  : <span className="result-status failure"><AlertCircle size={17} /> Failed</span>}
              </header>

              {result.status === 'error' ? (
                <div className="detection-alert">{result.error_message}</div>
              ) : (
                <>
                  <div className="detection-metrics">
                    <div><span>Raw components</span><strong>{result.raw_component_count}</strong></div>
                    <div><span>Trajectory groups</span><strong>{result.trajectory_group_count}</strong></div>
                    <div><span>Linked groups</span><strong>{result.linked_group_count}</strong></div>
                    <div><span>Accepted tracks</span><strong>{result.accepted_track_count}</strong></div>
                  </div>

                  <div className="detection-output-grid">
                    <OutputImage
                      label="Annotated tracks"
                      url={result.outputs.annotated_image}
                      onOpen={() => setImageViewer({ kind: 'image', label: 'Annotated tracks', url: result.outputs.annotated_image })}
                    />
                    <OutputImage
                      label="Final cleaned mask"
                      url={result.outputs.cleaned_mask}
                      onOpen={() => setImageViewer({ kind: 'image', label: 'Final cleaned mask', url: result.outputs.cleaned_mask })}
                    />
                    <OutputImage
                      label="Binary mask"
                      url={result.outputs.binary_mask}
                      onOpen={() => setImageViewer({ kind: 'image', label: 'Binary mask', url: result.outputs.binary_mask })}
                    />
                    <OutputImage
                      label="Labelled mask"
                      url={result.outputs.labelled_mask}
                      onOpen={() => setImageViewer({ kind: 'image', label: 'Labelled mask', url: result.outputs.labelled_mask })}
                    />
                  </div>

                  <div className="detection-download-row">
                    <span>{result.segmentation_mode} · Final foreground {result.foreground_percent.final.toFixed(4)}%</span>
                    <button className="btn btn-outline" onClick={() => download(result.csv_url)}>
                      <Download size={16} /> Download track CSV
                    </button>
                  </div>

                  <div className="detection-track-heading">
                    <h3>Extracted track regions</h3>
                    <span>{result.tracks.length} ROI{result.tracks.length === 1 ? '' : 's'}</span>
                  </div>
                  {result.tracks.length ? (
                    <div className="detection-track-grid">
                      {result.tracks.map((track) => {
                        const key = classificationKey(result, track);
                        const classification = classifications[key]
                          || (track.roi_mask_url ? { status: 'loading' } : { status: 'unavailable' });
                        return (
                          <article className="detection-track" key={track.track_id}>
                            <div className="detection-track-overview">
                              <button
                                type="button"
                                className="track-image-button"
                                aria-label={`View Track ${track.label_id} full picture`}
                                onClick={() => setImageViewer({
                                  kind: 'image',
                                  label: `Track ${track.label_id}`,
                                  url: track.roi_image_url,
                                })}
                              >
                                <img src={track.roi_image_url} alt={`Track ${track.label_id}`} />
                                <span><Maximize2 size={13} /></span>
                              </button>
                              <div>
                                <strong>Track {track.label_id}</strong>
                                <span>{track.bbox_width} × {track.bbox_height}px · area {track.area_pixels}px²</span>
                                <small>{track.curve_fitted ? track.curve_model : 'component trajectory'}</small>
                              </div>
                            </div>
                            <TrackClassification state={classification} />
                          </article>
                        );
                      })}
                    </div>
                  ) : (
                    <div className="detection-empty"><FileImage size={20} /> No track candidates passed the Module 2 filters.</div>
                  )}

                  {!!result.tracks.length && (
                    <section className="classified-image-section">
                      <div className="detection-track-heading">
                        <h3>Classified track annotation</h3>
                        <div className="classified-image-actions">
                          <span>Click the image to view the full picture</span>
                          <button
                            type="button"
                            className="btn btn-outline"
                            disabled={result.tracks.some((track) => (
                              classificationStateForTrack(result, track, classifications).status === 'loading'
                            ))}
                            onClick={() => downloadClassifiedImage(result)}
                          >
                            <Download size={16} /> Download classified image
                          </button>
                        </div>
                      </div>
                      <ClassifiedImage
                        result={result}
                        sourceUrl={previews[result.original_filename] || result.outputs.annotated_image}
                        classifications={classifications}
                        onOpen={() => setImageViewer({
                          kind: 'classified',
                          label: `Classified tracks — ${result.original_filename}`,
                          result,
                          sourceUrl: previews[result.original_filename] || result.outputs.annotated_image,
                        })}
                      />
                    </section>
                  )}
                </>
              )}
            </section>
          ))}
        </div>
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
              sourceUrl={imageViewer.sourceUrl}
              classifications={classifications}
            />
          ) : null}
        </ImageLightbox>
      )}
    </div>
  );
}


function TrackClassification({ state }) {
  const view = classificationView(state);
  if (view.kind !== 'success') {
    return (
      <section className={`module3-result ${view.kind}`} aria-live="polite">
        <div className="module3-status-icon">
          {view.kind === 'loading'
            ? <Loader2 className="spin" size={18} />
            : <AlertCircle size={18} />}
        </div>
        <div>
          <strong>{view.title}</strong>
          <span>{view.message}</span>
        </div>
      </section>
    );
  }

  return (
    <section className={`module3-result success particle-${view.title.toLowerCase()}`} aria-live="polite">
      <div className="module3-result-heading">
        <div>
          <span>Particle classification</span>
          <strong>{view.title}</strong>
        </div>
        <div className="module3-confidence">
          <span>Confidence</span>
          <strong>{view.confidence}</strong>
        </div>
      </div>
      <div className="module3-probabilities" aria-label="Class probabilities">
        <span>Alpha: <strong>{view.alpha}</strong></span>
        <span>Electron: <strong>{view.electron}</strong></span>
      </div>
      <div className="module3-model">
        Model: <strong>{view.modelLabel}</strong> · Model stage: <strong>Development</strong>
      </div>
      <details className="module3-features">
        <summary>Seven geometric features</summary>
        <dl>
          {view.features.map((feature) => (
            <div key={feature.key}>
              <dt>{feature.label}</dt>
              <dd>{feature.value}</dd>
            </div>
          ))}
        </dl>
      </details>
    </section>
  );
}


function OutputImage({ label, url, onOpen }) {
  return (
    <figure className="detection-output">
      <button
        type="button"
        className="detection-image-button"
        aria-label={`View ${label} full picture`}
        onClick={onOpen}
      >
        <img src={url} alt={label} />
        <span className="image-expand-indicator"><Maximize2 size={15} /> View full picture</span>
      </button>
      <figcaption>{label}</figcaption>
    </figure>
  );
}


function classificationStateForTrack(result, track, classifications) {
  return classifications[classificationKey(result, track)]
    || (track.roi_mask_url ? { status: 'loading' } : { status: 'unavailable' });
}


function annotationForTrack(result, track, classifications) {
  const state = classificationStateForTrack(result, track, classifications);
  const view = classificationView(state);
  if (view.kind === 'success') {
    return {
      colour: view.title === 'Alpha' ? '#60a5fa' : '#c084fc',
      text: `Track ${track.label_id} · ${view.title} ${view.confidence}`,
    };
  }
  if (view.kind === 'loading') {
    return { colour: '#f59e0b', text: `Track ${track.label_id} · Classifying…` };
  }
  return { colour: '#f87171', text: `Track ${track.label_id} · Unavailable` };
}


function loadExportImage(sourceUrl) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('The source image could not be loaded.'));
    image.src = sourceUrl;
  });
}


function canvasBlob(canvas) {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error('The browser could not encode the PNG.'));
    }, 'image/png');
  });
}


async function exportClassifiedImage(result, sourceUrl, classifications) {
  const image = await loadExportImage(sourceUrl);
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas export is unavailable.');

  context.drawImage(image, 0, 0);
  const fontSize = Math.max(18, canvas.width * 0.014);
  const plaqueHeight = fontSize * 1.55;
  const strokeWidth = Math.max(2, canvas.width * 0.0015);
  context.font = `700 ${fontSize}px Inter, system-ui, sans-serif`;
  context.textBaseline = 'alphabetic';
  context.lineWidth = strokeWidth;

  result.tracks.forEach((track) => {
    const x = Number(track.bbox_x);
    const y = Number(track.bbox_y);
    const width = Number(track.bbox_width);
    const height = Number(track.bbox_height);
    if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0) return;

    const annotation = annotationForTrack(result, track, classifications);
    const plaqueWidth = Math.min(
      canvas.width,
      Math.max(width, context.measureText(annotation.text).width + fontSize),
    );
    const plaqueX = Math.max(0, Math.min(x, canvas.width - plaqueWidth));
    const plaqueY = y >= plaqueHeight + 3 ? y - plaqueHeight - 3 : y;

    context.strokeStyle = annotation.colour;
    context.strokeRect(x, y, width, height);
    context.fillStyle = annotation.colour;
    context.fillRect(plaqueX, plaqueY, plaqueWidth, plaqueHeight);
    context.fillStyle = '#07090c';
    context.fillText(annotation.text, plaqueX + fontSize * 0.5, plaqueY + fontSize * 1.08);
  });

  const blob = await canvasBlob(canvas);
  const objectUrl = URL.createObjectURL(blob);
  const sourceStem = String(result.original_filename || 'classified_tracks')
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
  const imageContent = (
    <div className="classified-image-frame">
      <img
        src={sourceUrl}
        alt={`Classified particle tracks for ${result.original_filename}`}
        onLoad={(event) => setImageSize({
          width: event.currentTarget.naturalWidth,
          height: event.currentTarget.naturalHeight,
        })}
      />
      {imageSize.width > 0 && imageSize.height > 0 && (
        <svg
          className="classified-image-overlay"
          viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
          aria-hidden="true"
        >
          {result.tracks.map((track) => {
            const x = Number(track.bbox_x);
            const y = Number(track.bbox_y);
            const width = Number(track.bbox_width);
            const height = Number(track.bbox_height);
            if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0) return null;

            const annotation = annotationForTrack(result, track, classifications);
            const fontSize = Math.max(18, imageSize.width * 0.014);
            const plaqueHeight = fontSize * 1.55;
            const plaqueWidth = Math.min(
              imageSize.width,
              Math.max(width, annotation.text.length * fontSize * 0.59 + fontSize),
            );
            const plaqueX = Math.max(0, Math.min(x, imageSize.width - plaqueWidth));
            const plaqueY = y >= plaqueHeight + 3 ? y - plaqueHeight - 3 : y;
            const strokeWidth = Math.max(2, imageSize.width * 0.0015);

            return (
              <g key={track.track_id}>
                <rect
                  x={x}
                  y={y}
                  width={width}
                  height={height}
                  fill="none"
                  stroke={annotation.colour}
                  strokeWidth={strokeWidth}
                  vectorEffect="non-scaling-stroke"
                />
                <rect
                  x={plaqueX}
                  y={plaqueY}
                  width={plaqueWidth}
                  height={plaqueHeight}
                  rx={fontSize * 0.18}
                  fill={annotation.colour}
                />
                <text
                  x={plaqueX + fontSize * 0.5}
                  y={plaqueY + fontSize * 1.08}
                  fill="#07090c"
                  fontSize={fontSize}
                  fontFamily="Inter, system-ui, sans-serif"
                  fontWeight="700"
                >
                  {annotation.text}
                </text>
              </g>
            );
          })}
        </svg>
      )}
      {onOpen && <span className="image-expand-indicator"><Maximize2 size={15} /> View full picture</span>}
    </div>
  );

  if (!onOpen) return imageContent;
  return (
    <button
      type="button"
      className="classified-image-button"
      aria-label="View classified track annotation full picture"
      onClick={onOpen}
    >
      {imageContent}
    </button>
  );
}


export default Detection;
