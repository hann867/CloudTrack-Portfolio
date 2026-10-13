import { classificationView } from '../pages/module3Presentation.js';
import { classifiedAnnotation } from '../pages/module4ClassifiedAnnotation.js';

const COLOURS = Object.freeze({
  ink: [24, 31, 43],
  muted: [91, 103, 120],
  pale: [244, 247, 251],
  line: [213, 220, 230],
  blue: [59, 130, 246],
  red: [239, 68, 68],
  purple: [168, 85, 247],
  white: [255, 255, 255],
});

const FEATURE_COLUMNS = [
  ['track_length_px', 'Length'],
  ['mean_width_px', 'Mean width'],
  ['median_width_px', 'Median width'],
  ['max_width_px', 'Max width'],
  ['mean_curvature', 'Mean curvature'],
  ['max_curvature', 'Max curvature'],
  ['tortuosity', 'Tortuosity'],
];

const ascii = (value) => String(value ?? '')
  .replace(/[–—−]/g, '-')
  .replace(/→/g, '->')
  .replace(/·/g, '-')
  .replace(/…/g, '...')
  .replace(/×/g, 'x')
  .replace(/²/g, '^2')
  .replace(/⁻¹/g, '^-1')
  .replace(/[^ -~]/g, '');

const number = (value, fallback = 0) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const timestampText = (value) => {
  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.getTime()) ? 'Unavailable' : date.toLocaleString();
};

const reportFilename = (model) => {
  const stem = model.primaryFilename.replace(/\.[^.]+$/, '').replace(/[^A-Za-z0-9._-]+/g, '_');
  return `${stem || 'cloudtrack'}_scientific_report.pdf`;
};

export function buildProcessingReportModel({
  selectedFiles,
  previews,
  preprocessing,
  detection,
  classifications,
  calibrations,
  durationMs,
  processingStartedAt,
  generatedAt = new Date(),
  selectedImageName,
}) {
  const files = Array.from(selectedFiles || []);
  const images = files.map((file) => {
    const filename = file.name;
    const module1 = preprocessing?.results?.find((item) => item.original_filename === filename) || null;
    const module2 = detection?.results?.find((item) => item.source_original_filename === filename) || null;
    const calibration = calibrations?.[filename] || { status: 'uncalibrated' };
    const tracks = module2?.status === 'success' && Array.isArray(module2.tracks)
      ? module2.tracks.map((track) => {
        const state = classifications?.[`${module2.run_id}:${track.track_id}`] || { status: 'unavailable' };
        return { ...track, classificationState: state, view: classificationView(state) };
      })
      : [];
    const classified = tracks.filter((track) => track.view.kind === 'success');
    return {
      filename,
      rawUrl: previews?.[filename] || '',
      module1,
      module2,
      calibration,
      tracks,
      alphaCount: classified.filter((track) => track.view.title === 'Alpha').length,
      electronCount: classified.filter((track) => track.view.title === 'Electron').length,
      classifiedCount: classified.length,
    };
  });
  const primary = images.find((image) => image.filename === selectedImageName)
    || images.find((image) => image.module2?.status === 'success')
    || images[0];
  return {
    generatedAt,
    processingStartedAt,
    durationMs,
    primaryFilename: primary?.filename || 'cloudtrack-run',
    primaryRunId: primary?.module2?.run_id || '',
    images,
    imagesProcessed: images.filter((image) => image.module1?.status === 'success').length,
    detectedTracks: images.reduce((total, image) => total + number(image.module2?.accepted_track_count), 0),
    alphaCount: images.reduce((total, image) => total + image.alphaCount, 0),
    electronCount: images.reduce((total, image) => total + image.electronCount, 0),
    classificationFailures: images.reduce(
      (total, image) => total + image.tracks.filter((track) => track.view.kind === 'error').length,
      0,
    ),
  };
}

const readBlob = (blob) => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => resolve(reader.result);
  reader.onerror = () => reject(new Error('An image could not be read for the PDF report.'));
  reader.readAsDataURL(blob);
});

const loadImage = (url) => new Promise((resolve, reject) => {
  const image = new Image();
  image.onload = () => resolve(image);
  image.onerror = () => reject(new Error('An image could not be decoded for the PDF report.'));
  image.src = url;
});

async function imageAsset(url) {
  if (!url) return null;
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Image unavailable (${response.status}).`);
  const source = await readBlob(await response.blob());
  const image = await loadImage(source);
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('The browser could not prepare an image for the PDF report.');
  context.drawImage(image, 0, 0);
  return { dataUrl: canvas.toDataURL('image/png'), width: canvas.width, height: canvas.height };
}

function drawClassifiedAsset(source, image) {
  if (!source || !image.module2) return null;
  const canvas = document.createElement('canvas');
  canvas.width = source.width;
  canvas.height = source.height;
  const context = canvas.getContext('2d');
  if (!context) return null;
  return loadImage(source.dataUrl).then((baseImage) => {
    context.drawImage(baseImage, 0, 0);
    const fontSize = Math.max(18, canvas.width * 0.014);
    image.tracks.forEach((track) => {
      const annotation = classifiedAnnotation(track, track.view);
      if (!annotation) return;
      const x = number(track.bbox_x, NaN);
      const y = number(track.bbox_y, NaN);
      const width = number(track.bbox_width, NaN);
      const height = number(track.bbox_height, NaN);
      if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0) return;
      const plaqueHeight = fontSize * 1.55;
      const plaqueWidth = Math.min(canvas.width, Math.max(width, annotation.text.length * fontSize * 0.59 + fontSize));
      const plaqueX = Math.max(0, Math.min(x, canvas.width - plaqueWidth));
      const plaqueY = y >= plaqueHeight + 3 ? y - plaqueHeight - 3 : y;
      context.strokeStyle = annotation.colour;
      context.lineWidth = Math.max(2, canvas.width * 0.0015);
      context.strokeRect(x, y, width, height);
      context.fillStyle = annotation.colour;
      context.fillRect(plaqueX, plaqueY, plaqueWidth, plaqueHeight);
      context.fillStyle = '#07090c';
      context.font = `700 ${fontSize}px Inter, Arial, sans-serif`;
      context.textBaseline = 'alphabetic';
      context.fillText(annotation.text, plaqueX + fontSize * 0.5, plaqueY + fontSize * 1.08);
    });
    return { dataUrl: canvas.toDataURL('image/png'), width: canvas.width, height: canvas.height };
  });
}

const imageUrls = (model) => model.images.flatMap((image) => {
  const outputs = image.module2?.outputs || {};
  return [
    image.rawUrl,
    image.module1?.processed_image_url,
    outputs.annotated_image,
    outputs.binary_mask,
    outputs.cleaned_mask,
    outputs.labelled_mask,
    ...image.tracks.flatMap((track) => [track.roi_image_url, track.roi_mask_url]),
  ].filter(Boolean);
});

export async function collectProcessingReportAssets(model) {
  const assets = new Map();
  await Promise.all([...new Set(imageUrls(model))].map(async (url) => {
    try {
      assets.set(url, await imageAsset(url));
    } catch (error) {
      assets.set(url, { error: error.message });
    }
  }));
  for (const image of model.images) {
    const processed = assets.get(image.module1?.processed_image_url);
    if (processed?.dataUrl) {
      try {
        assets.set(`classified:${image.filename}`, await drawClassifiedAsset(processed, image));
      } catch (error) {
        assets.set(`classified:${image.filename}`, { error: error.message });
      }
    }
  }
  return assets;
}

const setText = (doc, size = 10, colour = COLOURS.ink, style = 'normal') => {
  doc.setFont('helvetica', style);
  doc.setFontSize(size);
  doc.setTextColor(...colour);
};

const line = (doc, x1, y1, x2, y2, colour = COLOURS.line) => {
  doc.setDrawColor(...colour);
  doc.line(x1, y1, x2, y2);
};

const pageTitle = (doc, title, subtitle = '', colour = COLOURS.blue) => {
  doc.setFillColor(...colour);
  doc.rect(0, 0, doc.internal.pageSize.getWidth(), 5, 'F');
  setText(doc, 17, COLOURS.ink, 'bold');
  doc.text(ascii(title), 14, 20);
  if (subtitle) {
    setText(doc, 9, COLOURS.muted);
    doc.text(doc.splitTextToSize(ascii(subtitle), doc.internal.pageSize.getWidth() - 28), 14, 27);
  }
  line(doc, 14, 33, doc.internal.pageSize.getWidth() - 14, 33);
};

const addPage = (doc, title, subtitle = '', orientation = 'portrait', colour = COLOURS.blue) => {
  doc.addPage('a4', orientation);
  pageTitle(doc, title, subtitle, colour);
};

function imageBox(doc, asset, caption, x, y, width, height) {
  doc.setFillColor(247, 249, 252);
  doc.setDrawColor(...COLOURS.line);
  doc.roundedRect(x, y, width, height, 2, 2, 'FD');
  if (asset?.dataUrl && asset.width > 0 && asset.height > 0) {
    const scale = Math.min((width - 4) / asset.width, (height - 10) / asset.height);
    const drawWidth = asset.width * scale;
    const drawHeight = asset.height * scale;
    doc.addImage(asset.dataUrl, 'PNG', x + (width - drawWidth) / 2, y + 2 + (height - 10 - drawHeight) / 2, drawWidth, drawHeight);
  } else {
    setText(doc, 8, COLOURS.muted, 'italic');
    doc.text(ascii(asset?.error || 'Image unavailable'), x + width / 2, y + height / 2, { align: 'center', maxWidth: width - 10 });
  }
  setText(doc, 8, COLOURS.ink, 'bold');
  doc.text(ascii(caption), x + width / 2, y + height - 3, { align: 'center', maxWidth: width - 6 });
}

function metricCard(doc, label, value, x, y, width, colour = COLOURS.blue) {
  doc.setFillColor(...COLOURS.pale);
  doc.setDrawColor(...COLOURS.line);
  doc.roundedRect(x, y, width, 21, 2, 2, 'FD');
  doc.setFillColor(...colour);
  doc.rect(x, y, 2.2, 21, 'F');
  setText(doc, 7.5, COLOURS.muted, 'bold');
  doc.text(ascii(label).toUpperCase(), x + 6, y + 7);
  setText(doc, 13, COLOURS.ink, 'bold');
  doc.text(ascii(value), x + 6, y + 16, { maxWidth: width - 9 });
}

function stackedMetricCard(doc, label, value, x, y, width, colour) {
  doc.setFillColor(...COLOURS.pale);
  doc.setDrawColor(...COLOURS.line);
  doc.roundedRect(x, y, width, 27, 2, 2, 'FD');
  doc.setFillColor(...colour);
  doc.rect(x, y, 2.2, 27, 'F');
  setText(doc, 7.2, COLOURS.muted, 'bold');
  doc.text(doc.splitTextToSize(ascii(label).toUpperCase(), width - 10), x + 6, y + 8);
  setText(doc, 14, COLOURS.ink, 'bold');
  doc.text(ascii(value), x + 6, y + 21, { maxWidth: width - 10 });
}

function keyValue(doc, label, value, x, y, width) {
  setText(doc, 8, COLOURS.muted, 'bold');
  doc.text(ascii(label), x, y);
  setText(doc, 9, COLOURS.ink);
  const lines = doc.splitTextToSize(ascii(value || 'Unavailable'), width);
  doc.text(lines, x, y + 5);
  return y + 6 + lines.length * 4;
}

const featureValue = (track, key) => {
  const feature = track.view.kind === 'success' && track.view.features.find((item) => item.key === key);
  if (!feature) return 'Unavailable';
  return feature.physicalValue ? `${feature.value} / ${feature.physicalValue}` : feature.value;
};

const calibrationText = (calibration) => calibration?.status === 'calibrated'
  ? `${calibration.data?.profile_name || 'Unnamed calibration'} - 4-point measurement mapping`
  : 'Not calibrated - measurements reported in pixels';

const calibrationDetails = (calibration) => {
  if (calibration?.status !== 'calibrated') return null;
  const data = calibration.data || {};
  return {
    profile: data.profile_name || 'Unnamed calibration',
    mode: '4-point measurement-only',
    chamberSize: `${number(data.chamber_width_mm).toFixed(2)} x ${number(data.chamber_height_mm).toFixed(2)} mm`,
    horizontalScale: `${number(data.horizontal_mm_per_pixel).toFixed(4)} mm/px average`,
    verticalScale: `${number(data.vertical_mm_per_pixel).toFixed(4)} mm/px average`,
  };
};

function renderCover(doc, model, assets) {
  doc.setFillColor(15, 23, 42);
  doc.rect(0, 0, 210, 24, 'F');
  doc.setFillColor(...COLOURS.blue);
  doc.rect(0, 0, 6, 24, 'F');
  setText(doc, 15.5, COLOURS.white, 'bold');
  doc.text('CloudTrack Particle Image Analysis Report', 14, 16);
  const primary = model.images.find((image) => image.filename === model.primaryFilename);
  const finalAsset = assets.get(`classified:${primary?.filename}`)?.dataUrl
    ? assets.get(`classified:${primary?.filename}`)
    : assets.get(primary?.module1?.processed_image_url);
  const cards = [
    [assets.get(primary?.rawUrl), 'Raw image'],
    [assets.get(primary?.module1?.processed_image_url), 'Preprocessed image'],
    [finalAsset, 'Final classified result'],
  ];
  cards.forEach(([asset, caption], index) => imageBox(doc, asset, caption, 14 + index * 62, 30, 58, 145));
  metricCard(doc, 'Tracks detected', model.detectedTracks, 14, 182, 58, COLOURS.red);
  metricCard(doc, 'Alpha particles', model.alphaCount, 76, 182, 58, COLOURS.purple);
  metricCard(doc, 'Electron particles', model.electronCount, 138, 182, 58, COLOURS.blue);
  setText(doc, 11, COLOURS.ink, 'bold');
  doc.text('Processing Summary', 14, 216);
  line(doc, 14, 220, 196, 220);
  const calibration = primary?.calibration;
  const selectedCalibrationText = calibrationText(calibration);
  const statusText = `Module 1: ${primary?.module1?.status || 'unavailable'} | Module 2: ${primary?.module2?.status || 'unavailable'} | Module 3: ${primary?.classifiedCount || 0}/${primary?.tracks.length || 0} classified | Module 4: reported`;
  let leftY = 230;
  leftY = keyValue(doc, 'Source filename', model.primaryFilename, 14, leftY, 84);
  leftY = keyValue(doc, 'Processing date/time', timestampText(model.processingStartedAt), 14, leftY, 84);
  leftY = keyValue(doc, 'Processing duration', model.durationMs == null ? 'Unavailable' : `${(model.durationMs / 1000).toFixed(2)} seconds`, 14, leftY, 84);
  let rightY = 230;
  rightY = keyValue(doc, 'Run ID', model.primaryRunId || 'Unavailable', 108, rightY, 88);
  rightY = keyValue(doc, 'Calibration', selectedCalibrationText, 108, rightY, 88);
  keyValue(doc, 'Module completion status', statusText, 108, rightY, 88);
}

function renderModule1(doc, image, assets) {
  addPage(doc, 'Module 1 - Image Pre-processing', image.filename);
  imageBox(doc, assets.get(image.rawUrl), 'Original image', 14, 42, 87, 112);
  imageBox(doc, assets.get(image.module1?.processed_image_url), 'Final preprocessed particle image', 109, 42, 87, 112);
  setText(doc, 9, COLOURS.muted, 'italic');
  doc.text('Only preprocessing stages exposed by the current API are included.', 14, 166);
  setText(doc, 10, COLOURS.ink, 'bold');
  doc.text('Spatial Calibration', 14, 184);
  keyValue(doc, 'Run measurement configuration', calibrationText(image.calibration), 14, 193, 182);
}

function renderModule2(doc, image, assets) {
  const result = image.module2;
  addPage(doc, 'Module 2 - Particle Track Detection and Segmentation', image.filename, 'portrait', COLOURS.red);
  if (!result || result.status !== 'success') {
    setText(doc, 11, COLOURS.muted);
    doc.text('No successful Module 2 result is available for this image.', 14, 55);
    return;
  }
  imageBox(doc, assets.get(result.outputs?.annotated_image), 'Main segmentation result', 14, 42, 101, 168);
  const stages = [
    [image.module1?.processed_image_url, 'Preprocessed input'],
    [result.outputs?.binary_mask, 'Thresholded binary mask'],
    [result.outputs?.cleaned_mask, 'Clean segmentation mask'],
    [result.outputs?.labelled_mask, 'Labelled segmentation mask'],
  ];
  stages.forEach(([url, label], index) => imageBox(doc, assets.get(url), label, 121 + (index % 2) * 39.5, 42 + Math.floor(index / 2) * 84, 35.5, 80));
  metricCard(doc, 'Raw components', result.raw_component_count ?? 'Unavailable', 14, 224, 42.5, COLOURS.red);
  metricCard(doc, 'Trajectory groups', result.trajectory_group_count ?? 'Unavailable', 60.5, 224, 42.5, COLOURS.red);
  metricCard(doc, 'Linked groups', result.linked_group_count ?? 'Unavailable', 107, 224, 42.5, COLOURS.red);
  metricCard(doc, 'Tracks detected', result.accepted_track_count ?? image.tracks.length, 153.5, 224, 42.5, COLOURS.red);
}

function renderModule2Rois(doc, image, assets) {
  for (let index = 0; index < image.tracks.length; index += 2) {
    addPage(doc, 'Module 2 - Extracted Track Regions', `${image.filename} - tracks ${index + 1}-${Math.min(index + 2, image.tracks.length)}`, 'portrait', COLOURS.red);
    image.tracks.slice(index, index + 2).forEach((track, position) => {
      const y = 42 + position * 120;
      setText(doc, 8.5, COLOURS.ink, 'bold');
      doc.text(doc.splitTextToSize(`Track ${ascii(track.label_id)} - ${ascii(track.track_id)}`, 182).slice(0, 2), 14, y);
      imageBox(doc, assets.get(track.roi_image_url), 'Track ROI', 14, y + 12, 87, 83);
      imageBox(doc, assets.get(track.roi_mask_url), 'Track mask', 109, y + 12, 87, 83);
      setText(doc, 8, COLOURS.muted);
      doc.text(`ROI ${number(track.bbox_width)} x ${number(track.bbox_height)} px | Area ${number(track.area_pixels)} px^2`, 14, y + 102);
    });
  }
}

function renderModule3(doc, image, assets) {
  if (!image.tracks.length) {
    addPage(doc, 'Module 3 - Geometric Analysis and Particle Classification', image.filename);
    setText(doc, 11, COLOURS.muted);
    doc.text('No isolated tracks were available for geometric analysis or classification.', 14, 55);
    return;
  }
  image.tracks.forEach((track) => {
    addPage(doc, 'Module 3 - Geometric Analysis and Particle Classification', `${image.filename} - Track ${track.label_id}`);
    const colour = track.view.title === 'Alpha' ? COLOURS.purple : track.view.title === 'Electron' ? COLOURS.blue : COLOURS.muted;
    doc.setFillColor(...COLOURS.pale);
    doc.setDrawColor(...colour);
    doc.roundedRect(14, 42, 182, 35, 2, 2, 'FD');
    setText(doc, 7.5, COLOURS.muted, 'bold');
    doc.text('PARTICLE CLASSIFICATION', 20, 51);
    setText(doc, 14, colour, 'bold');
    doc.text(ascii(track.view.title), 20, 63);
    setText(doc, 8.5, COLOURS.ink, 'bold');
    doc.text(track.view.kind === 'success' ? `Confidence: ${ascii(track.view.confidence)}` : ascii(track.view.message), 190, 53, { align: 'right', maxWidth: 82 });
    if (track.view.kind === 'success') {
      setText(doc, 8, COLOURS.ink);
      doc.text(`Alpha probability: ${ascii(track.view.alpha)} | Electron probability: ${ascii(track.view.electron)}`, 190, 65, { align: 'right', maxWidth: 105 });
    }
    imageBox(doc, assets.get(track.roi_image_url), 'Track ROI', 14, 86, 87, 88);
    imageBox(doc, assets.get(track.roi_mask_url), 'Track mask / analysed image', 109, 86, 87, 88);
    setText(doc, 10, COLOURS.ink, 'bold');
    doc.text('Geometric Features', 14, 188);
    track.view.features?.forEach((feature, index) => metricCard(
      doc,
      feature.label,
      feature.physicalValue ? `${feature.value} / ${feature.physicalValue}` : feature.value,
      14 + (index % 3) * 62,
      197 + Math.floor(index / 3) * 27,
      56,
    ));
  });
}

function renderModule4(doc, image, assets) {
  addPage(doc, 'Module 4 - Particle Analytics and Result Management', image.filename);
  imageBox(doc, assets.get(`classified:${image.filename}`), 'Final classified annotation', 14, 42, 126, 136);
  stackedMetricCard(doc, 'Tracks detected', image.module2?.accepted_track_count ?? image.tracks.length, 147, 42, 49, COLOURS.red);
  stackedMetricCard(doc, 'Alpha particles', image.alphaCount, 147, 95, 49, COLOURS.purple);
  stackedMetricCard(doc, 'Electron particles', image.electronCount, 147, 148, 49, COLOURS.blue);
  setText(doc, 11, COLOURS.ink, 'bold');
  doc.text('Calibration', 14, 193);
  line(doc, 14, 197, 196, 197);
  const details = calibrationDetails(image.calibration);
  if (details) {
    let leftY = 207;
    leftY = keyValue(doc, 'Calibration Profile', details.profile, 14, leftY, 84);
    leftY = keyValue(doc, 'Mode', details.mode, 14, leftY + 1, 84);
    keyValue(doc, 'Chamber Size', details.chamberSize, 14, leftY + 1, 84);
    let rightY = 207;
    rightY = keyValue(doc, 'Horizontal Scale', details.horizontalScale, 108, rightY, 88);
    rightY = keyValue(doc, 'Vertical Scale', details.verticalScale, 108, rightY + 1, 88);
    keyValue(doc, 'Processing image', 'Unchanged', 108, rightY + 1, 88);
  } else {
    keyValue(doc, 'Status', 'Not calibrated - measurements reported in pixels', 14, 207, 182);
  }
  setText(doc, 9, COLOURS.muted);
  doc.text('Detailed geometric measurements continue on the following landscape page.', 14, 267);
}

function renderFeatureTable(doc, image) {
  const widths = [18, 27, 27, 27, 27, 27, 27, 23, 28, 25];
  const headers = ['Track', ...FEATURE_COLUMNS.map(([, label]) => label), 'Class', 'Confidence'];
  const startX = 10;
  const drawHeader = (y) => {
    doc.setFillColor(30, 41, 59);
    doc.rect(startX, y, widths.reduce((sum, width) => sum + width, 0), 12, 'F');
    let x = startX;
    headers.forEach((header, index) => {
      setText(doc, 7, COLOURS.white, 'bold');
      doc.text(doc.splitTextToSize(ascii(header), widths[index] - 3), x + 1.5, y + 5);
      x += widths[index];
    });
    return y + 12;
  };
  addPage(doc, 'Geometric Feature Summary', image.filename, 'landscape');
  let y = drawHeader(40);
  image.tracks.forEach((track, rowIndex) => {
    if (y > 188) {
      addPage(doc, 'Geometric Feature Summary - continued', image.filename, 'landscape');
      y = drawHeader(40);
    }
    const row = [
      `Track ${track.label_id}`,
      ...FEATURE_COLUMNS.map(([key]) => featureValue(track, key)),
      track.view.kind === 'success' ? track.view.title : 'Unavailable',
      track.view.kind === 'success' ? track.view.confidence : 'Unavailable',
    ];
    doc.setFillColor(...(rowIndex % 2 ? [248, 250, 252] : COLOURS.white));
    doc.rect(startX, y, widths.reduce((sum, width) => sum + width, 0), 19, 'F');
    let x = startX;
    row.forEach((value, index) => {
      setText(doc, 6.8, COLOURS.ink, index === 0 ? 'bold' : 'normal');
      doc.text(doc.splitTextToSize(ascii(value), widths[index] - 3).slice(0, 3), x + 1.5, y + 5);
      x += widths[index];
    });
    line(doc, startX, y + 19, startX + widths.reduce((sum, width) => sum + width, 0), y + 19);
    y += 19;
  });
  if (!image.tracks.length) {
    setText(doc, 9, COLOURS.muted);
    doc.text('No tracks were available for the feature table.', 10, 60);
  }
}

function renderMetadata(doc, model) {
  addPage(doc, 'Technical Metadata', 'Available provenance and completion details for this report.');
  let y = 44;
  y = keyValue(doc, 'Processing started', timestampText(model.processingStartedAt), 14, y, 182);
  y = keyValue(doc, 'Report generated', timestampText(model.generatedAt), 14, y + 2, 182);
  y = keyValue(doc, 'Images analysed', model.images.length, 14, y + 2, 182);
  model.images.forEach((image, index) => {
    if (y > 248) {
      addPage(doc, 'Technical Metadata - continued');
      y = 44;
    }
    setText(doc, 11, COLOURS.ink, 'bold');
    doc.text(`Image ${index + 1}`, 14, y);
    y += 8;
    y = keyValue(doc, 'Source filename', image.filename, 18, y, 178);
    y = keyValue(doc, 'Run ID', image.module2?.run_id || 'Unavailable', 18, y + 1, 178);
    y = keyValue(doc, 'Calibration scale', calibrationText(image.calibration), 18, y + 1, 178);
    y = keyValue(doc, 'Tracks analysed', image.classifiedCount, 18, y + 1, 178);
    const modelLabels = [...new Set(image.tracks.map((track) => track.view.modelLabel).filter(Boolean))];
    y = keyValue(doc, 'Classification model', modelLabels.join(', ') || 'Unavailable', 18, y + 1, 178);
    y = keyValue(doc, 'Module status', `Module 1: ${image.module1?.status || 'unavailable'} | Module 2: ${image.module2?.status || 'unavailable'} | Module 3: ${image.classifiedCount}/${image.tracks.length} classified | Module 4: reported`, 18, y + 1, 178);
    y += 8;
  });
}

function addFooters(doc) {
  const total = doc.getNumberOfPages();
  for (let page = 1; page <= total; page += 1) {
    doc.setPage(page);
    const width = doc.internal.pageSize.getWidth();
    const height = doc.internal.pageSize.getHeight();
    line(doc, 14, height - 12, width - 14, height - 12);
    setText(doc, 7.5, COLOURS.muted);
    doc.text('CloudTrack scientific processing report', 14, height - 7);
    doc.text(`Page ${page} of ${total}`, width - 14, height - 7, { align: 'right' });
  }
}

export async function createProcessingReportDocument(model, assets) {
  const { jsPDF } = await import('jspdf');
  const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4', compress: true });
  doc.setProperties({
    title: 'CloudTrack Particle Image Analysis Report',
    subject: 'Scientific particle-image processing report',
    author: 'CloudTrack',
    creator: 'CloudTrack',
  });
  renderCover(doc, model, assets);
  model.images.forEach((image) => {
    renderModule1(doc, image, assets);
    renderModule2(doc, image, assets);
    renderModule2Rois(doc, image, assets);
    renderModule3(doc, image, assets);
    renderModule4(doc, image, assets);
    renderFeatureTable(doc, image);
  });
  renderMetadata(doc, model);
  addFooters(doc);
  return doc;
}

export async function exportProcessingReport(input) {
  const model = buildProcessingReportModel(input);
  if (!model.images.length) throw new Error('No source images are available for this report.');
  const assets = await collectProcessingReportAssets(model);
  const doc = await createProcessingReportDocument(model, assets);
  doc.save(reportFilename(model));
  return { filename: reportFilename(model), pageCount: doc.getNumberOfPages() };
}
