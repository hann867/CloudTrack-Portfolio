export const MODULE3_FEATURES = [
  { key: 'track_length_px', label: 'Track length', decimals: 2, unit: 'px', physicalKey: 'track_length_mm', physicalUnit: 'mm' },
  { key: 'mean_width_px', label: 'Mean width', decimals: 2, unit: 'px', physicalKey: 'mean_width_mm', physicalUnit: 'mm' },
  { key: 'median_width_px', label: 'Median width', decimals: 2, unit: 'px', physicalKey: 'median_width_mm', physicalUnit: 'mm' },
  { key: 'max_width_px', label: 'Maximum width', decimals: 2, unit: 'px', physicalKey: 'max_width_mm', physicalUnit: 'mm' },
  { key: 'mean_curvature', label: 'Mean curvature', decimals: 4, unit: 'px⁻¹' },
  { key: 'max_curvature', label: 'Maximum curvature', decimals: 4, unit: 'px⁻¹' },
  { key: 'tortuosity', label: 'Tortuosity', decimals: 3, unit: '' },
];

export const formatProbability = (value) => `${(Number(value) * 100).toFixed(1)}%`;

const MODULE3_MODEL_LABELS = Object.freeze({
  module3_rf_development_v1: 'Development RF v1',
  module3_rf_development_v2: 'Development RF v2',
});

export const featureRows = (features = {}, physicalFeatures = {}) => MODULE3_FEATURES.map((definition) => {
  const value = Number(features[definition.key]);
  const physicalValue = Number(physicalFeatures[definition.physicalKey]);
  return {
    ...definition,
    value: Number.isFinite(value)
      ? `${value.toFixed(definition.decimals)}${definition.unit ? ` ${definition.unit}` : ''}`
      : 'Unavailable',
    physicalValue: definition.physicalKey && Number.isFinite(physicalValue)
      ? `${physicalValue.toFixed(definition.decimals)} ${definition.physicalUnit}`
      : '',
  };
});

export const developmentModelLabel = (model = {}) => {
  const version = String(model.model_version || '');
  if (MODULE3_MODEL_LABELS[version]) return MODULE3_MODEL_LABELS[version];
  const match = version.match(/_v(\d+)$/i);
  return `Development RF${match ? ` v${match[1]}` : ''}`;
};

export const classificationView = (state) => {
  if (!state || state.status === 'unavailable') {
    return { kind: 'unavailable', title: 'Module 3 classification unavailable', message: 'No valid Module 2 mask' };
  }
  if (state.status === 'loading') {
    return { kind: 'loading', title: 'Classifying…', message: 'Running the development classifier' };
  }
  if (state.status === 'error') {
    return { kind: 'error', title: 'Module 3 classification failed', message: state.message || 'Classification unavailable' };
  }
  const result = state.result;
  return {
    kind: 'success',
    title: result.predicted_class,
    confidence: formatProbability(result.confidence),
    alpha: formatProbability(result.probabilities.Alpha),
    electron: formatProbability(result.probabilities.Electron),
    modelLabel: developmentModelLabel(result.model),
    modelStage: result.model.model_stage,
    features: featureRows(result.features, result.physical_features),
    calibrated: Boolean(result.physical_features && Object.keys(result.physical_features).length),
  };
};
