const API_BASE_URL = import.meta.env?.VITE_API_BASE_URL || 'http://localhost:8000/api';

const readJob = async (path, signal, jobMissing = false) => {
  const timeout = AbortSignal.timeout(15000);
  const response = await fetch(`${API_BASE_URL}/${path}`, {
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout, cache: 'no-store',
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw Object.assign(new Error(payload.detail || `Unable to load job (${response.status}).`), {
      status: response.status, jobMissing,
    });
  }
  return response.json();
};

export class Module3ApiError extends Error {
  constructor(message, { errorCode = 'module3_request_failed', status = 0 } = {}) {
    super(message);
    this.name = 'Module3ApiError';
    this.errorCode = errorCode;
    this.status = status;
  }
}

const module3Error = async (response) => {
  const payload = await response.json().catch(() => ({}));
  const detail = typeof payload.detail === 'object' && payload.detail !== null
    ? payload.detail
    : payload;
  const message = detail.error_message
    || (typeof payload.detail === 'string' ? payload.detail : '')
    || `Module 3 classification failed with status ${response.status}.`;
  return new Module3ApiError(message, {
    errorCode: detail.error_code || 'module3_request_failed',
    status: response.status,
  });
};

const MAX_MODULE3_BATCH_TRACKS = 32;

const failedBatchTrack = (errorCode, message, provenance = {}) => ({
  processing_status: 'failed',
  error_code: errorCode,
  error_message: message,
  predicted_class: null,
  confidence: null,
  probabilities: {},
  features: {},
  model: {},
  provenance,
});

const summarizeBatchResults = (results) => {
  const successful = results.filter((result) => result.processing_status === 'success');
  const alpha = successful.filter((result) => result.predicted_class === 'Alpha').length;
  const electron = successful.filter((result) => result.predicted_class === 'Electron').length;
  const total = results.length;
  const failed = total - successful.length;
  return {
    processing_status: total > 0 && failed === 0
      ? 'SUCCESS'
      : successful.length > 0 ? 'PARTIAL_SUCCESS' : 'FAILED',
    error_code: total ? null : 'empty_batch',
    error_message: total ? null : 'At least one Module 2 mask is required for batch classification.',
    summary: {
      total_tracks: total,
      successful_tracks: successful.length,
      failed_tracks: failed,
      alpha_count: alpha,
      electron_count: electron,
      success_rate: total ? successful.length / total : 0,
    },
    results,
  };
};

const classifyTrackMaskBatchChunk = async (items) => {
  const downloads = await Promise.all(items.map(async ({ maskUrl, provenance = {} }, index) => {
    if (!maskUrl) {
      return {
        index,
        failure: failedBatchTrack('missing_module2_mask', 'No valid Module 2 mask is available.', provenance),
      };
    }
    try {
      const response = await fetch(maskUrl);
      if (!response.ok) {
        return {
          index,
          failure: failedBatchTrack('mask_fetch_failed', 'The Module 2 mask is unavailable.', provenance),
        };
      }
      return { index, blob: await response.blob(), provenance };
    } catch {
      return {
        index,
        failure: failedBatchTrack('mask_fetch_failed', 'The Module 2 mask could not be downloaded.', provenance),
      };
    }
  }));

  const valid = downloads.filter(({ failure }) => !failure);
  const ordered = downloads.map(({ failure }) => failure || null);
  if (valid.length) {
    const formData = new FormData();
    const metadata = [];
    valid.forEach(({ blob, provenance, index }) => {
      const safeTrackId = String(provenance.track_id || `track_${index + 1}`)
        .replace(/[^A-Za-z0-9._-]+/g, '_');
      formData.append('files', new File([blob], `${safeTrackId}_${index + 1}_mask.png`, { type: 'image/png' }));
      metadata.push(Object.fromEntries([
        'request_id',
        'track_id',
        'component_id',
        'label_id',
        'bbox_x',
        'bbox_y',
        'bbox_width',
        'bbox_height',
        'roi_x0',
        'roi_y0',
        'mm_per_pixel',
        'measurement_homography',
      ].flatMap((field) => {
        const value = provenance[field];
        return value !== undefined && value !== null && value !== '' ? [[field, value]] : [];
      })));
    });
    formData.append('provenance_json', JSON.stringify(metadata));
    const response = await fetch(`${API_BASE_URL}/classification/classify-batch`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) throw await module3Error(response);
    const serverBatch = await response.json();
    valid.forEach(({ index }, resultIndex) => { ordered[index] = serverBatch.results[resultIndex]; });
  }
  return summarizeBatchResults(ordered);
};

export const apiService = {
  async startImageJob(files, workflow = false, calibrations = {}) {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));
    formData.append('workflow', String(workflow));
    formData.append('calibrations_json', JSON.stringify(calibrations));
    const response = await fetch(`${API_BASE_URL}/image/process`, { method: 'POST', body: formData });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || 'Unable to start image processing.');
    }
    return response.json();
  },
  getImageStatus: (id, signal) => readJob(`image/status/${encodeURIComponent(id)}`, signal, true),
  getImageResult: (id, signal) => readJob(`image/result/${encodeURIComponent(id)}`, signal),
  /**
   * Downloads a backend image without re-encoding it and exposes it as a File
   * suitable for the next module's multipart request.
   * @param {string} imageUrl Public backend image URL
   * @param {string} filename Filename to send to the next module
   * @returns {Promise<File>}
   */
  async imageUrlToFile(imageUrl, filename = 'processed_image.png') {
    if (!imageUrl) throw new Error('The preprocessing response did not include an output image URL.');

    let response;
    try {
      response = await fetch(imageUrl);
    } catch {
      throw new Error('The preprocessed image could not be downloaded for track detection.');
    }
    if (!response.ok) {
      throw new Error(`The preprocessed image is unavailable (status ${response.status}).`);
    }

    const blob = await response.blob();
    if (!blob.size) throw new Error('The preprocessed image returned by Module 1 is empty.');
    return new File([blob], filename, { type: blob.type || 'image/png' });
  },

  /**
   * Fetches the health status of the backend API.
   * @returns {Promise<{status: string, service: string}>}
   */
  async checkHealth() {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error("Error communicating with backend:", error);
      throw new Error("Cannot connect to backend server. Please ensure the FastAPI server is running.");
    }
  },

  /**
   * Processes a single image through the Module 1 pipeline.
   * @param {File} file - The image file to process.
   * @returns {Promise<{original_filename: string, status: string, processed_image_url: string}>}
   */
  async processImage(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/preprocessing/image`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Processing failed with status: ${response.status}`);
    }

    return await response.json();
  },

  /**
   * Processes a batch of images through the Module 1 pipeline.
   * @param {File[]} files - The array of image files to process.
   * @returns {Promise<{total_images: number, successful_count: number, failed_count: number, results: Array}>}
   */
  async processBatch(files) {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    const response = await fetch(`${API_BASE_URL}/preprocessing/batch`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Batch processing failed with status: ${response.status}`);
    }

    return await response.json();
  },

  /** Saves a four-corner image-to-chamber measurement calibration. */
  async calculateCalibration(data) {
    const response = await fetch(`${API_BASE_URL}/calibration/calculate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Calibration failed with status: ${response.status}`);
    }

    return await response.json();
  },

  /**
   * Loads optional calibration metadata associated with one source filename.
   * A missing calibration is a valid uncalibrated state.
   * @param {string} imageFilename Exact uploaded source-image filename
   * @returns {Promise<Object|null>}
   */
  async getCalibration(imageFilename) {
    const query = new URLSearchParams({ image_filename: imageFilename });
    const response = await fetch(`${API_BASE_URL}/calibration/metadata?${query}`);
    if (response.status === 404) return null;
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Calibration lookup failed with status: ${response.status}`);
    }
    return await response.json();
  },

  /** Loads reusable camera/chamber calibration profiles. */
  async getCalibrationProfiles() {
    const response = await fetch(`${API_BASE_URL}/calibration/profiles`);
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Calibration profile lookup failed with status: ${response.status}`);
    }
    const payload = await response.json();
    return Array.isArray(payload.profiles) ? payload.profiles : [];
  },

  /**
   * Detects and segments particle tracks in one preprocessed image.
   * @param {File} file Preprocessed PNG or JPEG image
   * @returns {Promise<Object>} Module 2 outputs and track metadata
   */
  async detectTracks(file) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/detection/detect`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Detection failed with status: ${response.status}`);
    }
    return await response.json();
  },

  /**
   * Runs Module 2 independently for multiple preprocessed images.
   * @param {File[]} files Preprocessed PNG or JPEG images
   * @returns {Promise<Object>} Batch detection results
   */
  async detectTracksBatch(files) {
    const formData = new FormData();
    files.forEach((file) => formData.append('files', file));

    const response = await fetch(`${API_BASE_URL}/detection/batch`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Detection failed with status: ${response.status}`);
    }
    return await response.json();
  },

  /**
   * Classifies one existing lossless Module 2 track mask with Module 3.
   * @param {string} maskUrl Public Module 2 ROI mask URL
   * @param {Object} provenance Available non-pixel Module 2 track metadata
   * @returns {Promise<Object>} Structured Module 3 classification response
   */
  async classifyTrackMask(maskUrl, provenance = {}) {
    if (!maskUrl) {
      throw new Module3ApiError('No valid Module 2 mask is available.', {
        errorCode: 'missing_module2_mask',
      });
    }

    let maskResponse;
    try {
      maskResponse = await fetch(maskUrl);
    } catch {
      throw new Module3ApiError('The Module 2 mask could not be downloaded.', {
        errorCode: 'mask_fetch_failed',
      });
    }
    if (!maskResponse.ok) {
      throw new Module3ApiError('The Module 2 mask is unavailable.', {
        errorCode: 'mask_fetch_failed',
        status: maskResponse.status,
      });
    }

    const maskBlob = await maskResponse.blob();
    const safeTrackId = String(provenance.track_id || 'track').replace(/[^A-Za-z0-9._-]+/g, '_');
    const maskFile = new File([maskBlob], `${safeTrackId}_mask.png`, { type: 'image/png' });
    const formData = new FormData();
    formData.append('file', maskFile);
    [
      'track_id',
      'component_id',
      'label_id',
      'bbox_x',
      'bbox_y',
      'bbox_width',
      'bbox_height',
      'roi_x0',
      'roi_y0',
      'mm_per_pixel',
    ].forEach((field) => {
      const value = provenance[field];
      if (value !== undefined && value !== null && value !== '') {
        formData.append(field, String(value));
      }
    });

    const response = await fetch(`${API_BASE_URL}/classification/classify`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) throw await module3Error(response);
    return await response.json();
  },

  /**
   * Classifies existing Module 2 masks through the bounded Module 3 batch API.
   * Input order is preserved across request chunks and mask-download failures.
   * @param {{maskUrl: string, provenance?: Object}[]} items
   * @returns {Promise<Object>} Consolidated Module 3 batch response
   */
  async classifyTrackMasks(items) {
    if (!Array.isArray(items)) {
      throw new Module3ApiError('Module 3 batch input must be an ordered array.', {
        errorCode: 'invalid_batch',
      });
    }
    if (!items.length) return summarizeBatchResults([]);

    const results = [];
    for (let offset = 0; offset < items.length; offset += MAX_MODULE3_BATCH_TRACKS) {
      const chunk = items.slice(offset, offset + MAX_MODULE3_BATCH_TRACKS);
      const batch = await classifyTrackMaskBatchChunk(chunk);
      results.push(...batch.results);
    }
    return summarizeBatchResults(results);
  },

  /** Uploads one video and extracts a timestamp-based 30 FPS PNG sequence. */
  async processVideo(file) {
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE_URL}/video/process`, {
      method: 'POST',
      body: formData,
    });
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Video processing failed with status: ${response.status}`);
    }
    return await response.json();
  },

  getVideoStatus: (id, signal) => readJob(`video/status/${encodeURIComponent(id)}`, signal, true),
  getVideoResult: (id, signal) => readJob(`video/result/${encodeURIComponent(id)}`, signal),

  async getVideoFrameResult(jobId, frameIndex) {
    const response = await fetch(`${API_BASE_URL}/video/result/${encodeURIComponent(jobId)}/frame/${frameIndex}`);
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Unable to load frame result (${response.status}).`);
    }
    return await response.json();
  },

  async generateVideoReport(jobId) {
    const response = await fetch(`${API_BASE_URL}/video/${encodeURIComponent(jobId)}/report`, {
      method: 'POST',
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Unable to generate video report (${response.status}).`);
    }
    return await response.json();
  }
};
