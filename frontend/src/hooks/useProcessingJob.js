import { useEffect, useSyncExternalStore } from 'react';
import { apiService } from '../services/api';
import { createJobSession } from '../services/jobSession';

const sessions = new Map();
export function useProcessingJob(type) {
  if (!sessions.has(type)) {
    const video = type === 'video';
    sessions.set(type, createJobSession({
      key: `cloudtrack_active_${type}_job`,
      startJob: video ? apiService.processVideo : (files, calibrations) => apiService.startImageJob(files, type === 'workflow', calibrations),
      getStatus: video ? apiService.getVideoStatus : apiService.getImageStatus,
      getResult: video ? apiService.getVideoResult : apiService.getImageResult,
      getFailedResult: !video,
    }));
  }
  const session = sessions.get(type);
  const state = useSyncExternalStore(session.subscribe, session.getSnapshot);
  useEffect(() => session.connect(), [session]);
  return { ...state, start: session.start, clear: session.clear };
}
