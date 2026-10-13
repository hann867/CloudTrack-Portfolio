export const MODULE4_CLASS_COLOURS = Object.freeze({
  Alpha: '#c084fc',
  Electron: '#60a5fa',
});

export function classifiedAnnotation(track, view) {
  if (view?.kind !== 'success') return null;
  const colour = MODULE4_CLASS_COLOURS[view.title];
  if (!colour) return null;
  return {
    colour,
    text: `Track ${track.label_id} · ${view.title} ${view.confidence}`,
  };
}
