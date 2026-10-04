/**
 * data/timeline.ts
 * The replay timeline and map extent, supplied by the active DataSource at start-up
 * (mock: the built-in scenario; api: whatever forecasts the backend holds) and
 * refreshed when a live tick announces a new forecast.
 */
import { useSyncExternalStore } from 'react';

export interface Timeline {
  /** Analysis times with a forecast, oldest first (ISO-8601 UTC). Frame n = times[n]. */
  times: string[];
  stepMin: number;
  bbox: { west: number; east: number; south: number; north: number };
}

let current: Timeline | null = null;
const listeners = new Set<() => void>();

export function setTimeline(next: Timeline) {
  current = next;
  listeners.forEach((fn) => fn());
}

export function getTimeline(): Timeline {
  if (!current) throw new Error('Timeline not loaded yet. Call getDataSource() first.');
  return current;
}

function subscribe(fn: () => void) {
  listeners.add(fn);
  return () => listeners.delete(fn);
}

export function useTimeline(): Timeline {
  return useSyncExternalStore(subscribe, getTimeline);
}

export function clampFrame(frame: number, timeline: Timeline): number {
  return Math.max(0, Math.min(timeline.times.length - 1, frame));
}

/** Analysis time (t0) of a replay frame. */
export function frameT0(frame: number, timeline: Timeline): string {
  return timeline.times[clampFrame(frame, timeline)];
}
