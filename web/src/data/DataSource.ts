/**
 * data/DataSource.ts
 * The single swappable data layer interface.
 * UI components must never know which implementation is active.
 */
import type {
  ForecastResponse,
  Cell,
  Warning,
  SkillReport,
  HealthStatus,
  LiveTick,
  LightningStroke,
  DistrictForecast,
} from './types';
import type { Timeline } from './timeline';

export interface DataSource {
  /** Whether new forecasts arrive while the app is open (drives the live subscription) */
  readonly live: boolean;
  /** The analysis times available for replay and the map extent */
  getTimeline(): Promise<Timeline>;
  /** Get forecast imagery URLs and metadata for a given analysis time */
  getForecast(t0: string): Promise<ForecastResponse>;
  /** Get observed radar image URL for a specific time */
  getObservedRadarUrl(t: string): string;
  /** Get tracked storm cells for a given analysis time */
  getCells(t0: string): Promise<Cell[]>;
  /** Get lightning strokes for a given analysis time (last 30 min window) */
  getLightningStrokes(t0: string): Promise<LightningStroke[]>;
  /** Get per-district forecast and warning levels for a given analysis time */
  getDistricts(t0: string): Promise<DistrictForecast[]>;
  /** Get warnings for a given analysis time */
  getWarnings(t0: string): Promise<Warning[]>;
  /** Get aggregate skill report (changes rarely, can be cached) */
  getSkill(): Promise<SkillReport>;
  /** Get system health status */
  getHealth(t0?: string): Promise<HealthStatus>;
  /** Export a warning as CAP v1.2 XML and JSON */
  exportCap(warningId: string): Promise<{ xml: string; json: object }>;
  /**
   * Subscribe to live ticks.
   * In mock mode this uses a setInterval driven by the replay clock.
   * In API mode this uses a native WebSocket to /live.
   * Returns an unsubscribe function.
   */
  subscribe(onTick: (msg: LiveTick) => void): () => void;
}
