/**
 * data/types.ts
 * Central type definitions for the SIH26072 Nowcast Console.
 * Fields marked "PROPOSED CONTRACT - agree with backend owner" are defined
 * by this frontend mock and must be matched by future backend components.
 */

// ─── Warning levels ──────────────────────────────────────────────────────────
export type WarningLevel = 'green' | 'yellow' | 'orange' | 'red';

// ─── Input data sources / health chips ───────────────────────────────────────
export type InputStatus = 'live' | 'stale' | 'missing';

export interface InputHealth {
  name: 'radar' | 'satellite' | 'lightning' | 'nwp';
  status: InputStatus;
  /** ISO-8601 UTC of last received data */
  lastReceived: string | null;
  /** Staleness in seconds (null if live) */
  staleAgeSeconds: number | null;
}

// ─── Health / status ─────────────────────────────────────────────────────────
/** PROPOSED CONTRACT - agree with backend owner (F9) */
export interface HealthStatus {
  /** ISO-8601 UTC of last successful pipeline run */
  lastRunAt: string;
  /** ISO-8601 UTC of next scheduled run */
  nextRunAt: string;
  /** 'live' or 'replay' */
  mode: 'live' | 'replay';
  /** Model mode derived from ML output attrs */
  modelMode: 'full' | 'satellite_only';
  modelName: string;
  modelVersion: string;
  /** Channels missing in last run */
  missingChannels: string[];
  inputs: InputHealth[];
  /** Whether any ensemble members were used in last run */
  hasEnsemble: boolean;
  inferenceMsLast: number | null;
}

// ─── Forecast response ────────────────────────────────────────────────────────
/**
 * Exactly matching ML output contract v1.0 (Section 7b of the design spec).
 * Keys must NOT be renamed; a backend adapter is a thin mapping.
 */
export interface ForecastAttrs {
  model_name: string;
  model_version: string;
  /** ISO-8601 UTC ending in Z */
  t0: string;
  mode: 'full' | 'satellite_only';
  /** List of missing input channels */
  missing_channels: string[];
  inference_ms: number;
  output_contract_version: '1.0';
  first_flash_threshold: number;
}

export interface ForecastResponse {
  /** Map from lead minutes (10,20,...120) to image URL for reflectivity */
  reflectivityUrlByLead: Record<number, string>;
  /** URL for P(flash in 30 min) raster */
  lightningProb30Url: string;
  /** URL for P(flash in 60 min) raster */
  lightningProb60Url: string;
  /** URL for first-flash mask raster */
  firstFlashUrl: string;
  /** Ensemble spread uncertainty raster (optional, when members exist) */
  ensembleSpreadUrl?: string;
  attrs: ForecastAttrs;
}

// ─── Cell tracking ───────────────────────────────────────────────────────────
/** PROPOSED CONTRACT - agree with backend owner (F5 cell tracker) */
export type CellStatus = 'initiating' | 'growing' | 'mature' | 'decaying';
export type LightningTrend = 'rapidly_increasing' | 'increasing' | 'steady' | 'decreasing';
export type RiskLevel = 'high' | 'moderate' | 'low';

export interface CellTrackPoint {
  time: string; // ISO-8601 UTC
  lat: number;
  lon: number;
  maxDbz: number;
  flashRate: number; // flashes per minute
}

export interface ForecastPathPoint {
  leadMin: number;
  lat: number;
  lon: number;
  expectedDbz: number;
  /** risk derived from ML lightning prob */
  risk: RiskLevel;
  /** 'lightning_prob' for +30/+60, 'reflectivity_based' for +90/+120 */
  riskBasis: 'lightning_prob' | 'reflectivity_based';
}

export interface UncertaintyConePoint {
  leadMin: number;
  radiusKm: number;
}

export interface AffectedDistrict {
  districtId: string;
  name: string;
  etaMin: number; // ETA in minutes
}

export interface Cell {
  id: string;
  /** Whether this cell was initiated by the ML model (not in observation) */
  isNewCell: boolean;
  status: CellStatus;
  /** Current centroid */
  lat: number;
  lon: number;
  maxDbz: number;
  /** dBZ change per 10-minute interval */
  growthDbzPer10Min: number;
  lightningTrend: LightningTrend;
  /** Heading in degrees (0=N, clockwise) */
  headingDeg: number;
  /** Speed in km/h */
  speedKmh: number;
  /** Whether a lightning jump (2-sigma) was detected */
  lightningJumpFlag: boolean;
  lightningJumpAt: string | null; // ISO-8601 UTC
  /** Historical track points */
  track: CellTrackPoint[];
  /** Forecast path per lead */
  forecastPath: ForecastPathPoint[];
  uncertaintyCone: UncertaintyConePoint[];
  /** Districts likely to be affected */
  affectedDistricts: AffectedDistrict[];
  /**
   * Confidence is a provisional value (mock: ensemble spread + data quality).
   * Info tooltip: "Provisional: definition pending backend (ensemble spread + input completeness)"
   * Optional so it can be hidden if the backend does not supply it.
   */
  confidence?: number;
  /** 'full' or 'satellite_only' - from ForecastAttrs.mode */
  modelMode: 'full' | 'satellite_only';
  dataQuality: number; // 0-1
  /** Flash rate history (last 60 min) for charting */
  flashRateSeries: Array<{ time: string; rate: number }>;
  /** Max dBZ history (last 60 min + forecast) for charting */
  dbzSeries: Array<{ time: string; dbz: number; isForecast: boolean }>;
}

// ─── Lightning strokes ────────────────────────────────────────────────────────
export interface LightningStroke {
  id: string;
  time: string; // ISO-8601 UTC
  lat: number;
  lon: number;
  /** Not in the input contract's flash table; present only if the network reports it */
  polarity?: 'positive' | 'negative';
  peakCurrentKA?: number;
  cellId: string | null;
}

// ─── District data ────────────────────────────────────────────────────────────
/** PROPOSED CONTRACT - agree with backend owner (F8) */
export interface DistrictForecast {
  districtId: string;
  name: string;
  /** P(lightning) in next 30 min, 0-1 */
  lightningProb30: number;
  /** P(lightning) in next 60 min, 0-1 */
  lightningProb60: number;
  maxDbz: number;
  /** Fraction of district area with dBZ >= threshold */
  areaFractionAboveThreshold: number;
  warningLevel: WarningLevel;
}

// ─── Warnings ─────────────────────────────────────────────────────────────────
/** PROPOSED CONTRACT - agree with backend owner (F8 warnings/CAP) */
export interface Warning {
  id: string;
  level: WarningLevel;
  districtId: string;
  districtName: string;
  /** Human-readable cause description */
  causeText: string;
  /** Cell ID if triggered by a specific cell */
  causeCell: string | null;
  /** Rule that triggered the warning */
  ruleTriggered: string;
  /** Numerical value that triggered the rule */
  triggerValue: number;
  /** Threshold that was exceeded */
  triggerThreshold: number;
  issuedAt: string; // ISO-8601 UTC
  validUntil: string; // ISO-8601 UTC
  /** Source of the warning */
  issuedBy: 'system' | 'forecaster';
  /** Previous level for this district (null if first warning) */
  previousLevel: WarningLevel | null;
  /** Whether this is a level change (suppressed if level unchanged) */
  isLevelChange: boolean;
  /** Full CAP XML (generated by frontend in mock mode) */
  capXml?: string;
}

// ─── Skill report ─────────────────────────────────────────────────────────────
/** PROPOSED CONTRACT - agree with backend owner (F7 evaluation/report.py) */
export interface SkillCurvePoint {
  leadMin: number;
  csi: number;
}

export interface ModelSkill {
  modelId: string;
  label: string;
  color: string;
  /** CSI curves at each dBZ threshold */
  csiByThreshold: Record<string, SkillCurvePoint[]>;
  /** Reliability diagram: (forecast_prob, observed_freq) pairs */
  reliabilityDiagram: Array<{ forecastProb: number; observedFreq: number; count: number }>;
  /** Scores are null when they could not be computed (e.g. FAR with no warnings issued) */
  brierSkillScore: number | null;
  rocAuc: number | null;
  pod: number | null;
  far: number | null;
  /** First-flash specific metrics */
  firstFlashHitRate: number | null;
  firstFlashFAR: number | null;
  /** Median lead time before first flash, in minutes */
  medianFirstFlashLeadMin: number | null;
}

export interface SkillReport {
  /** Generated timestamp */
  reportAt: string; // ISO-8601 UTC
  /** Headline metric: median first-flash lead time in minutes */
  medianFirstFlashLeadMin: number | null;
  models: ModelSkill[];
  /** Evaluation event metadata */
  eventLabel: string;
  eventStartAt: string | null;
  eventEndAt: string | null;
  /** True when the scores come from synthetic data and say nothing about real skill */
  synthetic?: boolean;
}

// ─── Live tick ────────────────────────────────────────────────────────────────
export interface LiveTick {
  t0: string; // ISO-8601 UTC: the new analysis time
  newWarnings: Warning[];
  updatedCells: Cell[];
  health: HealthStatus;
}
