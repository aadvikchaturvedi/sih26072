/**
 * data/schemas.ts
 * Zod schemas for validating API responses.
 */
import { z } from 'zod';

export const InputStatusSchema = z.enum(['live', 'stale', 'missing']);
export const WarningLevelSchema = z.enum(['green', 'yellow', 'orange', 'red']);
export const CellStatusSchema = z.enum(['initiating', 'growing', 'mature', 'decaying']);
export const LightningTrendSchema = z.enum(['rapidly_increasing', 'increasing', 'steady', 'decreasing']);
export const RiskLevelSchema = z.enum(['high', 'moderate', 'low']);

export const InputHealthSchema = z.object({
  name: z.enum(['radar', 'satellite', 'lightning', 'nwp']),
  status: InputStatusSchema,
  lastReceived: z.string().nullable(),
  staleAgeSeconds: z.number().nullable(),
});

export const HealthStatusSchema = z.object({
  lastRunAt: z.string(),
  nextRunAt: z.string(),
  mode: z.enum(['live', 'replay']),
  modelMode: z.enum(['full', 'satellite_only']),
  modelName: z.string(),
  modelVersion: z.string(),
  missingChannels: z.array(z.string()),
  inputs: z.array(InputHealthSchema),
  hasEnsemble: z.boolean(),
  inferenceMsLast: z.number().nullable(),
});

export const ForecastAttrsSchema = z.object({
  model_name: z.string(),
  model_version: z.string(),
  t0: z.string(),
  mode: z.enum(['full', 'satellite_only']),
  missing_channels: z.array(z.string()),
  inference_ms: z.number(),
  output_contract_version: z.literal('1.0'),
  first_flash_threshold: z.number(),
});

export const ForecastResponseSchema = z.object({
  reflectivityUrlByLead: z.record(z.string(), z.string()),
  lightningProb30Url: z.string(),
  lightningProb60Url: z.string(),
  firstFlashUrl: z.string(),
  ensembleSpreadUrl: z.string().optional(),
  attrs: ForecastAttrsSchema,
});

export const CellTrackPointSchema = z.object({
  time: z.string(),
  lat: z.number(),
  lon: z.number(),
  maxDbz: z.number(),
  flashRate: z.number(),
});

export const ForecastPathPointSchema = z.object({
  leadMin: z.number(),
  lat: z.number(),
  lon: z.number(),
  expectedDbz: z.number(),
  risk: RiskLevelSchema,
  riskBasis: z.enum(['lightning_prob', 'reflectivity_based']),
});

export const CellSchema = z.object({
  id: z.string(),
  isNewCell: z.boolean(),
  status: CellStatusSchema,
  lat: z.number(),
  lon: z.number(),
  maxDbz: z.number(),
  growthDbzPer10Min: z.number(),
  lightningTrend: LightningTrendSchema,
  headingDeg: z.number(),
  speedKmh: z.number(),
  lightningJumpFlag: z.boolean(),
  lightningJumpAt: z.string().nullable(),
  track: z.array(CellTrackPointSchema),
  forecastPath: z.array(ForecastPathPointSchema),
  uncertaintyCone: z.array(z.object({ leadMin: z.number(), radiusKm: z.number() })),
  affectedDistricts: z.array(z.object({
    districtId: z.string(),
    name: z.string(),
    etaMin: z.number(),
  })),
  confidence: z.number().optional(),
  modelMode: z.enum(['full', 'satellite_only']),
  dataQuality: z.number(),
  flashRateSeries: z.array(z.object({ time: z.string(), rate: z.number() })),
  dbzSeries: z.array(z.object({ time: z.string(), dbz: z.number(), isForecast: z.boolean() })),
});

export const LightningStrokeSchema = z.object({
  id: z.string(),
  time: z.string(),
  lat: z.number(),
  lon: z.number(),
  polarity: z.enum(['positive', 'negative']).optional(),
  peakCurrentKA: z.number().optional(),
  cellId: z.string().nullable(),
});

export const DistrictForecastSchema = z.object({
  districtId: z.string(),
  name: z.string(),
  lightningProb30: z.number(),
  lightningProb60: z.number(),
  maxDbz: z.number(),
  areaFractionAboveThreshold: z.number(),
  warningLevel: WarningLevelSchema,
});

export const WarningSchema = z.object({
  id: z.string(),
  level: WarningLevelSchema,
  districtId: z.string(),
  districtName: z.string(),
  causeText: z.string(),
  causeCell: z.string().nullable(),
  ruleTriggered: z.string(),
  triggerValue: z.number(),
  triggerThreshold: z.number(),
  issuedAt: z.string(),
  validUntil: z.string(),
  issuedBy: z.enum(['system', 'forecaster']),
  previousLevel: WarningLevelSchema.nullable(),
  isLevelChange: z.boolean(),
  capXml: z.string().optional(),
});

export const SkillCurvePointSchema = z.object({
  leadMin: z.number(),
  csi: z.number(),
});

export const ModelSkillSchema = z.object({
  modelId: z.string(),
  label: z.string(),
  color: z.string(),
  csiByThreshold: z.record(z.string(), z.array(SkillCurvePointSchema)),
  reliabilityDiagram: z.array(z.object({
    forecastProb: z.number(),
    observedFreq: z.number(),
    count: z.number(),
  })),
  brierSkillScore: z.number().nullable(),
  rocAuc: z.number().nullable(),
  pod: z.number().nullable(),
  far: z.number().nullable(),
  firstFlashHitRate: z.number().nullable(),
  firstFlashFAR: z.number().nullable(),
  medianFirstFlashLeadMin: z.number().nullable(),
});

export const SkillReportSchema = z.object({
  reportAt: z.string(),
  medianFirstFlashLeadMin: z.number().nullable(),
  models: z.array(ModelSkillSchema),
  eventLabel: z.string(),
  eventStartAt: z.string().nullable(),
  eventEndAt: z.string().nullable(),
  synthetic: z.boolean().optional(),
});

export const TimelineSchema = z.object({
  times: z.array(z.string()).min(1),
  stepMin: z.number(),
  bbox: z.object({ west: z.number(), east: z.number(), south: z.number(), north: z.number() }),
});

export const LiveTickSchema = z.object({
  t0: z.string(),
  newWarnings: z.array(WarningSchema),
  updatedCells: z.array(CellSchema),
  health: HealthStatusSchema,
});
