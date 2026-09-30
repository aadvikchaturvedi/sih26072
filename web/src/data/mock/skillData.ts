/**
 * data/mock/skillData.ts
 * Mock skill/verification data for the Skill page.
 * Plausible, honest numbers: models beat baselines, skill decays with lead time.
 */
import type { SkillReport, ModelSkill } from '../types';

const LEADS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];

function csiCurve(
  peak: number,
  decayRate: number,
  offset = 0,
): Array<{ leadMin: number; csi: number }> {
  return LEADS.map((lead) => ({
    leadMin: lead,
    csi: Math.max(0, peak * Math.exp(-decayRate * (lead - 10 + offset) / 60)),
  }));
}

const MODELS: ModelSkill[] = [
  {
    modelId: 'persistence',
    label: 'Persistence',
    color: '#6B7280',
    csiByThreshold: {
      20: csiCurve(0.52, 1.5),
      35: csiCurve(0.38, 1.8),
      45: csiCurve(0.22, 2.2),
    },
    reliabilityDiagram: [
      { forecastProb: 0.1, observedFreq: 0.18, count: 1200 },
      { forecastProb: 0.3, observedFreq: 0.28, count: 800 },
      { forecastProb: 0.5, observedFreq: 0.41, count: 500 },
      { forecastProb: 0.7, observedFreq: 0.58, count: 300 },
      { forecastProb: 0.9, observedFreq: 0.72, count: 120 },
    ],
    brierSkillScore: 0.08,
    rocAuc: 0.68,
    pod: 0.61,
    far: 0.44,
    firstFlashHitRate: 0.38,
    firstFlashFAR: 0.55,
    medianFirstFlashLeadMin: 0,
  },
  {
    modelId: 'pysteps',
    label: 'pySTEPS',
    color: '#8B5CF6',
    csiByThreshold: {
      20: csiCurve(0.61, 1.2),
      35: csiCurve(0.47, 1.4),
      45: csiCurve(0.31, 1.7),
    },
    reliabilityDiagram: [
      { forecastProb: 0.1, observedFreq: 0.12, count: 1100 },
      { forecastProb: 0.3, observedFreq: 0.31, count: 750 },
      { forecastProb: 0.5, observedFreq: 0.50, count: 480 },
      { forecastProb: 0.7, observedFreq: 0.69, count: 280 },
      { forecastProb: 0.9, observedFreq: 0.85, count: 110 },
    ],
    brierSkillScore: 0.19,
    rocAuc: 0.75,
    pod: 0.70,
    far: 0.36,
    firstFlashHitRate: 0.48,
    firstFlashFAR: 0.45,
    medianFirstFlashLeadMin: 8,
  },
  {
    modelId: 'F3',
    label: 'F3 (SimVP)',
    color: '#06B6D4',
    csiByThreshold: {
      20: csiCurve(0.68, 0.9),
      35: csiCurve(0.54, 1.1),
      45: csiCurve(0.39, 1.4),
    },
    reliabilityDiagram: [
      { forecastProb: 0.1, observedFreq: 0.10, count: 1050 },
      { forecastProb: 0.3, observedFreq: 0.29, count: 720 },
      { forecastProb: 0.5, observedFreq: 0.52, count: 460 },
      { forecastProb: 0.7, observedFreq: 0.71, count: 270 },
      { forecastProb: 0.9, observedFreq: 0.88, count: 100 },
    ],
    brierSkillScore: 0.27,
    rocAuc: 0.81,
    pod: 0.75,
    far: 0.30,
    firstFlashHitRate: 0.58,
    firstFlashFAR: 0.35,
    medianFirstFlashLeadMin: 18,
  },
  {
    modelId: 'F3F4',
    label: 'F3 + F4 (Lightning head)',
    color: '#2F6FEB',
    csiByThreshold: {
      20: csiCurve(0.72, 0.8),
      35: csiCurve(0.60, 1.0),
      45: csiCurve(0.45, 1.2),
    },
    reliabilityDiagram: [
      { forecastProb: 0.1, observedFreq: 0.09, count: 1000 },
      { forecastProb: 0.3, observedFreq: 0.30, count: 700 },
      { forecastProb: 0.5, observedFreq: 0.51, count: 450 },
      { forecastProb: 0.7, observedFreq: 0.72, count: 260 },
      { forecastProb: 0.9, observedFreq: 0.91, count: 95 },
    ],
    brierSkillScore: 0.33,
    rocAuc: 0.85,
    pod: 0.80,
    far: 0.24,
    firstFlashHitRate: 0.68,
    firstFlashFAR: 0.27,
    medianFirstFlashLeadMin: 27, // HEADLINE NUMBER
  },
  {
    modelId: 'satellite_only',
    label: 'Satellite-only',
    color: '#F59E0B',
    csiByThreshold: {
      20: csiCurve(0.55, 1.1, 10),
      35: csiCurve(0.40, 1.4, 10),
      45: csiCurve(0.25, 1.8, 10),
    },
    reliabilityDiagram: [
      { forecastProb: 0.1, observedFreq: 0.14, count: 980 },
      { forecastProb: 0.3, observedFreq: 0.26, count: 660 },
      { forecastProb: 0.5, observedFreq: 0.45, count: 420 },
      { forecastProb: 0.7, observedFreq: 0.62, count: 240 },
      { forecastProb: 0.9, observedFreq: 0.78, count: 90 },
    ],
    brierSkillScore: 0.14,
    rocAuc: 0.72,
    pod: 0.65,
    far: 0.40,
    firstFlashHitRate: 0.50,
    firstFlashFAR: 0.42,
    medianFirstFlashLeadMin: 12,
  },
];

export const MOCK_SKILL_REPORT: SkillReport = {
  reportAt: '2026-04-18T12:00:00Z',
  medianFirstFlashLeadMin: 27, // F3+F4 headline
  models: MODELS,
  eventLabel: 'Odisha Pre-monsoon Kalbaisakhi Event (2026-04-18)',
  eventStartAt: '2026-04-18T09:30:00Z',
  eventEndAt: '2026-04-18T12:30:00Z',
};
