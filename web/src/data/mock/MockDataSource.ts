/**
 * data/mock/MockDataSource.ts
 * Full DataSource implementation using the procedural scenario generator.
 * Simulates 150-400ms network latency, WebSocket-style ticks.
 */
import type { DataSource } from '../DataSource';
import type {
  ForecastResponse,
  Cell,
  Warning,
  SkillReport,
  HealthStatus,
  LiveTick,
  LightningStroke,
  DistrictForecast,
} from '../types';
import {
  SCENARIO_T0_UTC,
  STEP_MIN,
  NUM_FRAMES,
  RADAR_DEGRADED_FRAME,
  ODISHA_DISTRICTS,
} from './scenario';
import {
  getRadarFrameUrl,
  getLightningProbUrl,
  getFirstFlashUrl,
  FORECAST_LEADS,
} from './radarGenerator';
import { generateCells } from './cellGenerator';
import { generateDistricts, generateWarnings } from './warningGenerator';
import { MOCK_SKILL_REPORT } from './skillData';
import { addMinutes, parseISO } from 'date-fns';
import { seedFromString, gaussian } from './seedRandom';
import type { InputHealth } from '../types';

/** Current replay frame (shared mutable state for the mock) */
let currentFrame = 8; // start at frame 8 = "now"
const frameListeners: Set<(frame: number) => void> = new Set();

export function setMockFrame(frame: number) {
  currentFrame = Math.max(0, Math.min(NUM_FRAMES - 1, frame));
  frameListeners.forEach((fn) => fn(currentFrame));
}

export function getMockFrame() {
  return currentFrame;
}

function frameToT0(frame: number): string {
  return addMinutes(parseISO(SCENARIO_T0_UTC), frame * STEP_MIN)
    .toISOString()
    .replace('.000', '');
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function mockLatency(): number {
  const rng = seedFromString(`latency-${Date.now()}`);
  return 150 + Math.floor(gaussian(rng, 0, 80) + 200);
}

function getFrameFromT0(t0: string): number {
  const base = parseISO(SCENARIO_T0_UTC).getTime();
  const t = parseISO(t0).getTime();
  const diffMin = (t - base) / 60000;
  return Math.round(diffMin / STEP_MIN);
}

function buildHealth(frame: number): HealthStatus {
  const isDegraded = frame >= RADAR_DEGRADED_FRAME;
  const t0 = frameToT0(frame);
  const nextRunAt = addMinutes(parseISO(t0), STEP_MIN).toISOString().replace('.000', '');

  const inputs: InputHealth[] = [
    {
      name: 'radar',
      status: isDegraded ? 'missing' : 'live',
      lastReceived: isDegraded ? frameToT0(RADAR_DEGRADED_FRAME - 1) : t0,
      staleAgeSeconds: isDegraded ? (frame - RADAR_DEGRADED_FRAME) * STEP_MIN * 60 : null,
    },
    { name: 'satellite', status: 'live', lastReceived: t0, staleAgeSeconds: null },
    {
      name: 'lightning',
      status: frame > 15 ? 'stale' : 'live',
      lastReceived: frame > 15 ? frameToT0(15) : t0,
      staleAgeSeconds: frame > 15 ? (frame - 15) * STEP_MIN * 60 : null,
    },
    { name: 'nwp', status: 'live', lastReceived: frameToT0(0), staleAgeSeconds: null },
  ];

  return {
    lastRunAt: t0,
    nextRunAt,
    mode: 'replay',
    modelMode: isDegraded ? 'satellite_only' : 'full',
    modelName: 'NowcastNet-IMD',
    modelVersion: '1.0.0-beta',
    missingChannels: isDegraded ? ['radar_z'] : [],
    inputs,
    hasEnsemble: !isDegraded,
    inferenceMsLast: isDegraded ? 380 : 210,
  };
}

/** All warnings accumulated across frames (newest first) */
const allWarnings: Warning[] = [];
let lastWarningFrame = -1;

function ensureWarningsUpToFrame(frame: number) {
  for (let f = Math.max(0, lastWarningFrame + 1); f <= frame; f++) {
    const districts = generateDistricts(f);
    const newWarnings = generateWarnings(f, districts);
    allWarnings.unshift(...newWarnings);
  }
  lastWarningFrame = Math.max(lastWarningFrame, frame);
}

/** Generate lightning strokes for a given frame */
function generateLightningStrokes(frame: number): LightningStroke[] {
  const strokes: LightningStroke[] = [];
  const rng = seedFromString(`lightning-${frame}`);
  const t0 = parseISO(frameToT0(frame));

  const cellData = generateCells(frame);

  for (const cell of cellData) {
    const count = Math.floor(cell.flashRateSeries.slice(-1)[0]?.rate ?? 0) * 2;
    for (let i = 0; i < count; i++) {
      const offsetMinutes = rng() * 30;
      const lat = cell.lat + gaussian(rng, 0, 0.15);
      const lon = cell.lon + gaussian(rng, 0, 0.15);
      strokes.push({
        id: `L${frame}-${cell.id}-${i}`,
        time: new Date(t0.getTime() - offsetMinutes * 60000).toISOString(),
        lat,
        lon,
        polarity: rng() > 0.7 ? 'positive' : 'negative',
        peakCurrentKA: 10 + Math.floor(rng() * 80),
        cellId: cell.id,
      });
    }
  }

  return strokes;
}

export class MockDataSource implements DataSource {
  getObservedRadarUrl(t: string): string {
    const frame = getFrameFromT0(t);
    const clampedFrame = Math.max(0, Math.min(NUM_FRAMES - 1, frame));
    return getRadarFrameUrl(clampedFrame, false);
  }

  async getForecast(t0: string): Promise<ForecastResponse> {
    await delay(mockLatency());
    const frame = getFrameFromT0(t0);
    const clampedFrame = Math.max(0, Math.min(NUM_FRAMES - 1, frame));
    const isDegraded = clampedFrame >= RADAR_DEGRADED_FRAME;

    const reflectivityUrlByLead: Record<number, string> = {};
    for (const lead of FORECAST_LEADS) {
      const futFrame = clampedFrame + lead / STEP_MIN;
      const futClamped = Math.max(0, Math.min(NUM_FRAMES - 1, Math.round(futFrame)));
      reflectivityUrlByLead[lead] = isDegraded
        ? '' // no radar in degraded mode
        : getRadarFrameUrl(futClamped, true);
    }

    return {
      reflectivityUrlByLead,
      lightningProb30Url: getLightningProbUrl(clampedFrame, 30),
      lightningProb60Url: getLightningProbUrl(clampedFrame, 60),
      firstFlashUrl: getFirstFlashUrl(clampedFrame),
      attrs: {
        model_name: 'NowcastNet-IMD',
        model_version: '1.0.0-beta',
        t0: frameToT0(clampedFrame),
        mode: isDegraded ? 'satellite_only' : 'full',
        missing_channels: isDegraded ? ['radar_z'] : [],
        inference_ms: isDegraded ? 380 : 210,
        output_contract_version: '1.0',
        first_flash_threshold: 0.5,
      },
    };
  }

  async getCells(t0: string): Promise<Cell[]> {
    await delay(mockLatency());
    const frame = getFrameFromT0(t0);
    return generateCells(Math.max(0, Math.min(NUM_FRAMES - 1, frame)));
  }

  async getLightningStrokes(t0: string): Promise<LightningStroke[]> {
    await delay(mockLatency());
    const frame = getFrameFromT0(t0);
    return generateLightningStrokes(Math.max(0, Math.min(NUM_FRAMES - 1, frame)));
  }

  async getDistricts(t0: string): Promise<DistrictForecast[]> {
    await delay(mockLatency());
    const frame = getFrameFromT0(t0);
    return generateDistricts(Math.max(0, Math.min(NUM_FRAMES - 1, frame)));
  }

  async getWarnings(t0: string): Promise<Warning[]> {
    await delay(mockLatency());
    const frame = getFrameFromT0(t0);
    const clampedFrame = Math.max(0, Math.min(NUM_FRAMES - 1, frame));
    ensureWarningsUpToFrame(clampedFrame);
    return [...allWarnings];
  }

  async getSkill(): Promise<SkillReport> {
    await delay(mockLatency());
    return MOCK_SKILL_REPORT;
  }

  async getHealth(t0?: string): Promise<HealthStatus> {
    await delay(50);
    const frame = t0 ? getFrameFromT0(t0) : currentFrame;
    return buildHealth(frame);
  }

  async exportCap(warningId: string): Promise<{ xml: string; json: object }> {
    await delay(mockLatency());
    ensureWarningsUpToFrame(currentFrame);
    const warning = allWarnings.find((w) => w.id === warningId);

    if (!warning) throw new Error(`Warning ${warningId} not found`);

    const district = ODISHA_DISTRICTS.find((d) => d.id === warning.districtId);
    const identifier = `IMD-SIH26072-${warning.id}`;
    const sent = warning.issuedAt;

    const xml = `<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
  <identifier>${identifier}</identifier>
  <sender>nowcast@imd.gov.in</sender>
  <sent>${sent}</sent>
  <status>Test</status>
  <msgType>Alert</msgType>
  <scope>Public</scope>
  <note>WARNING THRESHOLDS ARE DRAFT PLACEHOLDERS — NOT IMD OPERATIONAL CRITERIA</note>
  <info>
    <language>en-IN</language>
    <category>Met</category>
    <event>Thunderstorm/Lightning Nowcast</event>
    <responseType>Prepare</responseType>
    <urgency>Expected</urgency>
    <severity>${warning.level === 'red' ? 'Extreme' : warning.level === 'orange' ? 'Severe' : warning.level === 'yellow' ? 'Moderate' : 'Minor'}</severity>
    <certainty>Likely</certainty>
    <onset>${sent}</onset>
    <expires>${warning.validUntil}</expires>
    <senderName>IMD SIH26072 Nowcast System (Prototype)</senderName>
    <headline>${warning.level.toUpperCase()} THUNDERSTORM WARNING — ${warning.districtName}</headline>
    <description>${warning.causeText}</description>
    <instruction>Seek shelter. Avoid open areas. Keep away from tall objects.</instruction>
    <area>
      <areaDesc>${warning.districtName} District, Odisha</areaDesc>
      <polygon>${district ? `${district.lat - 0.5},${district.lon - 0.5} ${district.lat + 0.5},${district.lon - 0.5} ${district.lat + 0.5},${district.lon + 0.5} ${district.lat - 0.5},${district.lon + 0.5} ${district.lat - 0.5},${district.lon - 0.5}` : ''}</polygon>
    </area>
  </info>
</alert>`;

    const json = {
      identifier,
      sender: 'nowcast@imd.gov.in',
      sent,
      status: 'Test',
      msgType: 'Alert',
      scope: 'Public',
      note: 'WARNING THRESHOLDS ARE DRAFT PLACEHOLDERS — NOT IMD OPERATIONAL CRITERIA',
      info: {
        language: 'en-IN',
        category: 'Met',
        event: 'Thunderstorm/Lightning Nowcast',
        urgency: 'Expected',
        severity: warning.level,
        certainty: 'Likely',
        onset: sent,
        expires: warning.validUntil,
        headline: `${warning.level.toUpperCase()} THUNDERSTORM WARNING — ${warning.districtName}`,
        description: warning.causeText,
        area: {
          areaDesc: `${warning.districtName} District, Odisha`,
          district: district,
        },
        warningObject: warning,
      },
    };

    return { xml, json };
  }

  subscribe(onTick: (msg: LiveTick) => void): () => void {
    // Mock replay ticker — emits a tick every 10 simulated minutes
    // Actual interval driven by useReplayClock hook
    const id = setInterval(() => {
      const health = buildHealth(currentFrame);
      const districts = generateDistricts(currentFrame);
      const newWarnings = generateWarnings(currentFrame, districts);
      const updatedCells = generateCells(currentFrame);

      onTick({
        t0: frameToT0(currentFrame),
        newWarnings,
        updatedCells,
        health,
      });
    }, 5000); // poll every 5s in mock (real replay advances via ReplayControls)

    return () => clearInterval(id);
  }
}
