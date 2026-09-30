/**
 * data/mock/cellGenerator.ts
 * Generates Cell objects for each replay frame.
 */
import {
  CELL_DEFS,
  ODISHA_DISTRICTS,
  SCENARIO_T0_UTC,
  STEP_MIN,
} from './scenario';
import type { CellDef } from './scenario';

import { seedFromString, gaussian } from './seedRandom';
import type {
  Cell,
  CellStatus,
  LightningTrend,
  ForecastPathPoint,
  CellTrackPoint,
  RiskLevel,
} from '../types';
import { addMinutes, parseISO } from 'date-fns';

function getFrameTime(frame: number): string {
  return addMinutes(parseISO(SCENARIO_T0_UTC), frame * STEP_MIN).toISOString().replace('.000', '');
}

function cellEnvelope(cell: CellDef, frame: number): number {
  const relFrame = frame - cell.startFrame;
  const totalFrames = cell.endFrame - cell.startFrame;
  const peakRelFrame = cell.peakFrame - cell.startFrame;
  if (relFrame <= peakRelFrame) {
    return relFrame / Math.max(peakRelFrame, 1);
  }
  return 1 - (relFrame - peakRelFrame) / Math.max(totalFrames - peakRelFrame, 1);
}

function getCellStatus(cell: CellDef, frame: number): CellStatus {
  const env = cellEnvelope(cell, frame);
  const relFrame = frame - cell.startFrame;
  const peakRelFrame = cell.peakFrame - cell.startFrame;
  if (relFrame < 2) return 'initiating';
  if (relFrame <= peakRelFrame && env < 0.8) return 'growing';
  if (env >= 0.8) return 'mature';
  return 'decaying';
}

function getLightningTrend(cell: CellDef, frame: number): LightningTrend {
  const env = cellEnvelope(cell, frame);
  const peakRelFrame = cell.peakFrame - cell.startFrame;
  const relFrame = frame - cell.startFrame;
  if (relFrame < peakRelFrame - 2 && env < 0.6) return 'rapidly_increasing';
  if (relFrame < peakRelFrame) return 'increasing';
  if (relFrame === peakRelFrame) return 'steady';
  return 'decreasing';
}

function getFlashRate(cell: CellDef, frame: number, rng: () => number): number {
  const env = cellEnvelope(cell, frame);
  const base = cell.initialFlashRate + (cell.peakFlashRate - cell.initialFlashRate) * env;
  return Math.max(0, base + gaussian(rng, 0, 0.5));
}

function getRisk(dbz: number, prob?: number): RiskLevel {
  if (prob !== undefined) {
    if (prob > 0.6) return 'high';
    if (prob > 0.3) return 'moderate';
    return 'low';
  }
  if (dbz >= 50) return 'high';
  if (dbz >= 40) return 'moderate';
  return 'low';
}

function distanceKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

export function generateCells(frame: number): Cell[] {
  const rng = seedFromString(`cells-frame-${frame}`);
  const cells: Cell[] = [];

  for (const def of CELL_DEFS) {
    if (frame < def.startFrame || frame > def.endFrame) continue;

    const lat = def.startLat + def.dLatPerFrame * (frame - def.startFrame);
    const lon = def.startLon + def.dLonPerFrame * (frame - def.startFrame);
    const env = cellEnvelope(def, frame);
    const maxDbz = def.peakDbz * Math.max(0, env);
    const flashRate = getFlashRate(def, frame, rng);
    const lightningJumpFlag = def.lightningJumpFrame !== null && frame >= def.lightningJumpFrame;
    const lightningJumpAt = lightningJumpFlag && def.lightningJumpFrame !== null
      ? getFrameTime(def.lightningJumpFrame)
      : null;

    // Build historical track
    const track: CellTrackPoint[] = [];
    for (let f = Math.max(def.startFrame, frame - 6); f <= frame; f++) {
      const tLat = def.startLat + def.dLatPerFrame * (f - def.startFrame);
      const tLon = def.startLon + def.dLonPerFrame * (f - def.startFrame);
      const tEnv = cellEnvelope(def, f);
      const tRng = seedFromString(`tr-${def.id}-${f}`);
      track.push({
        time: getFrameTime(f),
        lat: tLat,
        lon: tLon,
        maxDbz: def.peakDbz * Math.max(0, tEnv),
        flashRate: getFlashRate(def, f, tRng),
      });
    }

    // Build forecast path
    const forecastPath: ForecastPathPoint[] = [];
    for (let lead = 10; lead <= 120; lead += 10) {
      const leadFrames = lead / 10;
      const futFrame = frame + leadFrames;
      const futEnv = futFrame <= def.endFrame ? cellEnvelope(def, futFrame) : 0;
      const futLat = def.startLat + def.dLatPerFrame * (futFrame - def.startFrame);
      const futLon = def.startLon + def.dLonPerFrame * (futFrame - def.startFrame);
      const futDbz = def.peakDbz * Math.max(0, futEnv);

      // Lightning prob only for +30 and +60
      let prob: number | undefined;
      if (lead <= 60) {
        prob = Math.min(1, futEnv * 0.85 + gaussian(rng, 0, 0.05));
      }

      forecastPath.push({
        leadMin: lead,
        lat: futLat,
        lon: futLon,
        expectedDbz: futDbz,
        risk: getRisk(futDbz, prob),
        riskBasis: lead <= 60 ? 'lightning_prob' : 'reflectivity_based',
      });
    }

    // Uncertainty cone
    const uncertaintyCone = forecastPath.map((p) => ({
      leadMin: p.leadMin,
      radiusKm: 5 + p.leadMin * 0.3 + (1 - env) * 10,
    }));

    // Affected districts
    const affectedDistricts = ODISHA_DISTRICTS
      .map((d) => {
        const distKm = distanceKm(lat, lon, d.lat, d.lon);
        const etaMin = Math.round(distKm / def.speedKmh * 60);
        return { districtId: d.id, name: d.name, etaMin, distKm };
      })
      .filter((d) => d.distKm < 150 && d.etaMin < 120)
      .sort((a, b) => a.etaMin - b.etaMin)
      .slice(0, 5)
      .map(({ districtId, name, etaMin }) => ({ districtId, name, etaMin }));

    // Time series for charts
    const flashRateSeries = Array.from({ length: 13 }, (_, i) => {
      const f = Math.max(def.startFrame, frame - 6 + i - 6);
      const isForecast = f > frame;
      const tRng = seedFromString(`frs-${def.id}-${f}`);
      return {
        time: getFrameTime(f),
        rate: isForecast ? getFlashRate(def, f, tRng) : track[i - (frame - 6 - Math.max(def.startFrame, frame - 6))]?.flashRate ?? 0,
        isForecast,
      };
    });

    const dbzSeries = Array.from({ length: 13 }, (_, i) => {
      const f = frame - 6 + i;
      const isForecast = f > frame;
      const fEnv = f >= def.startFrame && f <= def.endFrame ? cellEnvelope(def, f) : 0;
      return {
        time: getFrameTime(f),
        dbz: def.peakDbz * Math.max(0, fEnv),
        isForecast,
      };
    });

    const dataQuality = frame >= 12 ? 0.65 : 0.92; // degraded after frame 12
    const confidence = Math.round((env * 0.7 + dataQuality * 0.3) * 100);

    cells.push({
      id: def.id,
      isNewCell: def.isNewCell,
      status: getCellStatus(def, frame),
      lat,
      lon,
      maxDbz,
      growthDbzPer10Min: frame < def.peakFrame
        ? (def.peakDbz * env - def.peakDbz * Math.max(0, cellEnvelope(def, frame - 1))) / 1
        : -(def.peakDbz * env - def.peakDbz * Math.max(0, cellEnvelope(def, frame - 1))),
      lightningTrend: getLightningTrend(def, frame),
      headingDeg: def.headingDeg,
      speedKmh: def.speedKmh,
      lightningJumpFlag,
      lightningJumpAt,
      track,
      forecastPath,
      uncertaintyCone,
      affectedDistricts,
      confidence,
      modelMode: frame >= 12 ? 'satellite_only' : 'full',
      dataQuality,
      flashRateSeries: flashRateSeries.map(({ time, rate }) => ({ time, rate })),
      dbzSeries,
    });
  }

  return cells;
}
