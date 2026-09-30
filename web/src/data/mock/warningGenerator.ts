/**
 * data/mock/warningGenerator.ts
 * Generates district warnings based on draft rules and district forecast data.
 */
import { ODISHA_DISTRICTS, CELL_DEFS, STEP_MIN, SCENARIO_T0_UTC } from './scenario';
import type { DistrictForecast, Warning, WarningLevel } from '../types';
import { addMinutes, parseISO } from 'date-fns';
import { seedFromString, gaussian } from './seedRandom';

function getFrameTime(frame: number): string {
  return addMinutes(parseISO(SCENARIO_T0_UTC), frame * STEP_MIN).toISOString().replace('.000', '');
}

function distanceKm(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) * Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function cellEnvelope(startFrame: number, peakFrame: number, endFrame: number, frame: number): number {
  const relFrame = frame - startFrame;
  const totalFrames = endFrame - startFrame;
  const peakRelFrame = peakFrame - startFrame;
  if (relFrame <= peakRelFrame) return relFrame / Math.max(peakRelFrame, 1);
  return 1 - (relFrame - peakRelFrame) / Math.max(totalFrames - peakRelFrame, 1);
}

/**
 * Draft warning rules (next 60 min per district):
 * Green: lightning probability <30% and no cell >= 45 dBZ forecast
 * Yellow: probability 30-60% or a cell >= 40 dBZ forecast
 * Orange: probability >60% or a lightning jump in an approaching cell
 * Red: Orange plus forecast >= 50 dBZ core or a first-flash flag in a populated block
 */
function computeWarningLevel(
  prob30: number,
  prob60: number,
  maxDbz: number,
  hasLightningJump: boolean,
  hasFirstFlash: boolean,
): WarningLevel {
  const prob = Math.max(prob30, prob60);
  const hasOrangeCondition = prob > 0.6 || hasLightningJump;
  const hasRedCondition = hasOrangeCondition && (maxDbz >= 50 || hasFirstFlash);

  if (hasRedCondition) return 'red';
  if (hasOrangeCondition) return 'orange';
  if (prob >= 0.3 || maxDbz >= 40) return 'yellow';
  return 'green';
}

export function generateDistricts(frame: number): DistrictForecast[] {
  const rng = seedFromString(`districts-${frame}`);
  const results: DistrictForecast[] = [];

  for (const district of ODISHA_DISTRICTS) {
    let maxProb30 = 0;
    let maxProb60 = 0;
    let maxDbz = 0;
    let hasLightningJump = false;
    let hasFirstFlash = false;

    for (const cell of CELL_DEFS) {
      if (frame < cell.startFrame || frame > cell.endFrame) continue;

      const cellLat = cell.startLat + cell.dLatPerFrame * (frame - cell.startFrame);
      const cellLon = cell.startLon + cell.dLonPerFrame * (frame - cell.startFrame);
      const distKm = distanceKm(district.lat, district.lon, cellLat, cellLon);

      if (distKm > 200) continue;

      const env = Math.max(0, cellEnvelope(cell.startFrame, cell.peakFrame, cell.endFrame, frame));
      const cellDbz = cell.peakDbz * env;

      // Proximity-based probability
      const proximityFactor = Math.exp(-distKm / 80);

      // Future position at +30 min
      const fut30Lat = cell.startLat + cell.dLatPerFrame * (frame + 3 - cell.startFrame);
      const fut30Lon = cell.startLon + cell.dLonPerFrame * (frame + 3 - cell.startFrame);
      const dist30 = distanceKm(district.lat, district.lon, fut30Lat, fut30Lon);
      const fut30Env = Math.max(0, cellEnvelope(cell.startFrame, cell.peakFrame, cell.endFrame, frame + 3));

      const fut60Lat = cell.startLat + cell.dLatPerFrame * (frame + 6 - cell.startFrame);
      const fut60Lon = cell.startLon + cell.dLonPerFrame * (frame + 6 - cell.startFrame);
      const dist60 = distanceKm(district.lat, district.lon, fut60Lat, fut60Lon);
      const fut60Env = Math.max(0, cellEnvelope(cell.startFrame, cell.peakFrame, cell.endFrame, frame + 6));

      const p30 = Math.min(0.98, fut30Env * Math.exp(-dist30 / 70) + gaussian(rng, 0, 0.04));
      const p60 = Math.min(0.98, fut60Env * Math.exp(-dist60 / 80) + gaussian(rng, 0, 0.04));

      maxProb30 = Math.max(maxProb30, Math.max(0, p30));
      maxProb60 = Math.max(maxProb60, Math.max(0, p60));
      maxDbz = Math.max(maxDbz, cellDbz * proximityFactor);

      if (cell.lightningJumpFrame !== null && frame >= cell.lightningJumpFrame && distKm < 100) {
        hasLightningJump = true;
      }
      if (cell.lightningJumpFrame !== null && frame >= cell.lightningJumpFrame && distKm < 50) {
        hasFirstFlash = true;
      }
    }

    const warningLevel = computeWarningLevel(maxProb30, maxProb60, maxDbz, hasLightningJump, hasFirstFlash);

    results.push({
      districtId: district.id,
      name: district.name,
      lightningProb30: Math.round(maxProb30 * 100) / 100,
      lightningProb60: Math.round(maxProb60 * 100) / 100,
      maxDbz: Math.round(maxDbz),
      areaFractionAboveThreshold: Math.min(1, maxDbz / 50 * 0.8),
      warningLevel,
    });
  }

  return results;
}

/** Previously generated warnings cache to implement change-only emission */
const warningStateCache: Map<string, WarningLevel> = new Map();
const warningIdCounter = { n: 100 };

export function generateWarnings(frame: number, districts: DistrictForecast[]): Warning[] {
  const warnings: Warning[] = [];
  const frameTime = getFrameTime(frame);
  const validUntil = addMinutes(parseISO(frameTime), 60).toISOString().replace('.000', '');

  for (const district of districts) {
    const previousLevel = warningStateCache.get(district.districtId) ?? null;
    const isLevelChange = previousLevel !== district.warningLevel;

    if (!isLevelChange && previousLevel !== null) continue; // suppress if unchanged

    warningStateCache.set(district.districtId, district.warningLevel);

    if (district.warningLevel === 'green' && previousLevel === null) continue; // skip initial green

    // Find the primary cell causing this warning
    const districtDef = ODISHA_DISTRICTS.find((d) => d.id === district.districtId);
    const causeCell = districtDef ? CELL_DEFS.find((c) => {
      if (frame < c.startFrame || frame > c.endFrame) return false;
      const lat = c.startLat + c.dLatPerFrame * (frame - c.startFrame);
      const lon = c.startLon + c.dLonPerFrame * (frame - c.startFrame);
      const d = distanceKm(districtDef.lat, districtDef.lon, lat, lon);
      return d < 150;
    }) : undefined;


    const ruleText = (() => {
      switch (district.warningLevel) {
        case 'red': return 'prob>60% + dBZ≥50 or first-flash in populated block';
        case 'orange': return causeCell?.lightningJumpFrame !== null ? 'lightning jump in approaching cell' : 'prob>60%';
        case 'yellow': return district.lightningProb30 >= 0.3 ? 'prob30>=30%' : 'maxDbz>=40';
        default: return 'no significant threat';
      }
    })();

    warnings.push({
      id: `W${warningIdCounter.n++}`,
      level: district.warningLevel,
      districtId: district.districtId,
      districtName: district.name,
      causeText: causeCell
        ? `Cell ${causeCell.id} moving SE at ${causeCell.speedKmh} km/h${causeCell.lightningJumpFrame !== null && frame >= causeCell.lightningJumpFrame ? `, lightning jump ${getFrameTime(causeCell.lightningJumpFrame).slice(11, 16)} UTC` : ''}`
        : `District-level lightning probability ${Math.round(district.lightningProb60 * 100)}%`,
      causeCell: causeCell?.id ?? null,
      ruleTriggered: ruleText,
      triggerValue: Math.max(district.lightningProb30, district.lightningProb60),
      triggerThreshold: district.warningLevel === 'yellow' ? 0.3 : 0.6,
      issuedAt: frameTime,
      validUntil,
      issuedBy: 'system',
      previousLevel,
      isLevelChange,
    });
  }

  return warnings;
};

export { computeWarningLevel };
