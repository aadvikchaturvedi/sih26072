/**
 * data/mock/radarGenerator.ts
 * Generates per-frame radar reflectivity images as canvas data URLs.
 * Gaussian blob composites with seeded noise, georeferenced to the Odisha bbox.
 */
import { DOMAIN_BBOX, CELL_DEFS, NUM_FRAMES, RADAR_DEGRADED_FRAME } from './scenario';
import type { CellDef } from './scenario';

import { seedFromString, gaussian } from './seedRandom';

const CANVAS_W = 256;
const CANVAS_H = 256;

/** dBZ to RGBA colormap (standard meteorological reflectivity palette) */
function dbzToRGBA(dbz: number): [number, number, number, number] {
  if (dbz < 15) return [0, 0, 0, 0];           // transparent
  if (dbz < 20) return [180, 240, 255, 180];   // light blue
  if (dbz < 25) return [100, 200, 255, 200];   // cyan
  if (dbz < 30) return [0, 255, 100, 210];     // green
  if (dbz < 35) return [80, 220, 0, 220];      // yellow-green
  if (dbz < 40) return [255, 230, 0, 230];     // yellow
  if (dbz < 45) return [255, 140, 0, 240];     // orange
  if (dbz < 50) return [255, 40, 40, 245];     // red
  if (dbz < 55) return [200, 0, 200, 250];     // magenta
  return [140, 0, 140, 255];                    // purple (55+)
}

/** Evaluate cell intensity at a grid point */
function cellDbzAtPoint(
  cell: CellDef,
  frame: number,
  pixLat: number,
  pixLon: number,
  rng: () => number
): number {
  if (frame < cell.startFrame || frame > cell.endFrame) return 0;

  const centerLat = cell.startLat + cell.dLatPerFrame * (frame - cell.startFrame);
  const centerLon = cell.startLon + cell.dLonPerFrame * (frame - cell.startFrame);

  // Growth/decay envelope
  const relFrame = frame - cell.startFrame;
  const totalFrames = cell.endFrame - cell.startFrame;
  const peakRelFrame = cell.peakFrame - cell.startFrame;
  let envelope: number;
  if (relFrame <= peakRelFrame) {
    envelope = relFrame / Math.max(peakRelFrame, 1);
  } else {
    envelope = 1 - (relFrame - peakRelFrame) / Math.max(totalFrames - peakRelFrame, 1);
  }
  envelope = Math.max(0, envelope);

  const peakDbz = cell.peakDbz * envelope;

  // Gaussian spread (in degrees, ~50km radius at peak)
  const spreadLat = 0.35 * (0.5 + 0.5 * envelope);
  const spreadLon = 0.45 * (0.5 + 0.5 * envelope);

  const dLat = pixLat - centerLat;
  const dLon = pixLon - centerLon;
  const gaussVal = Math.exp(-0.5 * ((dLat / spreadLat) ** 2 + (dLon / spreadLon) ** 2));

  // Add noise (seeded)
  const noise = gaussian(rng, 0, 2.5);
  return Math.max(0, peakDbz * gaussVal + noise);
}

/** Generate a single frame's radar image as a canvas data URL */
function generateFrameCanvas(frame: number, isForecast: boolean): string {
  const canvas = document.createElement('canvas');
  canvas.width = CANVAS_W;
  canvas.height = CANVAS_H;
  const ctx = canvas.getContext('2d')!;

  const imgData = ctx.createImageData(CANVAS_W, CANVAS_H);
  const data = imgData.data;

  const rng = seedFromString(`radar-frame-${frame}`);

  const latRange = DOMAIN_BBOX.north - DOMAIN_BBOX.south;
  const lonRange = DOMAIN_BBOX.east - DOMAIN_BBOX.west;

  for (let py = 0; py < CANVAS_H; py++) {
    for (let px = 0; px < CANVAS_W; px++) {
      // Row 0 = northern edge (matching ML output convention)
      const lat = DOMAIN_BBOX.north - (py / CANVAS_H) * latRange;
      const lon = DOMAIN_BBOX.west + (px / CANVAS_W) * lonRange;

      let totalDbz = 0;
      for (const cell of CELL_DEFS) {
        totalDbz += cellDbzAtPoint(cell, frame, lat, lon, rng);
      }
      totalDbz = Math.min(65, totalDbz);

      const [r, g, b, a] = dbzToRGBA(totalDbz);
      const idx = (py * CANVAS_W + px) * 4;
      data[idx] = r;
      data[idx + 1] = g;
      data[idx + 2] = b;
      data[idx + 3] = totalDbz < 15 ? 0 : a;
    }
  }

  ctx.putImageData(imgData, 0, 0);

  // Add forecast hatching overlay
  if (isForecast) {
    ctx.save();
    ctx.globalAlpha = 0.12;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1;
    for (let i = -CANVAS_W; i < CANVAS_W * 2; i += 8) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i + CANVAS_H, CANVAS_H);
      ctx.stroke();
    }
    ctx.restore();
  }

  return canvas.toDataURL('image/png');
}

/** Cache of generated frames */
const frameCache: Map<string, string> = new Map();

export function getRadarFrameUrl(frame: number, isForecast: boolean): string {
  const key = `${frame}-${isForecast}`;
  if (!frameCache.has(key)) {
    frameCache.set(key, generateFrameCanvas(frame, isForecast));
  }
  return frameCache.get(key)!;
}

/** Generate lightning probability raster for +30 or +60 min */
export function getLightningProbUrl(frame: number, lead: 30 | 60): string {
  const key = `lp-${frame}-${lead}`;
  if (frameCache.has(key)) return frameCache.get(key)!;

  const canvas = document.createElement('canvas');
  canvas.width = CANVAS_W;
  canvas.height = CANVAS_H;
  const ctx = canvas.getContext('2d')!;
  const imgData = ctx.createImageData(CANVAS_W, CANVAS_H);
  const data = imgData.data;
  const rng = seedFromString(`lp-${frame}-${lead}`);

  const latRange = DOMAIN_BBOX.north - DOMAIN_BBOX.south;
  const lonRange = DOMAIN_BBOX.east - DOMAIN_BBOX.west;

  for (let py = 0; py < CANVAS_H; py++) {
    for (let px = 0; px < CANVAS_W; px++) {
      const lat = DOMAIN_BBOX.north - (py / CANVAS_H) * latRange;
      const lon = DOMAIN_BBOX.west + (px / CANVAS_W) * lonRange;

      let maxProb = 0;
      for (const cell of CELL_DEFS) {
        // Extrapolate cell position at lead time
        const leadFrames = lead / 10;
        const futFrame = frame + leadFrames;
        if (futFrame < cell.startFrame || futFrame > cell.endFrame) continue;

        const centerLat = cell.startLat + cell.dLatPerFrame * (futFrame - cell.startFrame);
        const centerLon = cell.startLon + cell.dLonPerFrame * (futFrame - cell.startFrame);

        const envelope = Math.min(1, (futFrame - cell.startFrame) / Math.max(cell.peakFrame - cell.startFrame, 1));
        const spread = 0.3 * (0.6 + 0.4 * envelope);

        const dLat = lat - centerLat;
        const dLon = lon - centerLon;
        const prob = envelope * Math.exp(-0.5 * ((dLat / spread) ** 2 + (dLon / spread) ** 2));
        maxProb = Math.max(maxProb, prob);
      }

      const noise = Math.max(0, gaussian(rng, 0, 0.04));
      const p = Math.min(1, maxProb + noise);
      const idx = (py * CANVAS_W + px) * 4;
      // Heatmap: transparent=low, yellow=mid, red=high
      data[idx] = Math.round(p * 255);
      data[idx + 1] = Math.round((1 - p) * 180);
      data[idx + 2] = 0;
      data[idx + 3] = p > 0.05 ? Math.round(p * 200) : 0;
    }
  }

  ctx.putImageData(imgData, 0, 0);
  const url = canvas.toDataURL('image/png');
  frameCache.set(key, url);
  return url;
}

/** Generate first-flash risk mask */
export function getFirstFlashUrl(frame: number): string {
  const key = `ff-${frame}`;
  if (frameCache.has(key)) return frameCache.get(key)!;

  const canvas = document.createElement('canvas');
  canvas.width = CANVAS_W;
  canvas.height = CANVAS_H;
  const ctx = canvas.getContext('2d')!;
  const imgData = ctx.createImageData(CANVAS_W, CANVAS_H);
  const data = imgData.data;

  const latRange = DOMAIN_BBOX.north - DOMAIN_BBOX.south;
  const lonRange = DOMAIN_BBOX.east - DOMAIN_BBOX.west;

  for (let py = 0; py < CANVAS_H; py++) {
    for (let px = 0; px < CANVAS_W; px++) {
      const lat = DOMAIN_BBOX.north - (py / CANVAS_H) * latRange;
      const lon = DOMAIN_BBOX.west + (px / CANVAS_W) * lonRange;
      let isFirstFlash = false;

      for (const cell of CELL_DEFS) {
        if (!cell.lightningJumpFrame || frame < cell.lightningJumpFrame) continue;
        const futFrame = frame + 3; // +30 min
        if (futFrame > cell.endFrame) continue;

        const centerLat = cell.startLat + cell.dLatPerFrame * (futFrame - cell.startFrame);
        const centerLon = cell.startLon + cell.dLonPerFrame * (futFrame - cell.startFrame);
        const dist = Math.sqrt((lat - centerLat) ** 2 + (lon - centerLon) ** 2);
        if (dist < 0.15) { isFirstFlash = true; break; }
      }

      const idx = (py * CANVAS_W + px) * 4;
      if (isFirstFlash) {
        data[idx] = 255; data[idx + 1] = 80; data[idx + 2] = 0; data[idx + 3] = 180;
      } else {
        data[idx + 3] = 0;
      }
    }
  }

  ctx.putImageData(imgData, 0, 0);
  const url = canvas.toDataURL('image/png');
  frameCache.set(key, url);
  return url;
}

/** All lead times for forecast (10..120 in steps of 10) */
export const FORECAST_LEADS = Array.from({ length: 12 }, (_, i) => (i + 1) * 10);

/** Check if frame is in degraded (satellite-only) mode */
export function isRadarDegraded(frame: number): boolean {
  return frame >= RADAR_DEGRADED_FRAME;
}
