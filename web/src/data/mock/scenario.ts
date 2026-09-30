/**
 * data/mock/scenario.ts
 * Defines the Odisha nor'wester (Kalbaisakhi) event scenario.
 * 3-hour replay, 19 frames at 10-minute steps.
 *
 * Cells:
 *  - Cell #14 "MainStorm": Keonjhar/Angul belt → SE toward Bhubaneswar/Puri
 *  - Cell #15 "BalasoreCell": NEW CELL initiating near Balasore/Bhadrak
 *  - Cell #16 "MayurbhanjCell": Weak cell decaying over Mayurbhanj
 *
 * Degraded scenario: at frame 12 (t+110 min from start) radar goes missing,
 * system switches to SATELLITE-ONLY mode.
 */

// Odisha domain bbox
export const DOMAIN_BBOX = {
  west: 84.0,
  east: 87.5,
  south: 19.5,
  north: 22.5,
} as const;

// Event base time (UTC): a typical pre-monsoon afternoon
export const SCENARIO_T0_UTC = '2026-04-18T09:30:00Z'; // 15:00 IST

// 19 frames: frames 0-6 observed (t-60 to 0), frames 7-18 forecast (+10 to +120)
// Actually we have a 3-hour replay with 10-min steps = 18 steps + initial = 19 frames
// Frames 0..8 are "observed" (past 80 min + current), frames 9..18 forecast (+10..+100)
export const NUM_FRAMES = 19;
export const STEP_MIN = 10;

// Frame 12 is when radar goes missing (t0 + 110 min offset from replay start)
export const RADAR_DEGRADED_FRAME = 12;

/** Odisha districts used in the demo */
export const ODISHA_DISTRICTS = [
  { id: 'keonjhar',      name: 'Keonjhar',      lat: 21.63, lon: 85.58 },
  { id: 'angul',         name: 'Angul',          lat: 20.84, lon: 85.10 },
  { id: 'dhenkanal',     name: 'Dhenkanal',      lat: 20.66, lon: 85.60 },
  { id: 'cuttack',       name: 'Cuttack',        lat: 20.46, lon: 85.88 },
  { id: 'jajpur',        name: 'Jajpur',         lat: 20.84, lon: 86.33 },
  { id: 'kendrapara',    name: 'Kendrapara',     lat: 20.50, lon: 86.42 },
  { id: 'jagatsinghpur', name: 'Jagatsinghpur',  lat: 20.25, lon: 86.17 },
  { id: 'khordha',       name: 'Khordha',        lat: 20.18, lon: 85.61 },
  { id: 'puri',          name: 'Puri',           lat: 19.81, lon: 85.83 },
  { id: 'nayagarh',      name: 'Nayagarh',       lat: 20.13, lon: 85.09 },
  { id: 'balasore',      name: 'Balasore',       lat: 21.49, lon: 86.93 },
  { id: 'bhadrak',       name: 'Bhadrak',        lat: 21.06, lon: 86.50 },
  { id: 'mayurbhanj',    name: 'Mayurbhanj',     lat: 21.94, lon: 86.74 },
  { id: 'ganjam',        name: 'Ganjam',         lat: 19.38, lon: 84.79 },
] as const;

export type DistrictId = typeof ODISHA_DISTRICTS[number]['id'];

/** Cell definitions for the scenario */
export interface CellDef {
  id: string;
  label: string;
  isNewCell: boolean;
  /** Frame at which the cell appears (0-indexed) */
  startFrame: number;
  /** Frame at which the cell disappears */
  endFrame: number;
  /** Starting lat/lon */
  startLat: number;
  startLon: number;
  /** Movement per frame: lat change and lon change */
  dLatPerFrame: number;
  dLonPerFrame: number;
  /** Max dBZ at peak frame */
  peakDbz: number;
  /** Frame of peak intensity */
  peakFrame: number;
  /** Initial flash rate (flashes/min) */
  initialFlashRate: number;
  /** Peak flash rate */
  peakFlashRate: number;
  /** Heading in degrees */
  headingDeg: number;
  /** Speed km/h */
  speedKmh: number;
  /** Frame of lightning jump */
  lightningJumpFrame: number | null;
}

export const CELL_DEFS: CellDef[] = [
  {
    id: 'C14',
    label: 'Main Storm (C14)',
    isNewCell: false,
    startFrame: 0,
    endFrame: 18,
    startLat: 21.10,
    startLon: 85.30,
    dLatPerFrame: -0.065,  // moving south
    dLonPerFrame: +0.075,  // moving east
    peakDbz: 58,
    peakFrame: 10,
    initialFlashRate: 2,
    peakFlashRate: 18,
    headingDeg: 142,
    speedKmh: 41,
    lightningJumpFrame: 7,
  },
  {
    id: 'C15',
    label: 'Balasore Cell (C15)',
    isNewCell: true, // NEW CELL - demonstrates nowcast-only prediction
    startFrame: 5,  // initiates mid-replay
    endFrame: 18,
    startLat: 21.45,
    startLon: 86.80,
    dLatPerFrame: -0.04,
    dLonPerFrame: +0.02,
    peakDbz: 48,
    peakFrame: 14,
    initialFlashRate: 0,
    peakFlashRate: 8,
    headingDeg: 155,
    speedKmh: 28,
    lightningJumpFrame: null,
  },
  {
    id: 'C16',
    label: 'Mayurbhanj Cell (C16)',
    isNewCell: false,
    startFrame: 0,
    endFrame: 12,  // decays and disappears
    startLat: 21.80,
    startLon: 86.70,
    dLatPerFrame: -0.02,
    dLonPerFrame: -0.01,
    peakDbz: 32,
    peakFrame: 3,
    initialFlashRate: 1,
    peakFlashRate: 3,
    headingDeg: 200,
    speedKmh: 15,
    lightningJumpFrame: null,
  },
];
