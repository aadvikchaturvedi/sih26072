/**
 * data/mock/seedRandom.ts
 * Deterministic seeded random number generator (Mulberry32).
 */
export function mulberry32(seed: number) {
  return function () {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Create a seeded RNG from a string seed */
export function seedFromString(s: string): () => number {
  let hash = 0x12345678;
  for (let i = 0; i < s.length; i++) {
    hash = Math.imul(hash ^ s.charCodeAt(i), 0x9e3779b9);
  }
  return mulberry32(hash);
}

/** Gaussian random using Box-Muller transform */
export function gaussian(rng: () => number, mean = 0, std = 1): number {
  const u1 = rng();
  const u2 = rng();
  const z = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
  return mean + z * std;
}
