/**
 * data/hooks.ts
 * TanStack Query hooks for all server data.
 * These are the only place components should fetch data.
 */
import { useQuery } from '@tanstack/react-query';
import { getDataSourceSync } from './dataSourceFactory';
import { SCENARIO_T0_UTC, NUM_FRAMES, STEP_MIN } from './mock/scenario';
import { addMinutes, parseISO } from 'date-fns';

function frameToT0(frame: number): string {
  return addMinutes(parseISO(SCENARIO_T0_UTC), frame * STEP_MIN)
    .toISOString()
    .replace('.000', '');
}

const ds = () => getDataSourceSync();

export function useForecast(frame: number) {
  const t0 = frameToT0(Math.max(0, Math.min(NUM_FRAMES - 1, frame)));
  return useQuery({
    queryKey: ['forecast', t0],
    queryFn: () => ds().getForecast(t0),
    staleTime: 5 * 60 * 1000,
  });
}

export function useCells(frame: number) {
  const t0 = frameToT0(Math.max(0, Math.min(NUM_FRAMES - 1, frame)));
  return useQuery({
    queryKey: ['cells', t0],
    queryFn: () => ds().getCells(t0),
    staleTime: 1 * 60 * 1000,
  });
}

export function useLightningStrokes(frame: number) {
  const t0 = frameToT0(Math.max(0, Math.min(NUM_FRAMES - 1, frame)));
  return useQuery({
    queryKey: ['lightning', t0],
    queryFn: () => ds().getLightningStrokes(t0),
    staleTime: 1 * 60 * 1000,
  });
}

export function useDistricts(frame: number) {
  const t0 = frameToT0(Math.max(0, Math.min(NUM_FRAMES - 1, frame)));
  return useQuery({
    queryKey: ['districts', t0],
    queryFn: () => ds().getDistricts(t0),
    staleTime: 1 * 60 * 1000,
  });
}

export function useWarnings(frame: number) {
  const t0 = frameToT0(Math.max(0, Math.min(NUM_FRAMES - 1, frame)));
  return useQuery({
    queryKey: ['warnings', t0],
    queryFn: () => ds().getWarnings(t0),
    staleTime: 30 * 1000,
  });
}

export function useSkill() {
  return useQuery({
    queryKey: ['skill'],
    queryFn: () => ds().getSkill(),
    staleTime: 60 * 60 * 1000, // 1 hour
  });
}

export function useHealth(frame: number) {
  const t0 = frameToT0(Math.max(0, Math.min(NUM_FRAMES - 1, frame)));
  return useQuery({
    queryKey: ['health', t0],
    queryFn: () => ds().getHealth(t0),
    refetchInterval: 10 * 1000, // poll every 10s
    staleTime: 5 * 1000,
  });
}
