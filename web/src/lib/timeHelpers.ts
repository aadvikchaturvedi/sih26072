/**
 * lib/timeHelpers.ts
 * Time formatting utilities. All internal data is UTC; display in IST by default.
 */
import { format, parseISO, addMinutes } from 'date-fns';
import { addMinutes as dfnAddMinutes } from 'date-fns';

export const IST_OFFSET_MIN = 330; // UTC+5:30

export function toIST(utcIso: string): Date {
  const utc = parseISO(utcIso);
  return dfnAddMinutes(utc, IST_OFFSET_MIN);
}

export function formatTime(utcIso: string, tz: 'UTC' | 'IST'): string {
  if (tz === 'IST') {
    return format(toIST(utcIso), 'HH:mm') + ' IST';
  }
  return format(parseISO(utcIso), 'HH:mm') + ' UTC';
}

export function formatDateTime(utcIso: string, tz: 'UTC' | 'IST'): string {
  if (tz === 'IST') {
    return format(toIST(utcIso), 'dd MMM HH:mm') + ' IST';
  }
  return format(parseISO(utcIso), 'dd MMM HH:mm') + ' UTC';
}

export function formatFullDateTime(utcIso: string, tz: 'UTC' | 'IST'): string {
  if (tz === 'IST') {
    return format(toIST(utcIso), 'yyyy-MM-dd HH:mm:ss') + ' IST';
  }
  return format(parseISO(utcIso), 'yyyy-MM-dd HH:mm:ss') + ' UTC';
}

export function frameToUtcIso(baseT0Utc: string, frame: number, stepMin = 10): string {
  return addMinutes(parseISO(baseT0Utc), frame * stepMin)
    .toISOString()
    .replace('.000', '');
}

export function nowUtcIso(): string {
  return new Date().toISOString().replace('.000', '');
}

export function durationHuman(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  if (m < 60) return s > 0 ? `${m}m ${s}s` : `${m}m`;
  const h = Math.floor(m / 60);
  const rem = m % 60;
  return rem > 0 ? `${h}h ${rem}m` : `${h}h`;
}

export function secondsAgo(utcIso: string): number {
  return Math.round((Date.now() - parseISO(utcIso).getTime()) / 1000);
}
