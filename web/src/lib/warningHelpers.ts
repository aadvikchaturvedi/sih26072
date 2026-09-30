/**
 * lib/warningHelpers.ts
 * Utility functions for warning levels.
 */
import type { WarningLevel } from '@/data/types';

export function warningColor(level: WarningLevel): string {
  switch (level) {
    case 'red':    return 'var(--warn-red)';
    case 'orange': return 'var(--warn-orange)';
    case 'yellow': return 'var(--warn-yellow)';
    default:       return 'var(--warn-green)';
  }
}

export function warningFill(level: WarningLevel): string {
  switch (level) {
    case 'red':    return 'var(--warn-red-fill)';
    case 'orange': return 'var(--warn-orange-fill)';
    case 'yellow': return 'var(--warn-yellow-fill)';
    default:       return 'var(--warn-green-fill)';
  }
}

export function warningLabel(level: WarningLevel): string {
  switch (level) {
    case 'red':    return 'RED';
    case 'orange': return 'ORANGE';
    case 'yellow': return 'YELLOW';
    default:       return 'GREEN';
  }
}

export function warningLevelOrder(level: WarningLevel): number {
  switch (level) {
    case 'red':    return 4;
    case 'orange': return 3;
    case 'yellow': return 2;
    default:       return 1;
  }
}

export function warningIcon(level: WarningLevel): string {
  switch (level) {
    case 'red':    return 'octagon-x';
    case 'orange': return 'octagon-alert';
    case 'yellow': return 'triangle-alert';
    default:       return 'circle-check';
  }
}

export function warningBgClass(level: WarningLevel): string {
  switch (level) {
    case 'red':    return 'bg-warn-red';
    case 'orange': return 'bg-warn-orange';
    case 'yellow': return 'bg-warn-yellow';
    default:       return 'bg-warn-green';
  }
}

export function warningTextClass(level: WarningLevel): string {
  switch (level) {
    case 'red':    return 'warn-red';
    case 'orange': return 'warn-orange';
    case 'yellow': return 'warn-yellow';
    default:       return 'warn-green';
  }
}
