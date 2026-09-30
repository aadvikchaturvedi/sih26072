/**
 * components/WarningBadge.tsx
 * Displays a warning level badge with icon + text label.
 * Never relies on color alone - always icon + text.
 */
import React from 'react';
import { CircleCheck, TriangleAlert, OctagonAlert, Siren } from 'lucide-react';
import type { WarningLevel } from '@/data/types';
import { warningColor, warningLabel } from '@/lib/warningHelpers';

interface WarningBadgeProps {
  level: WarningLevel;
  size?: 'sm' | 'md' | 'lg';
  showText?: boolean;
  className?: string;
}

const icons: Record<WarningLevel, React.FC<{ size?: number }>> = {
  green: ({ size = 14 }) => <CircleCheck size={size} />,
  yellow: ({ size = 14 }) => <TriangleAlert size={size} />,
  orange: ({ size = 14 }) => <OctagonAlert size={size} />,
  red: ({ size = 14 }) => <Siren size={size} />,
};

const sizes = {
  sm: { icon: 11, text: 'text-[9px]', pad: 'px-1.5 py-0.5', gap: 'gap-1' },
  md: { icon: 13, text: 'text-[11px]', pad: 'px-2 py-1', gap: 'gap-1.5' },
  lg: { icon: 16, text: 'text-[13px]', pad: 'px-3 py-1.5', gap: 'gap-2' },
};

export function WarningBadge({ level, size = 'md', showText = true, className = '' }: WarningBadgeProps) {
  const color = warningColor(level);
  const Icon = icons[level];
  const { icon: iconSize, text, pad, gap } = sizes[size];

  return (
    <span
      role="img"
      aria-label={`${warningLabel(level)} warning level`}
      className={`inline-flex items-center font-semibold tracking-wide uppercase border rounded ${pad} ${gap} ${text} ${className}`}
      style={{
        color,
        borderColor: color,
        background: color.replace(')', ', 0.12)').replace('var(', 'color-mix(in srgb, ').replace(')', ' 12%, transparent)'),
      }}
    >
      <Icon size={iconSize} />
      {showText && <span>{warningLabel(level)}</span>}
    </span>
  );
}
