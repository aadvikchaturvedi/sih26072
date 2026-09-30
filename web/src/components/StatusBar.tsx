/**
 * components/StatusBar.tsx
 * 32px bottom status bar with input health chips.
 */
import React from 'react';
import { Wifi, WifiOff, AlertCircle, Satellite, Zap, Cloud, Clock, Info } from 'lucide-react';
import { useHealth } from '@/data/hooks';
import { durationHuman } from '@/lib/timeHelpers';
import { useUIStore } from '@/store/uiStore';
import type { InputHealth, InputStatus } from '@/data/types';

function StatusChip({ input }: { input: InputHealth }) {
  const icons = {
    radar: <Satellite size={10} />,
    satellite: <Cloud size={10} />,
    lightning: <Zap size={10} />,
    nwp: <Cloud size={10} />,
  };

  const colors: Record<InputStatus, string> = {
    live:    '#22C55E',
    stale:   '#EAB308',
    missing: '#EF4444',
  };

  const color = colors[input.status];
  const staleText = input.staleAgeSeconds !== null
    ? ` (${durationHuman(input.staleAgeSeconds)} ago)`
    : '';

  return (
    <span
      role="status"
      aria-label={`${input.name}: ${input.status}${staleText}`}
      className="chip"
      style={{ color, background: `${color}18`, border: `1px solid ${color}40` }}
      title={`${input.name.toUpperCase()}: ${input.status.toUpperCase()}${staleText}`}
    >
      {icons[input.name]}
      <span className="uppercase font-bold tracking-wider" style={{ fontSize: 9 }}>
        {input.name.slice(0, 3)}
      </span>
      {input.status === 'live'
        ? <span style={{ opacity: 0.8 }}>LIVE</span>
        : input.status === 'stale'
        ? <span>STALE{staleText}</span>
        : <span>MISS</span>
      }
    </span>
  );
}

export function StatusBar() {
  const { setScreen, replayFrame } = useUIStore();
  const { data: health } = useHealth(replayFrame);

  const isDegraded = health?.modelMode === 'satellite_only';
  const modelLabel = health
    ? `${health.modelName} v${health.modelVersion}${health.hasEnsemble ? ' + diffusion refiner' : ''}`
    : '—';

  return (
    <footer
      className="flex items-center h-8 px-3 gap-3 border-t shrink-0"
      style={{ background: 'var(--panel)', fontSize: 10 }}
      role="contentinfo"
    >
      {/* Input chips */}
      <div className="flex items-center gap-1.5" aria-label="Input data health">
        {health?.inputs.map((input) => (
          <StatusChip key={input.name} input={input} />
        ))}
      </div>

      {/* Satellite-only badge */}
      {isDegraded && (
        <span
          className="chip"
          style={{
            color: '#EAB308',
            background: '#EAB30820',
            border: '1px solid #EAB30860',
            fontWeight: 700,
            animation: 'pulse-dot 2s ease-in-out infinite',
          }}
          role="alert"
          aria-live="polite"
        >
          <Satellite size={10} />
          SATELLITE-ONLY
        </span>
      )}

      <div
        className="text-muted font-mono"
        style={{ fontSize: 10 }}
        aria-label="Active model"
      >
        {health?.mode === 'replay'
          ? <span className="text-muted">REPLAY</span>
          : <span style={{ color: 'var(--accent)' }}>LIVE</span>
        }
        {' | '}
        {modelLabel}
      </div>

      <div className="flex-1" />

      {/* Last run */}
      {health?.lastRunAt && (
        <span className="text-muted font-mono flex items-center gap-1">
          <Clock size={9} />
          Last run: {health.lastRunAt.slice(11, 16)} UTC
        </span>
      )}

      {/* Disclaimer */}
      <button
        id="disclaimer-link"
        onClick={() => setScreen('status')}
        className="text-muted flex items-center gap-1 transition-ui hover:text-accent"
        style={{ fontSize: 10 }}
        title="Warning thresholds are draft placeholders tuned on validation events, not IMD operational criteria."
        aria-label="Disclaimer: warning thresholds are draft placeholders"
      >
        <Info size={9} />
        Draft thresholds
      </button>
    </footer>
  );
}
