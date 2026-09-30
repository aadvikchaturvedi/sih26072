/**
 * features/status/StatusPage.tsx
 * System status detail page/popover.
 */
import React from 'react';
import { Satellite, Cloud, Zap, Activity, Clock, AlertTriangle } from 'lucide-react';
import { useHealth } from '@/data/hooks';
import { formatDateTime, durationHuman } from '@/lib/timeHelpers';
import { useUIStore } from '@/store/uiStore';
import type { InputHealth } from '@/data/types';

function InputRow({ input, timezone }: { input: InputHealth; timezone: 'UTC' | 'IST' }) {
  const icons = {
    radar: <Satellite size={14} />,
    satellite: <Cloud size={14} />,
    lightning: <Zap size={14} />,
    nwp: <Activity size={14} />,
  };

  const colors = {
    live:    '#22C55E',
    stale:   '#EAB308',
    missing: '#EF4444',
  };

  const color = colors[input.status];

  return (
    <div
      className="flex items-center gap-3 px-4 py-3"
      style={{ borderBottom: '1px solid var(--border)' }}
    >
      <div style={{ color: 'var(--muted)' }}>{icons[input.name]}</div>
      <div className="flex-1">
        <div className="font-semibold text-sm uppercase">{input.name}</div>
        {input.lastReceived && (
          <div className="text-muted text-xs font-mono">
            Last: {formatDateTime(input.lastReceived, timezone)}
            {input.staleAgeSeconds !== null && ` (${durationHuman(input.staleAgeSeconds)} ago)`}
          </div>
        )}
      </div>
      <div
        className="chip"
        style={{
          color,
          background: `${color}18`,
          border: `1px solid ${color}40`,
          fontWeight: 700,
        }}
      >
        {input.status.toUpperCase()}
        {input.staleAgeSeconds !== null && ` +${durationHuman(input.staleAgeSeconds)}`}
      </div>
    </div>
  );
}

export function StatusPage() {
  const { timezone, replayFrame } = useUIStore();
  const { data: health } = useHealth(replayFrame);

  const isDegraded = health?.modelMode === 'satellite_only';

  return (
    <div
      className="flex-1 overflow-y-auto"
      style={{ background: 'var(--bg)' }}
    >
      <div className="max-w-2xl mx-auto p-6 flex flex-col gap-4">
        <h1 className="text-base font-bold">System Status</h1>

        {/* Degraded mode banner */}
        {isDegraded && (
          <div
            className="flex items-center gap-3 px-4 py-3 rounded"
            style={{
              background: '#EAB30818',
              border: '1px solid #EAB30840',
            }}
            role="alert"
          >
            <AlertTriangle size={16} style={{ color: '#EAB308' }} />
            <div>
              <div className="font-bold text-sm" style={{ color: '#EAB308' }}>SATELLITE-ONLY MODE ACTIVE</div>
              <div className="text-xs text-muted mt-0.5">
                Radar feed missing. Model running with satellite, lightning, and NWP inputs only.
                Reflectivity forecasts not available. Lightning probabilities derived from satellite TIR + NWP.
              </div>
            </div>
          </div>
        )}

        {/* Pipeline status */}
        <div
          className="rounded overflow-hidden"
          style={{ background: 'var(--panel)', border: '1px solid var(--border)' }}
        >
          <div className="panel-heading">Pipeline</div>
          <div className="grid grid-cols-2 gap-4 px-4 py-3 text-xs">
            <div>
              <div className="label">Last successful run</div>
              <div className="mono mt-0.5">{health?.lastRunAt ? formatDateTime(health.lastRunAt, timezone) : '—'}</div>
            </div>
            <div>
              <div className="label">Next scheduled run</div>
              <div className="mono mt-0.5">{health?.nextRunAt ? formatDateTime(health.nextRunAt, timezone) : '—'}</div>
            </div>
            <div>
              <div className="label">Mode</div>
              <div className="font-bold uppercase mt-0.5">{health?.mode ?? '—'}</div>
            </div>
            <div>
              <div className="label">Model mode</div>
              <div className="font-bold uppercase mt-0.5" style={{ color: isDegraded ? '#EAB308' : '#22C55E' }}>
                {isDegraded ? '🛰 SATELLITE-ONLY' : '📡 FULL'}
              </div>
            </div>
            <div>
              <div className="label">Model</div>
              <div className="mono mt-0.5">{health?.modelName} v{health?.modelVersion}</div>
            </div>
            <div>
              <div className="label">Inference time</div>
              <div className="mono mt-0.5">{health?.inferenceMsLast != null ? `${health.inferenceMsLast} ms` : '—'}</div>
            </div>
            {health?.missingChannels && health.missingChannels.length > 0 && (
              <div className="col-span-2">
                <div className="label">Missing channels</div>
                <div className="mono mt-0.5 text-warn-orange">{health.missingChannels.join(', ')}</div>
              </div>
            )}
            <div>
              <div className="label">Ensemble</div>
              <div className="font-bold mt-0.5">{health?.hasEnsemble ? 'YES' : 'NO'}</div>
            </div>
          </div>
        </div>

        {/* Input health */}
        <div
          className="rounded overflow-hidden"
          style={{ background: 'var(--panel)', border: '1px solid var(--border)' }}
        >
          <div className="panel-heading">Input Data Health</div>
          {health?.inputs.map((input) => (
            <InputRow key={input.name} input={input} timezone={timezone} />
          ))}
        </div>

        {/* Disclaimer */}
        <div
          className="text-xs text-muted p-4 rounded"
          style={{ background: 'var(--raised)', border: '1px solid var(--border)' }}
          role="note"
        >
          <div className="font-bold mb-1">⚠ Important Disclaimer</div>
          <p>
            Warning thresholds are draft placeholders tuned on validation events, <strong>not IMD operational criteria</strong>.
            This system is a prototype developed for Smart India Hackathon 2026, PS 26072,
            Ministry of Earth Sciences. All warnings must be reviewed by a qualified forecaster before dissemination.
          </p>
        </div>
      </div>
    </div>
  );
}
