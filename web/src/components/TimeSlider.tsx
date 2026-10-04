/**
 * components/TimeSlider.tsx
 * Lead-time slider spanning -60 to 0 (observed) and +10 to +120 (forecast).
 * Keyboard: left/right arrows step by 10 minutes.
 */
import React, { useCallback } from 'react';
import { useUIStore } from '@/store/uiStore';
import { frameT0, useTimeline } from '@/data/timeline';
import { formatTime } from '@/lib/timeHelpers';
import { addMinutes, parseISO } from 'date-fns';

const LEADS = Array.from({ length: 19 }, (_, i) => (i - 6) * 10);

function formatLeadLabel(lead: number): string {
  if (lead === 0) return 'NOW';
  if (lead > 0) return `+${lead}m`;
  return `${lead}m`;
}

export function TimeSlider() {
  const { replayFrame, selectedLead, setSelectedLead, timezone } = useUIStore();
  
  // The actual reference t0 for the current frame
  const t0 = parseISO(frameT0(replayFrame, useTimeline()));

  const handleKey = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'ArrowLeft') {
      e.preventDefault();
      setSelectedLead(Math.max(-60, selectedLead - 10));
    } else if (e.key === 'ArrowRight') {
      e.preventDefault();
      setSelectedLead(Math.min(120, selectedLead + 10));
    }
  }, [selectedLead, setSelectedLead]);

  const isObs = selectedLead <= 0;

  return (
    <div
      className="flex flex-col gap-1 px-4 py-2"
      style={{ background: 'var(--panel)', borderTop: '1px solid var(--border)' }}
      aria-label="Time slider"
    >
      {/* Labels row */}
      <div className="flex items-center justify-between mb-0.5">
        <span className="label">◀ OBSERVED</span>
        <div className="flex flex-col items-center">
          <span
            className="mono font-bold text-xs"
            style={{ color: isObs ? 'var(--text)' : 'var(--warn-yellow)' }}
          >
            {formatLeadLabel(selectedLead)}
          </span>
          <span className="mono text-muted" style={{ fontSize: 10 }}>
            {formatTime(addMinutes(t0, selectedLead).toISOString(), timezone)}
          </span>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setSelectedLead(0)}
            className="label transition-ui hover:text-white min-h-[44px] md:min-h-0"
            style={{ 
              background: 'var(--raised)', 
              padding: '2px 6px', 
              borderRadius: 4,
              border: '1px solid var(--border)' 
            }}
          >
            JUMP TO NOW
          </button>
          <span className="label">FORECAST ▶</span>
        </div>
      </div>

      {/* Slider track */}
      <div className="relative flex items-center" style={{ height: 28 }}>
        {/* Background track segments */}
        <div
          className="absolute inset-y-0 left-0 rounded-l"
          style={{
            width: `${(6 / 18) * 100}%`,
            background: 'var(--raised)',
            border: '1px solid var(--border)',
            borderRight: 'none',
          }}
        />
        <div
          className="absolute inset-y-0 rounded-r"
          style={{
            left: `${(6 / 18) * 100}%`,
            right: 0,
            background: 'rgba(255,200,0,0.08)',
            border: '1px dashed var(--warn-yellow)',
            borderLeft: 'none',
          }}
        />

        {/* NOW marker */}
        <div
          className="absolute z-10 w-px h-6"
          style={{
            left: `${(6 / 18) * 100}%`,
            background: 'var(--muted)',
          }}
        />

        {/* Tick marks */}
        <div className="absolute inset-x-0 flex justify-between px-0 pointer-events-none" style={{ top: '50%', transform: 'translateY(-50%)' }}>
          {LEADS.map((lead) => (
            <button
              key={lead}
              onClick={() => setSelectedLead(lead)}
              className="w-2 h-2 rounded-full transition-ui"
              style={{
                background: lead === selectedLead ? 'var(--accent)' : lead <= 0 ? 'var(--muted)' : 'var(--warn-yellow)',
                transform: lead === selectedLead ? 'scale(1.6)' : 'scale(1)',
                pointerEvents: 'auto',
              }}
              aria-label={`Lead ${lead}: ${formatLeadLabel(lead)}`}
            />
          ))}
        </div>

        {/* Native range input for keyboard access */}
        <input
          id="time-slider"
          type="range"
          min={0}
          max={18}
          value={(selectedLead + 60) / 10}
          onChange={(e) => setSelectedLead((Number(e.target.value) - 6) * 10)}
          onKeyDown={handleKey}
          className="absolute inset-0 w-full opacity-0 cursor-pointer"
          style={{ height: '100%' }}
          aria-label="Select lead time"
          aria-valuetext={`${formatLeadLabel(selectedLead)}`}
        />
      </div>

      {/* Forecast tag */}
      {!isObs && (
        <div className="flex justify-end mt-0.5" aria-live="polite">
          <span className="forecast-tag">FORECAST {formatLeadLabel(selectedLead)}</span>
        </div>
      )}
    </div>
  );
}
