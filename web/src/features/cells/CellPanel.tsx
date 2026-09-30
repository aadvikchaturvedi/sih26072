/**
 * features/cells/CellPanel.tsx
 * Right-side panel opened when a storm cell is selected.
 * Shows concept-card fields and time-series charts.
 */
import React from 'react';
import { X, Navigation, Zap, TrendingUp, AlertTriangle, Info } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Legend,
} from 'recharts';
import { useUIStore } from '@/store/uiStore';
import { useCells } from '@/data/hooks';
import type { Cell, RiskLevel } from '@/data/types';
import { formatTime } from '@/lib/timeHelpers';
import { addMinutes, parseISO } from 'date-fns';

const RISK_COLORS: Record<RiskLevel, string> = {
  high:     '#EF4444',
  moderate: '#F97316',
  low:      '#22C55E',
};

const TREND_LABELS: Record<string, string> = {
  rapidly_increasing: 'RAPIDLY INCREASING ↑↑',
  increasing:         'INCREASING ↑',
  steady:             'STEADY →',
  decreasing:         'DECREASING ↓',
};

const STATUS_LABELS: Record<string, string> = {
  initiating: 'INITIATING',
  growing:    'GROWING',
  mature:     'MATURE',
  decaying:   'DECAYING',
};

function StatBlock({ label, value, unit }: { label: string; value: React.ReactNode; unit?: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={{ fontSize: 16 }}>
        {value}
        {unit && <span className="text-muted" style={{ fontSize: 11, marginLeft: 3 }}>{unit}</span>}
      </div>
    </div>
  );
}

function ConfidenceTooltip() {
  const [show, setShow] = React.useState(false);
  return (
    <span className="relative inline-block">
      <button
        onMouseEnter={() => setShow(true)}
        onMouseLeave={() => setShow(false)}
        className="text-muted ml-1"
        aria-label="Confidence metric information"
      >
        <Info size={11} />
      </button>
      {show && (
        <div
          className="absolute left-4 bottom-4 w-52 text-xs p-2 rounded z-50"
          style={{
            background: 'var(--raised)',
            border: '1px solid var(--border)',
            lineHeight: 1.4,
          }}
        >
          Provisional: definition pending backend (ensemble spread + input completeness)
        </div>
      )}
    </span>
  );
}

export function CellPanel() {
  const { selectedCellId, setSelectedCellId, cellPanelOpen, timezone, replayFrame, selectedLead } = useUIStore();
  const { data: cells } = useCells(replayFrame);

  const cell: Cell | undefined = cells?.find((c) => c.id === selectedCellId);

  if (!cellPanelOpen || !cell) return null;

  const statusColor = {
    initiating: '#60A5FA',
    growing:    '#F97316',
    mature:     '#EAB308',
    decaying:   '#8B97A6',
  }[cell.status];

  // Forecast path by lead
  const forecastByLead = [30, 60, 90].map((lead) => {
    const fp = cell.forecastPath.find((p) => p.leadMin === lead);
    return { lead, fp };
  });

  // Chart data: combined dBZ series
  const chartData = cell.dbzSeries.map((d, i) => ({
    name: formatTime(d.time, timezone),
    dbz: Math.round(d.dbz),
    flash: cell.flashRateSeries[i]?.rate.toFixed(1),
    isForecast: d.isForecast,
  }));

  return (
    <aside
      id="cell-panel"
      className="flex flex-col absolute md:relative bottom-0 md:bottom-auto left-0 right-0 md:left-auto md:right-auto z-40 md:z-auto h-[60vh] md:h-full animate-fade-in transition-transform border-t md:border-t-0 md:border-l border-[var(--border)] rounded-t-xl md:rounded-none w-full md:w-[360px]"
      style={{
        background: 'var(--panel)',
        flexShrink: 0,
        overflowY: 'auto',
      }}
      aria-label={`Cell ${cell.id} details`}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-3 py-2.5"
        style={{ borderBottom: '1px solid var(--border)', background: 'var(--raised)' }}
      >
        <div className="flex items-center gap-2">
          <span className="font-bold text-sm">Cell {cell.id}</span>
          <span
            className="badge"
            style={{ color: statusColor, borderColor: statusColor, background: `${statusColor}18`, fontSize: 10 }}
          >
            {STATUS_LABELS[cell.status]}
          </span>
          {cell.isNewCell && (
            <span
              className="badge"
              style={{ color: '#000', background: '#EAB308', borderColor: '#EAB308', fontSize: 9 }}
              role="img"
              aria-label="New cell - predicted by nowcast, not tracked from observation"
            >
              NEW CELL
            </span>
          )}
        </div>
        <button
          id="cell-panel-close"
          onClick={() => setSelectedCellId(null)}
          className="transition-ui text-muted hover:text-text p-1 rounded"
          aria-label="Close cell panel"
        >
          <X size={14} />
        </button>
      </div>

      {/* Lightning jump alert */}
      {cell.lightningJumpFlag && (
        <div
          className="flex items-center gap-2 px-3 py-2 text-xs"
          style={{
            background: '#EF444418',
            borderBottom: '1px solid #EF444440',
            color: '#EF4444',
          }}
          role="alert"
          aria-live="polite"
        >
          <Zap size={12} />
          <span className="font-bold">Lightning jump detected</span>
          {cell.lightningJumpAt && (
            <span className="text-muted ml-1">at {formatTime(cell.lightningJumpAt, timezone)}</span>
          )}
        </div>
      )}

      <div className="flex flex-col gap-0 overflow-y-auto flex-1">
        {/* Key stats */}
        <div
          className="grid grid-cols-2 gap-3 px-3 py-3"
          style={{ borderBottom: '1px solid var(--border)' }}
        >
          <StatBlock label="Max dBZ" value={Math.round(cell.maxDbz)} unit="dBZ" />
          <StatBlock
            label="Growth"
            value={
              <span style={{ color: cell.growthDbzPer10Min > 0 ? '#F97316' : '#8B97A6' }}>
                {cell.growthDbzPer10Min > 0 ? '+' : ''}{cell.growthDbzPer10Min.toFixed(1)}
              </span>
            }
            unit="dBZ/10m"
          />
          <div className="col-span-2">
            <div className="stat-label">Lightning Trend</div>
            <div className="font-bold text-xs mt-0.5" style={{ color: '#EAB308' }}>
              <Zap size={11} className="inline mr-1" />
              {TREND_LABELS[cell.lightningTrend]}
            </div>
          </div>
          <div className="col-span-2">
            <div className="stat-label">Movement</div>
            <div className="flex items-center gap-2 mt-0.5">
              <Navigation size={12} style={{ transform: `rotate(${cell.headingDeg}deg)`, color: 'var(--accent)' }} />
              <span className="mono font-bold text-sm">{cell.headingDeg}°</span>
              <span className="text-muted text-xs">at</span>
              <span className="mono font-bold text-sm">{cell.speedKmh} km/h</span>
            </div>
          </div>
        </div>

        {/* Nowcast block */}
        <div
          className="px-3 py-3"
          style={{ borderBottom: '1px solid var(--border)' }}
        >
          <div className="panel-heading px-0 py-1 mb-2" style={{ borderBottom: 'none' }}>
            NOWCAST
          </div>
          <div className="flex flex-col gap-1.5">
            {forecastByLead.map(({ lead, fp }) => (
              <div key={lead} className="flex items-center justify-between text-xs">
                <span className="mono text-muted">+{lead}m</span>
                {fp ? (
                  <>
                    <span className="mono">{fp.lon.toFixed(2)}, {fp.lat.toFixed(2)}</span>
                    <span className="mono text-muted">{Math.round(fp.expectedDbz)} dBZ</span>
                    <span
                      className="badge"
                      style={{
                        color: RISK_COLORS[fp.risk],
                        borderColor: RISK_COLORS[fp.risk],
                        background: `${RISK_COLORS[fp.risk]}18`,
                        fontSize: 9,
                      }}
                    >
                      {fp.risk.toUpperCase()}
                      {fp.riskBasis === 'reflectivity_based' && (
                        <span title="Derived from reflectivity — no lightning probability available for this lead time" style={{ marginLeft: 2 }}>
                          dBZ
                        </span>
                      )}
                    </span>
                  </>
                ) : (
                  <span className="text-muted">—</span>
                )}
              </div>
            ))}
          </div>

          {/* Confidence */}
          {cell.confidence !== undefined && (
            <div className="mt-3 flex items-center gap-2">
              <div className="stat-label">Confidence</div>
              <ConfidenceTooltip />
              <div className="flex-1 h-1.5 rounded-full" style={{ background: 'var(--border)' }}>
                <div
                  className="h-full rounded-full transition-ui"
                  style={{
                    width: `${cell.confidence}%`,
                    background: cell.confidence > 70 ? '#22C55E' : cell.confidence > 40 ? '#EAB308' : '#EF4444',
                  }}
                />
              </div>
              <span className="mono text-xs">{cell.confidence}%</span>
            </div>
          )}

          {/* Model mode + data quality */}
          <div className="flex items-center gap-3 mt-2 text-xs text-muted">
            <span className="font-bold uppercase">{cell.modelMode === 'satellite_only' ? '🛰 SAT-ONLY' : '📡 FULL'}</span>
            <span>|</span>
            <span>Data quality: {Math.round(cell.dataQuality * 100)}%</span>
          </div>
        </div>

        {/* Affected districts */}
        {cell.affectedDistricts.length > 0 && (
          <div
            className="px-3 py-3"
            style={{ borderBottom: '1px solid var(--border)' }}
          >
            <div className="label mb-2">DISTRICTS LIKELY AFFECTED</div>
            <div className="flex flex-col gap-1">
              {cell.affectedDistricts.map((d) => (
                <div key={d.districtId} className="flex items-center justify-between text-xs">
                  <span>{d.name}</span>
                  <span className="mono text-muted">ETA +{d.etaMin}m</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Charts */}
        <div className="px-3 py-3" style={{ borderBottom: '1px solid var(--border)' }}>
          <div className="label mb-2">MAX dBZ + FLASH RATE TREND</div>
          <ResponsiveContainer width="100%" height={140}>
            <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.5} />
              <XAxis dataKey="name" tick={{ fontSize: 9, fill: 'var(--muted)' }} interval={2} />
              <YAxis tick={{ fontSize: 9, fill: 'var(--muted)' }} />
              <Tooltip
                contentStyle={{
                  background: 'var(--raised)',
                  border: '1px solid var(--border)',
                  borderRadius: 4,
                  fontSize: 11,
                }}
              />
              {cell.lightningJumpAt && (
                <ReferenceLine
                  x={formatTime(cell.lightningJumpAt, timezone)}
                  stroke="#EF4444"
                  strokeDasharray="4 2"
                  label={{ value: '⚡', position: 'top', fontSize: 12 }}
                />
              )}
              {selectedLead !== 0 && (
                <ReferenceLine
                  x={formatTime(addMinutes(parseISO(cell.dbzSeries[0]?.time || new Date().toISOString()), selectedLead).toISOString(), timezone)}
                  stroke="var(--accent)"
                  strokeDasharray="2 2"
                />
              )}
              <Line
                type="monotone"
                dataKey="dbz"
                stroke="#2F6FEB"
                strokeWidth={2}
                dot={false}
                name="Max dBZ"
                strokeDasharray="0"
              />
              <Line
                type="monotone"
                dataKey="flash"
                stroke="#EAB308"
                strokeWidth={1.5}
                dot={false}
                name="Flash/min"
              />
              <Legend wrapperStyle={{ fontSize: 10, paddingTop: 4 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Review button */}
        <div className="px-3 py-3">
          <button
            id={`cell-review-warning-${cell.id}`}
            className="w-full flex items-center justify-center gap-2 py-2 rounded font-semibold text-xs transition-ui"
            style={{
              background: 'var(--accent)',
              color: '#fff',
            }}
            onClick={() => {
              useUIStore.getState().setScreen('warnings');
            }}
          >
            <AlertTriangle size={13} />
            Review / Issue Warning
          </button>
        </div>
      </div>
    </aside>
  );
}
