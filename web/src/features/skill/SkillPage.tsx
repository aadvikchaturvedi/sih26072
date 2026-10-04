/**
 * features/skill/SkillPage.tsx
 * Aggregate skill verification metrics and charts.
 */
import React, { useCallback } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ScatterChart, Scatter, ReferenceLine,
} from 'recharts';
import { useSkill } from '@/data/hooks';
import { useUIStore } from '@/store/uiStore';
import type { ModelSkill } from '@/data/types';
import { Download } from 'lucide-react';

const THRESHOLD_OPTIONS = [20, 35, 45] as const;

function HeadlineCard({ label, value, unit, sub }: {
  label: string;
  value: string | number;
  unit?: string;
  sub?: string;
}) {
  return (
    <div
      className="flex flex-col gap-1 p-4 rounded"
      style={{ background: 'var(--raised)', border: '1px solid var(--border)', minWidth: 140 }}
    >
      <div className="label">{label}</div>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 28, fontWeight: 700, color: 'var(--accent)' }}>
        {value}
        {unit && <span style={{ fontSize: 14, fontWeight: 400, color: 'var(--muted)', marginLeft: 4 }}>{unit}</span>}
      </div>
      {sub && <div className="text-muted" style={{ fontSize: 11 }}>{sub}</div>}
    </div>
  );
}

function MetricsTable({ models }: { models: ModelSkill[] }) {
  const cols = ['label', 'pod', 'far', 'brierSkillScore', 'rocAuc', 'firstFlashHitRate', 'firstFlashFAR', 'medianFirstFlashLeadMin'];
  const headers: Record<string, string> = {
    label: 'Model',
    pod: 'POD',
    far: 'FAR',
    brierSkillScore: 'BSS',
    rocAuc: 'ROC-AUC',
    firstFlashHitRate: '1st-Flash HR',
    firstFlashFAR: '1st-Flash FAR',
    medianFirstFlashLeadMin: 'Lead Time (min)',
  };

  const downloadCsv = useCallback(() => {
    const header = cols.map((c) => headers[c]).join(',');
    const rows = models.map((m) =>
      cols.map((c) => {
        const v = m[c as keyof ModelSkill];
        return typeof v === 'number' ? v.toFixed(3) : v == null ? '' : String(v);
      }).join(',')
    );
    const csv = [header, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'skill_metrics.csv';
    a.click();
    URL.revokeObjectURL(url);
  }, [models]);

  return (
    <div>
      <div className="flex items-center justify-between mb-2">
        <div className="label">MODEL COMPARISON</div>
        <button
          id="skill-download-csv"
          onClick={downloadCsv}
          className="flex items-center gap-1.5 text-xs px-2 py-1 rounded"
          style={{ background: 'var(--raised)', border: '1px solid var(--border)', color: 'var(--text)' }}
        >
          <Download size={11} /> CSV
        </button>
      </div>
      <div className="overflow-x-auto">
        <table
          className="w-full text-xs"
          style={{ borderCollapse: 'collapse' }}
          aria-label="Model skill comparison table"
        >
          <thead>
            <tr style={{ borderBottom: '1px solid var(--border)' }}>
              {cols.map((c) => (
                <th key={c} className="text-left px-2 py-1.5 text-muted font-semibold whitespace-nowrap">
                  {headers[c]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {models.map((m, i) => (
              <tr
                key={m.modelId}
                style={{
                  borderBottom: '1px solid var(--border)',
                  background: i % 2 === 0 ? 'transparent' : 'var(--raised)',
                }}
              >
                {cols.map((c) => {
                  const v = m[c as keyof ModelSkill];
                  const isNum = typeof v === 'number';
                  const isHighlight = c === 'medianFirstFlashLeadMin' && m.modelId === 'F3F4';
                  return (
                    <td
                      key={c}
                      className="px-2 py-1.5 font-mono"
                      style={{
                        color: isHighlight ? 'var(--accent)' : 'var(--text)',
                        fontWeight: isHighlight ? 700 : 400,
                      }}
                    >
                      {c === 'label' ? (
                        <div className="flex items-center gap-1.5">
                          <div style={{ width: 10, height: 10, borderRadius: 2, background: m.color, flexShrink: 0 }} />
                          {String(v)}
                        </div>
                      ) : isNum ? (
                        (v as number).toFixed(3)
                      ) : v == null ? '—' : String(v)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function SkillPage() {
  const { data: skill, isLoading } = useSkill();
  const { csiThreshold, setCsiThreshold } = useUIStore();

  if (isLoading || !skill) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted text-sm">
        Loading skill data…
      </div>
    );
  }

  // Prepare CSI data for selected threshold
  const csiData = skill.models[0].csiByThreshold[csiThreshold].map((_, i) => {
    const point: Record<string, number> = {
      lead: skill.models[0].csiByThreshold[csiThreshold][i].leadMin,
    };
    skill.models.forEach((m) => {
      const curve = m.csiByThreshold[csiThreshold];
      if (curve[i]) point[m.modelId] = parseFloat(curve[i].csi.toFixed(3));
    });
    return point;
  });

  // Reliability diagram data (use F3F4)
  const f3f4 = skill.models.find((m) => m.modelId === 'F3F4') ?? skill.models[skill.models.length - 1];
  const pct = (v: number | null) => (v == null ? '—' : `${(v * 100).toFixed(0)}%`);
  const reliabilityData = f3f4.reliabilityDiagram.map((p) => ({
    forecast: p.forecastProb,
    observed: p.observedFreq,
    count: p.count,
  }));

  return (
    <div
      className="flex-1 overflow-y-auto p-6 flex flex-col gap-6"
      style={{ background: 'var(--bg)' }}
    >
      {/* Headline cards */}
      <div>
        <h1 className="text-base font-bold mb-3">Forecast Skill — {skill.eventLabel}</h1>
        {skill.synthetic && (
          <div
            className="text-xs p-3 rounded mb-3"
            style={{ background: '#EAB30818', border: '1px solid #EAB30840', color: '#EAB308' }}
            role="note"
          >
            These scores were computed on synthetic storms because no real training or
            verification data is available yet. They show that the pipeline works, not how
            well the model forecasts real weather.
          </div>
        )}
        <div className="flex gap-3 flex-wrap">
          <HeadlineCard
            label="MEDIAN 1ST-FLASH LEAD TIME"
            value={skill.medianFirstFlashLeadMin ?? '—'}
            unit="min"
            sub="F3 + F4 (lightning head)"
          />
          <HeadlineCard
            label="FIRST-FLASH HIT RATE"
            value={pct(f3f4.firstFlashHitRate)}
            sub="F3 + F4"
          />
          <HeadlineCard
            label="FIRST-FLASH FAR"
            value={pct(f3f4.firstFlashFAR)}
            sub="F3 + F4"
          />
          <HeadlineCard
            label="ROC-AUC (BEST)"
            value={f3f4.rocAuc?.toFixed(2) ?? '—'}
            sub="F3 + F4"
          />
        </div>
      </div>

      {/* CSI vs Lead time */}
      <div
        className="p-4 rounded"
        style={{ background: 'var(--panel)', border: '1px solid var(--border)' }}
      >
        <div className="flex items-center justify-between mb-3">
          <div className="label">CSI vs LEAD TIME</div>
          <div className="flex items-center gap-1">
            {THRESHOLD_OPTIONS.map((t) => (
              <button
                key={t}
                id={`csi-threshold-${t}`}
                onClick={() => setCsiThreshold(t)}
                aria-pressed={csiThreshold === t}
                className="px-2 py-1 rounded text-xs font-semibold"
                style={{
                  background: csiThreshold === t ? 'var(--accent)' : 'var(--raised)',
                  color: csiThreshold === t ? '#fff' : 'var(--muted)',
                  border: `1px solid ${csiThreshold === t ? 'var(--accent)' : 'var(--border)'}`,
                }}
              >
                {t} dBZ
              </button>
            ))}
          </div>
        </div>
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={csiData} margin={{ top: 4, right: 8, bottom: 4, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.6} />
            <XAxis dataKey="lead" tickFormatter={(v: number) => `+${v}m`} tick={{ fontSize: 10, fill: 'var(--muted)' }} />
            <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: 'var(--muted)' }} tickFormatter={(v: number) => v.toFixed(1)} />
            <Tooltip
              formatter={(v: unknown) => typeof v === 'number' ? v.toFixed(3) : String(v)}
              labelFormatter={(l: unknown) => `Lead +${l}m`}
              contentStyle={{ background: 'var(--raised)', border: '1px solid var(--border)', borderRadius: 4, fontSize: 11 }}
            />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {skill.models.map((m) => (
              <Line
                key={m.modelId}
                type="monotone"
                dataKey={m.modelId}
                name={m.label}
                stroke={m.color}
                strokeWidth={m.modelId === 'F3F4' ? 2.5 : 1.5}
                strokeDasharray={m.modelId === 'persistence' ? '4 2' : undefined}
                dot={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Reliability diagram */}
      <div
        className="p-4 rounded"
        style={{ background: 'var(--panel)', border: '1px solid var(--border)' }}
      >
        <div className="label mb-3">RELIABILITY DIAGRAM (F3 + F4)</div>
        <ResponsiveContainer width="100%" height={200}>
          <ScatterChart margin={{ top: 4, right: 8, bottom: 4, left: -10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" strokeOpacity={0.6} />
            <XAxis dataKey="forecast" type="number" domain={[0, 1]} name="Forecast probability"
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              tick={{ fontSize: 10, fill: 'var(--muted)' }}
            />
            <YAxis dataKey="observed" type="number" domain={[0, 1]} name="Observed frequency"
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              tick={{ fontSize: 10, fill: 'var(--muted)' }}
            />
            <ReferenceLine stroke="var(--muted)" strokeDasharray="4 2"
              segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
            />
            <Scatter data={reliabilityData} fill="var(--accent)" />
            <Tooltip
              cursor={{ strokeDasharray: '3 3' }}
              formatter={(v: unknown) => typeof v === 'number' ? `${(v * 100).toFixed(1)}%` : String(v)}
              contentStyle={{ background: 'var(--raised)', border: '1px solid var(--border)', borderRadius: 4, fontSize: 11 }}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Metrics table */}
      <div
        className="p-4 rounded"
        style={{ background: 'var(--panel)', border: '1px solid var(--border)' }}
      >
        <MetricsTable models={skill.models} />
      </div>

      {/* Sticky note */}
      <div
        className="text-xs text-muted p-3 rounded"
        style={{ background: 'var(--raised)', border: '1px solid var(--border)' }}
        role="note"
      >
        Generated by the F7 evaluation suite · Report timestamp: {skill.reportAt.slice(0, 10)}
      </div>
    </div>
  );
}
