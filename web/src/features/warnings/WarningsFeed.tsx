/**
 * features/warnings/WarningsFeed.tsx
 * Time-ordered warnings list with filtering, detail drawer, and CAP export.
 */
import React, { useState, useMemo, useCallback } from 'react';
import { Search, Filter, Download, MapPin, ExternalLink, ChevronRight, X } from 'lucide-react';
import { useWarnings } from '@/data/hooks';
import { useUIStore } from '@/store/uiStore';
import { WarningBadge } from '@/components/WarningBadge';
import { formatDateTime } from '@/lib/timeHelpers';
import type { Warning, WarningLevel } from '@/data/types';
import { getDataSourceSync } from '@/data/dataSourceFactory';

const LEVEL_ORDER: Record<WarningLevel, number> = {
  red: 4, orange: 3, yellow: 2, green: 1,
};

export function WarningDetailDrawer({ warning, onClose }: { warning: Warning; onClose: () => void }) {
  const [capXml, setCapXml] = useState<string | null>(null);
  const [capJson, setCapJson] = useState<object | null>(null);
  const [exporting, setExporting] = useState(false);
  const { timezone, setScreen, setSelectedCellId } = useUIStore();

  const exportCap = async () => {
    setExporting(true);
    try {
      const ds = getDataSourceSync();
      const result = await ds.exportCap(warning.id);
      setCapXml(result.xml);
      setCapJson(result.json);
    } finally {
      setExporting(false);
    }
  };

  const downloadXml = () => {
    if (!capXml) return;
    const blob = new Blob([capXml], { type: 'application/xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${warning.id}.cap.xml`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadJson = () => {
    if (!capJson) return;
    const blob = new Blob([JSON.stringify(capJson, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${warning.id}.cap.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // SMS templates
  const smsTemplates = {
    en: `⚠ ${warning.level.toUpperCase()} WEATHER WARNING\nDistrict: ${warning.districtName}\n${warning.causeText}\nValid until ${formatDateTime(warning.validUntil, timezone)}\nSource: IMD SIH26072`,
    hi: `⚠ ${warning.level.toUpperCase()} मौसम चेतावनी\nज़िला: ${warning.districtName}\nवैध तक: ${formatDateTime(warning.validUntil, timezone)}`,
    or: `⚠ ${warning.level.toUpperCase()} ପାଣିପାଗ ସତର୍କତା\nଜିଲ୍ଲା: ${warning.districtName}\nବୈଧ ଯାଏ: ${formatDateTime(warning.validUntil, timezone)}`,
  };

  return (
    <div
      className="fixed inset-x-0 bottom-0 top-auto md:top-0 md:bottom-0 md:left-auto md:right-0 z-50 flex flex-col animate-fade-in h-[70vh] md:h-full w-full md:w-[420px] rounded-t-xl md:rounded-none border-t md:border-t-0 md:border-l border-[var(--border)]"
      style={{
        background: 'var(--panel)',
        overflowY: 'auto',
      }}
      role="dialog"
      aria-label={`Warning detail: ${warning.districtName}`}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-4 py-3 sticky top-0"
        style={{ background: 'var(--raised)', borderBottom: '1px solid var(--border)' }}
      >
        <div className="flex items-center gap-2">
          <WarningBadge level={warning.level} size="md" />
          <span className="font-bold text-sm">{warning.districtName}</span>
        </div>
        <button onClick={onClose} className="text-muted" aria-label="Close warning detail">
          <X size={16} />
        </button>
      </div>

      <div className="flex flex-col gap-4 p-4">
        {/* Cause chain */}
        <section>
          <div className="label mb-2">CAUSE CHAIN</div>
          <div
            className="text-xs flex flex-col gap-2 p-3 rounded"
            style={{ background: 'var(--raised)', border: '1px solid var(--border)' }}
          >
            <div className="flex items-start gap-2">
              <span className="mono text-muted w-20 shrink-0">Model out</span>
              <span>{warning.causeText}</span>
            </div>
            <div className="w-full h-px" style={{ background: 'var(--border)' }} />
            <div className="flex items-start gap-2">
              <span className="mono text-muted w-20 shrink-0">Rule</span>
              <span>{warning.ruleTriggered}</span>
            </div>
            <div className="w-full h-px" style={{ background: 'var(--border)' }} />
            <div className="flex items-start gap-2">
              <span className="mono text-muted w-20 shrink-0">Value</span>
              <span className="mono">
                {(warning.triggerValue * 100).toFixed(0)}% (threshold: {(warning.triggerThreshold * 100).toFixed(0)}%)
              </span>
            </div>
          </div>

          {/* Cell link */}
          {warning.causeCell && (
            <button
              id={`warning-view-cell-${warning.id}`}
              onClick={() => {
                setSelectedCellId(warning.causeCell);
                setScreen('map');
              }}
              className="flex items-center gap-1.5 mt-2 text-accent text-xs"
            >
              <ExternalLink size={11} />
              View Cell {warning.causeCell} →
            </button>
          )}
        </section>

        {/* Timing */}
        <section>
          <div className="label mb-2">TIMING</div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <div className="text-muted">Issued</div>
              <div className="mono">{formatDateTime(warning.issuedAt, timezone)}</div>
            </div>
            <div>
              <div className="text-muted">Valid until</div>
              <div className="mono">{formatDateTime(warning.validUntil, timezone)}</div>
            </div>
            <div>
              <div className="text-muted">Issued by</div>
              <div className="uppercase font-bold">{warning.issuedBy}</div>
            </div>
            <div>
              <div className="text-muted">Level change</div>
              <div>{warning.previousLevel ? `${warning.previousLevel} → ${warning.level}` : 'First warning'}</div>
            </div>
          </div>
        </section>

        {/* CAP Export */}
        <section>
          <div className="label mb-2">CAP EXPORT (v1.2)</div>
          {!capXml ? (
            <button
              id={`warning-export-cap-${warning.id}`}
              onClick={exportCap}
              disabled={exporting}
              className="flex items-center gap-2 px-3 py-2 rounded text-xs font-semibold transition-ui"
              style={{
                background: 'var(--raised)',
                border: '1px solid var(--border)',
                color: exporting ? 'var(--muted)' : 'var(--text)',
              }}
            >
              <Download size={12} />
              {exporting ? 'Generating...' : 'Generate CAP XML + JSON'}
            </button>
          ) : (
            <div className="flex gap-2">
              <button
                onClick={downloadXml}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-semibold"
                style={{ background: 'var(--accent)', color: '#fff' }}
              >
                <Download size={11} /> XML
              </button>
              <button
                onClick={downloadJson}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-semibold"
                style={{ background: 'var(--raised)', border: '1px solid var(--border)', color: 'var(--text)' }}
              >
                <Download size={11} /> JSON
              </button>
              <pre
                className="mt-2 text-xs font-mono overflow-auto"
                style={{
                  background: 'var(--bg)',
                  border: '1px solid var(--border)',
                  borderRadius: 4,
                  padding: 8,
                  maxHeight: 200,
                  width: '100%',
                  color: 'var(--text)',
                }}
              >
                {capXml.slice(0, 500)}...
              </pre>
            </div>
          )}
        </section>

        {/* SMS templates */}
        <section>
          <div className="label mb-2">SMS PREVIEW</div>
          {Object.entries(smsTemplates).map(([lang, text]) => (
            <div key={lang} className="mb-2">
              <div className="label mb-1">{lang.toUpperCase()}</div>
              <pre
                className="text-xs whitespace-pre-wrap p-2 rounded"
                style={{ background: 'var(--bg)', border: '1px solid var(--border)', color: 'var(--text)' }}
              >
                {text}
              </pre>
            </div>
          ))}
        </section>
      </div>
    </div>
  );
}

export function WarningsFeed() {
  const { replayFrame, timezone, selectedWarningId, setSelectedWarningId,
    warningLevelFilter, setWarningLevelFilter,
    warningDistrictFilter, setWarningDistrictFilter,
    warningShowExpired, setWarningShowExpired,
    setScreen, setSelectedCellId,
  } = useUIStore();
  const { data: warnings, isLoading } = useWarnings(replayFrame);

  const filtered = useMemo(() => {
    let items = warnings ?? [];
    if (warningLevelFilter) items = items.filter((w) => w.level === warningLevelFilter);
    if (warningDistrictFilter) {
      const q = warningDistrictFilter.toLowerCase();
      items = items.filter((w) => w.districtName.toLowerCase().includes(q));
    }
    // Sort newest first
    return [...items].sort((a, b) =>
      new Date(b.issuedAt).getTime() - new Date(a.issuedAt).getTime()
    );
  }, [warnings, warningLevelFilter, warningDistrictFilter]);

  const selectedWarning = warnings?.find((w) => w.id === selectedWarningId);

  if (isLoading) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted text-sm">
        Loading warnings…
      </div>
    );
  }

  return (
    <div className="flex flex-1 overflow-hidden relative">
      {/* Feed */}
      <div className="flex flex-col flex-1 overflow-hidden">
        {/* Toolbar */}
        <div
          className="flex items-center gap-2 px-4 py-2.5"
          style={{ borderBottom: '1px solid var(--border)', background: 'var(--panel)' }}
          role="toolbar"
          aria-label="Warning filters"
        >
          <div
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded flex-1 max-w-xs"
            style={{ background: 'var(--raised)', border: '1px solid var(--border)' }}
          >
            <Search size={12} className="text-muted" />
            <input
              id="warning-search"
              value={warningDistrictFilter}
              onChange={(e) => setWarningDistrictFilter(e.target.value)}
              placeholder="Search district…"
              className="bg-transparent border-none outline-none text-xs flex-1"
              style={{ color: 'var(--text)' }}
              aria-label="Search warnings by district"
            />
          </div>

          {/* Level filter buttons */}
          {(['red', 'orange', 'yellow', 'green'] as WarningLevel[]).map((level) => (
            <button
              key={level}
              id={`warning-filter-${level}`}
              onClick={() => setWarningLevelFilter(warningLevelFilter === level ? null : level)}
              aria-pressed={warningLevelFilter === level}
              style={{ fontSize: 10 }}
            >
              <WarningBadge
                level={level}
                size="sm"
                showText={false}
              />
            </button>
          ))}

          <span className="text-muted text-xs ml-auto">{filtered.length} warnings</span>
        </div>

        {/* List */}
        <div
          className="flex-1 overflow-y-auto"
          role="feed"
          aria-live="polite"
          aria-label="Warnings feed"
        >
          {filtered.length === 0 && (
            <div className="flex items-center justify-center h-32 text-muted text-sm">
              No warnings match current filters.
            </div>
          )}
          {filtered.map((warning) => (
            <button
              key={warning.id}
              id={`warning-row-${warning.id}`}
              onClick={() => setSelectedWarningId(warning.id === selectedWarningId ? null : warning.id)}
              className="w-full text-left px-4 py-3 transition-ui border-b"
              style={{
                borderColor: 'var(--border)',
                background: warning.id === selectedWarningId ? 'var(--raised)' : 'transparent',
              }}
              aria-selected={warning.id === selectedWarningId}
            >
              <div className="flex items-center gap-2">
                <WarningBadge level={warning.level} size="sm" />
                <span className="font-semibold text-sm">{warning.districtName}</span>
                {warning.causeCell && (
                  <span className="text-xs font-bold" style={{ color: '#EAB308' }}>NEW</span>
                )}
                <span className="text-muted text-xs ml-auto font-mono">
                  {formatDateTime(warning.issuedAt, timezone)}
                </span>
                <ChevronRight size={12} className="text-muted" />
              </div>
              <div className="text-xs text-muted mt-1 truncate">{warning.causeText}</div>
              <div className="flex items-center gap-3 mt-1.5">
                {warning.causeCell && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedCellId(warning.causeCell);
                      setScreen('map');
                    }}
                    className="text-xs text-accent flex items-center gap-1"
                    id={`warning-view-cell-row-${warning.id}`}
                  >
                    <ExternalLink size={10} />
                    Cell {warning.causeCell}
                  </button>
                )}
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    setSelectedWarningId(warning.id);
                    setScreen('map');
                  }}
                  className="text-xs text-accent flex items-center gap-1"
                  id={`warning-view-map-${warning.id}`}
                >
                  <MapPin size={10} />
                  View on map
                </button>
                <span className="text-muted text-xs ml-auto">
                  Valid until {formatDateTime(warning.validUntil, timezone)}
                </span>
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Detail drawer */}
      {selectedWarning && (
        <WarningDetailDrawer
          warning={selectedWarning}
          onClose={() => setSelectedWarningId(null)}
        />
      )}
    </div>
  );
}
