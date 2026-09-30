/**
 * components/LayerPanel.tsx
 * Collapsible left panel with layer toggles.
 */
import React from 'react';
import { ChevronLeft, ChevronRight, Eye, EyeOff } from 'lucide-react';
import { useUIStore, type LayerVisibility } from '@/store/uiStore';

// Re-export LayerVisibility from the store
type LayerKey = keyof LayerVisibility;

interface LayerDef {
  key: LayerKey;
  label: string;
  group?: string;
  hint?: string;
}

const LAYERS: LayerDef[] = [
  { key: 'observedRadar',    label: 'Observed Radar',         group: 'Observation' },
  { key: 'lightningStrokes', label: 'Lightning Strokes (−30m)', group: 'Observation' },
  { key: 'satelliteTIR',     label: 'Satellite TIR',           group: 'Observation' },
  { key: 'forecastRadar',    label: 'Forecast Radar',          group: 'Forecast', hint: 'at selected lead time' },
  { key: 'lightningProb30',  label: 'Lightning P(+30 min)',    group: 'Forecast' },
  { key: 'lightningProb60',  label: 'Lightning P(+60 min)',    group: 'Forecast' },
  { key: 'firstFlash',       label: 'First-Flash Risk',         group: 'Forecast' },
  { key: 'cellTracks',       label: 'Cell Tracks / Paths',     group: 'Cells' },
  { key: 'districtWarnings', label: 'District Warnings',       group: 'Warnings' },
  { key: 'districtLabels',   label: 'District Labels',         group: 'Warnings' },
];

export function LayerPanel() {
  const { layerPanelOpen, setLayerPanelOpen, layers, setLayerVisible } = useUIStore();

  if (!layerPanelOpen) {
    return (
      <button
        id="layer-panel-toggle-collapsed"
        onClick={() => setLayerPanelOpen(true)}
        className="flex flex-col items-center justify-center w-8 h-full transition-ui"
        style={{
          background: 'var(--panel)',
          borderRight: '1px solid var(--border)',
          color: 'var(--muted)',
        }}
        aria-label="Open layer panel"
        aria-expanded={false}
      >
        <ChevronRight size={14} />
        <span style={{ writingMode: 'vertical-rl', fontSize: 9, letterSpacing: '0.1em', marginTop: 8, textTransform: 'uppercase', fontWeight: 700 }}>
          Layers
        </span>
      </button>
    );
  }

  const groups = Array.from(new Set(LAYERS.map((l) => l.group)));

  return (
    <aside
      id="layer-panel"
      className="flex flex-col absolute md:relative bottom-0 md:bottom-auto left-0 right-0 md:left-auto md:right-auto z-40 md:z-auto h-[60vh] md:h-full transition-transform border-t md:border-t-0 md:border-r border-[var(--border)] rounded-t-xl md:rounded-none w-full md:w-[280px]"
      style={{
        background: 'var(--panel)',
        flexShrink: 0,
        overflowY: 'auto',
      }}
      aria-label="Map layers"
    >
      <div className="flex items-center justify-between panel-heading">
        <span>Layers</span>
        <button
          id="layer-panel-toggle"
          onClick={() => setLayerPanelOpen(false)}
          className="transition-ui"
          style={{ color: 'var(--muted)' }}
          aria-label="Close layer panel"
          aria-expanded={true}
        >
          <ChevronLeft size={14} />
        </button>
      </div>

      <div className="flex flex-col gap-0 overflow-y-auto flex-1">
        {groups.map((group) => (
          <div key={group}>
            <div
              className="px-3 py-1.5 label"
              style={{ borderTop: '1px solid var(--border)' }}
            >
              {group}
            </div>
            {LAYERS.filter((l) => l.group === group).map((layer) => (
              <label
                key={layer.key}
                className="flex items-center gap-2.5 px-3 py-2 cursor-pointer transition-ui hover:bg-raised min-h-[44px] md:min-h-0"
                style={{ fontSize: 12 }}
              >
                <button
                  role="checkbox"
                  aria-checked={layers[layer.key]}
                  onClick={() => setLayerVisible(layer.key, !layers[layer.key])}
                  className="flex items-center justify-center w-4 h-4 rounded transition-ui border"
                  style={{
                    background: layers[layer.key] ? 'var(--accent)' : 'transparent',
                    borderColor: layers[layer.key] ? 'var(--accent)' : 'var(--border)',
                    color: '#fff',
                    flexShrink: 0,
                  }}
                  aria-label={`Toggle ${layer.label}`}
                >
                  {layers[layer.key] && <Eye size={10} />}
                  {!layers[layer.key] && <EyeOff size={10} style={{ opacity: 0.4 }} />}
                </button>
                <span style={{ color: layers[layer.key] ? 'var(--text)' : 'var(--muted)' }}>
                  {layer.label}
                </span>
                {layer.hint && (
                  <span className="text-muted ml-auto" style={{ fontSize: 10 }}>
                    {layer.hint}
                  </span>
                )}
              </label>
            ))}
          </div>
        ))}
      </div>
    </aside>
  );
}

// Export LayerVisibility type for consumers
export type { LayerKey };
