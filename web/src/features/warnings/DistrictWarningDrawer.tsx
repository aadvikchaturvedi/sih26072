import React from 'react';
import { useWarnings } from '@/data/hooks';
import { useUIStore } from '@/store/uiStore';
import { WarningDetailDrawer } from './WarningsFeed';

export function DistrictWarningDrawer() {
  const { selectedDistrictId, setSelectedDistrictId, replayFrame } = useUIStore();
  const { data: warnings } = useWarnings(replayFrame);

  if (!selectedDistrictId || !warnings) return null;

  const warning = warnings.find((w) => w.districtId === selectedDistrictId);

  // If there's no warning for this district, we could just show empty or nothing.
  // The prompt says: click district -> warning detail drawer
  if (!warning) {
    return (
      <div 
        className="absolute right-0 top-0 bottom-0 w-80 shadow-2xl flex flex-col overflow-hidden animate-slide-in-right z-50"
        style={{ background: 'var(--panel)', borderLeft: '1px solid var(--border)' }}
      >
        <div className="flex items-center justify-between p-4" style={{ borderBottom: '1px solid var(--border)' }}>
          <h2 className="font-bold text-lg">District Warning</h2>
          <button onClick={() => setSelectedDistrictId(null)} className="p-1 hover-ui rounded">
            ✕
          </button>
        </div>
        <div className="p-4 text-muted">No active warnings for this district.</div>
      </div>
    );
  }

  return (
    <div className="absolute right-0 top-0 bottom-0 z-50 animate-slide-in-right">
      <WarningDetailDrawer warning={warning} onClose={() => setSelectedDistrictId(null)} />
    </div>
  );
}
