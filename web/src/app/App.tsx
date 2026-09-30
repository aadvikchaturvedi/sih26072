/**
 * app/App.tsx
 * Main application shell: router-free single-page layout.
 * Screen switching via Zustand store.
 */
import React, { useEffect, useState } from 'react';
import { TopBar } from '@/components/TopBar';
import { StatusBar } from '@/components/StatusBar';
import { TimeSlider } from '@/components/TimeSlider';
import { ReplayControls } from '@/components/ReplayControls';
import { LayerPanel } from '@/components/LayerPanel';
import { BottomTabBar } from '@/components/BottomTabBar';
import { MapCanvas } from '@/features/map/MapCanvas';
import { CellPanel } from '@/features/cells/CellPanel';
import { WarningsFeed } from '@/features/warnings/WarningsFeed';
import { SkillPage } from '@/features/skill/SkillPage';
import { StatusPage } from '@/features/status/StatusPage';
import { useUIStore } from '@/store/uiStore';
import { getDataSource } from '@/data/dataSourceFactory';

import { DistrictWarningDrawer } from '@/features/warnings/DistrictWarningDrawer';

function MapView() {
  const { layerPanelOpen, cellPanelOpen } = useUIStore();
  return (
    <div className="flex flex-1 overflow-hidden relative">
      <LayerPanel />
      <div className="flex flex-col flex-1 overflow-hidden">
        <div className="flex flex-1 overflow-hidden">
          <MapCanvas />
          {cellPanelOpen && <CellPanel />}
        </div>
        <TimeSlider />
        <ReplayControls />
      </div>
      <DistrictWarningDrawer />
    </div>
  );
}

export function App() {
  const { screen } = useUIStore();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    getDataSource().then(() => setReady(true));
  }, []);

  if (!ready) {
    return (
      <div
        className="flex items-center justify-center h-full"
        style={{ background: 'var(--bg)', color: 'var(--muted)', fontSize: 14 }}
        aria-live="polite"
      >
        <div className="flex flex-col items-center gap-3">
          <div
            style={{
              width: 32,
              height: 32,
              border: '3px solid var(--border)',
              borderTopColor: 'var(--accent)',
              borderRadius: '50%',
              animation: 'spin 0.8s linear infinite',
            }}
          />
          <span>Initialising Nowcast Console…</span>
        </div>
      </div>
    );
  }

  return (
    <div
      className="flex flex-col h-full"
      style={{ background: 'var(--bg)' }}
    >
      <TopBar />
      <main className="flex flex-1 overflow-hidden" role="main">
        {screen === 'map'      && <MapView />}
        {screen === 'warnings' && <WarningsFeed />}
        {screen === 'skill'    && <SkillPage />}
        {screen === 'status'   && <StatusPage />}
      </main>
      <StatusBar />
      <BottomTabBar />
    </div>
  );
}
