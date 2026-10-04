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
import { setTimeline } from '@/data/timeline';
import { useQueryClient } from '@tanstack/react-query';

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
  const [error, setError] = useState<string | null>(null);
  const queryClient = useQueryClient();

  useEffect(() => {
    let unsubscribe = () => {};
    let cancelled = false;
    getDataSource()
      .then((ds) => {
        if (cancelled) return;
        setReady(true);
        if (!ds.live) return;
        // A new forecast arrived: extend the timeline and refetch what is on screen.
        unsubscribe = ds.subscribe(() => {
          ds.getTimeline().then(setTimeline).catch(() => {});
          queryClient.invalidateQueries();
        });
      })
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [queryClient]);

  if (error) {
    return (
      <div
        className="flex items-center justify-center h-full"
        style={{ background: 'var(--bg)', color: 'var(--text)', fontSize: 14 }}
        role="alert"
      >
        <div className="flex flex-col items-center gap-2 text-center" style={{ maxWidth: 460 }}>
          <span className="font-bold">Cannot reach the nowcast backend</span>
          <span className="text-muted text-xs">{error}</span>
          <span className="text-muted text-xs">
            Start it with <code>make demo</code> in the repository root, or set
            VITE_DATA_SOURCE=mock in web/.env to use the built-in mock scenario.
          </span>
          <button
            onClick={() => window.location.reload()}
            className="px-3 py-1 rounded text-xs mt-2"
            style={{ background: 'var(--raised)', border: '1px solid var(--border)', color: 'var(--text)' }}
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

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
