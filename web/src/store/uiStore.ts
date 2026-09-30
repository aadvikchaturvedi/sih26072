/**
 * store/uiStore.ts
 * Zustand store for all UI/app state.
 * Server data lives in TanStack Query — never duplicated here.
 */
import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { WarningLevel } from '@/data/types';

export type Theme = 'dark' | 'light';
export type TimeZone = 'UTC' | 'IST';
export type ReplaySpeed = 1 | 2 | 4 | 8;
export type AppScreen = 'map' | 'warnings' | 'skill' | 'status';

export interface LayerVisibility {
  observedRadar: boolean;
  forecastRadar: boolean;
  lightningStrokes: boolean;
  cellTracks: boolean;
  districtWarnings: boolean;
  districtLabels: boolean;
  lightningProb30: boolean;
  lightningProb60: boolean;
  firstFlash: boolean;
  satelliteTIR: boolean;
}

interface UIState {
  // Theme
  theme: Theme;
  setTheme: (t: Theme) => void;
  toggleTheme: () => void;

  // Timezone display
  timezone: TimeZone;
  toggleTimezone: () => void;

  // Current screen / tab
  screen: AppScreen;
  setScreen: (s: AppScreen) => void;

  // Replay state
  isReplaying: boolean;
  replayFrame: number;
  replaySpeed: ReplaySpeed;
  setReplayFrame: (f: number) => void;
  setReplaySpeed: (s: ReplaySpeed) => void;
  setIsReplaying: (v: boolean) => void;
  stepFrame: (delta: number, maxFrames: number) => void;

  // Selected lead time for forecast layer (minutes: 10..120)
  selectedLead: number;
  setSelectedLead: (l: number) => void;

  // Selected cell (opened CellPanel)
  selectedCellId: string | null;
  setSelectedCellId: (id: string | null) => void;

  // Selected district
  selectedDistrictId: string | null;
  setSelectedDistrictId: (id: string | null) => void;

  // Selected warning
  selectedWarningId: string | null;
  setSelectedWarningId: (id: string | null) => void;

  // Layer visibility
  layers: LayerVisibility;
  setLayerVisible: (key: keyof LayerVisibility, visible: boolean) => void;
  setLayerOpacity: (key: keyof LayerVisibility, opacity: number) => void;

  layerPanelOpen: boolean;
  setLayerPanelOpen: (v: boolean) => void;

  cellPanelOpen: boolean;
  setCellPanelOpen: (v: boolean) => void;

  // Warning filters
  warningLevelFilter: WarningLevel | null;
  setWarningLevelFilter: (l: WarningLevel | null) => void;

  warningDistrictFilter: string;
  setWarningDistrictFilter: (s: string) => void;

  warningShowExpired: boolean;
  setWarningShowExpired: (v: boolean) => void;

  // Mobile district
  mobileDistrict: string | null;
  setMobileDistrict: (d: string | null) => void;

  // Skill page
  csiThreshold: 20 | 35 | 45;
  setCsiThreshold: (t: 20 | 35 | 45) => void;

  dbzThreshold: number;
  setDbzThreshold: (v: number) => void;
}

export const useUIStore = create<UIState>()(
  persist<UIState>(
    (set) => ({
      theme: (typeof window !== 'undefined' && window.matchMedia?.('(prefers-color-scheme: dark)').matches) ? 'dark' : 'light',
      setTheme: (theme: Theme) => set({ theme }),
      toggleTheme: () => set((s: UIState) => ({ theme: s.theme === 'dark' ? 'light' : 'dark' })),

      timezone: 'IST',
      toggleTimezone: () => set((s: UIState) => ({ timezone: s.timezone === 'IST' ? 'UTC' : 'IST' })),

      screen: 'map',
      setScreen: (screen: AppScreen) => set({ screen }),

      isReplaying: false,
      replayFrame: 8,
      replaySpeed: 1,
      setReplayFrame: (replayFrame: number) => set({ replayFrame }),
      setReplaySpeed: (replaySpeed: ReplaySpeed) => set({ replaySpeed }),
      setIsReplaying: (isReplaying: boolean) => set({ isReplaying }),
      stepFrame: (delta: number, maxFrames: number) =>
        set((s: UIState) => ({
          replayFrame: Math.max(0, Math.min(maxFrames - 1, s.replayFrame + delta)),
        })),

      selectedLead: 0,
      setSelectedLead: (selectedLead: number) => set({ selectedLead }),

      selectedCellId: null,
      setSelectedCellId: (selectedCellId: string | null) =>
        set({ selectedCellId, cellPanelOpen: selectedCellId !== null }),

      selectedDistrictId: null,
      setSelectedDistrictId: (selectedDistrictId: string | null) => set({ selectedDistrictId }),

      selectedWarningId: null,
      setSelectedWarningId: (selectedWarningId: string | null) => set({ selectedWarningId }),

      layers: {
        observedRadar: true,
        forecastRadar: true,
        lightningStrokes: true,
        cellTracks: true,
        districtWarnings: true,
        districtLabels: true,
        lightningProb30: false,
        lightningProb60: false,
        firstFlash: false,
        satelliteTIR: false,
      },
      setLayerVisible: (key: keyof LayerVisibility, visible: boolean) =>
        set((s: UIState) => ({ layers: { ...s.layers, [key]: visible } })),
      setLayerOpacity: (_key: keyof LayerVisibility, _opacity: number) => {
        // TODO: per-layer opacity
      },

      layerPanelOpen: true,
      setLayerPanelOpen: (layerPanelOpen: boolean) => set({ layerPanelOpen }),

      cellPanelOpen: false,
      setCellPanelOpen: (cellPanelOpen: boolean) => set({ cellPanelOpen }),

      warningLevelFilter: null,
      setWarningLevelFilter: (warningLevelFilter: WarningLevel | null) => set({ warningLevelFilter }),

      warningDistrictFilter: '',
      setWarningDistrictFilter: (warningDistrictFilter: string) => set({ warningDistrictFilter }),

      warningShowExpired: false,
      setWarningShowExpired: (warningShowExpired: boolean) => set({ warningShowExpired }),

      mobileDistrict: null,
      setMobileDistrict: (mobileDistrict: string | null) => set({ mobileDistrict }),

      csiThreshold: 35,
      setCsiThreshold: (csiThreshold: 20 | 35 | 45) => set({ csiThreshold }),

      dbzThreshold: 35,
      setDbzThreshold: (dbzThreshold: number) => set({ dbzThreshold }),
    }),
    {
      name: 'nowcast-ui-state',
      partialState: (s: UIState) => ({
        theme: s.theme,
        timezone: s.timezone,
        layers: s.layers,
        mobileDistrict: s.mobileDistrict,
        layerPanelOpen: s.layerPanelOpen,
      }),
    } as Parameters<typeof persist<UIState>>[1],
  ),
);
