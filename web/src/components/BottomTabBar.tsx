import React from 'react';
import { Map, AlertTriangle, BarChart3, Activity } from 'lucide-react';
import { useUIStore } from '@/store/uiStore';
import type { AppScreen } from '@/store/uiStore';

const tabs: Array<{ id: AppScreen; label: string; Icon: React.FC<{ size?: number }> }> = [
  { id: 'map',      label: 'Map',      Icon: ({ size }) => <Map size={size} /> },
  { id: 'warnings', label: 'Warnings', Icon: ({ size }) => <AlertTriangle size={size} /> },
  { id: 'skill',    label: 'Skill',    Icon: ({ size }) => <BarChart3 size={size} /> },
  { id: 'status',   label: 'Status',   Icon: ({ size }) => <Activity size={size} /> },
];

export function BottomTabBar() {
  const { screen, setScreen } = useUIStore();

  return (
    <nav 
      className="md:hidden flex items-center justify-around h-14 border-t shrink-0 pb-safe"
      style={{ background: 'var(--panel)' }}
      role="navigation" 
      aria-label="Mobile navigation"
    >
      {tabs.map(({ id, label, Icon }) => (
        <button
          key={id}
          onClick={() => setScreen(id)}
          aria-current={screen === id ? 'page' : undefined}
          className="flex flex-col items-center justify-center gap-1 w-full h-full text-[10px] font-semibold transition-ui"
          style={{
            color: screen === id ? 'var(--accent)' : 'var(--muted)',
          }}
        >
          <Icon size={18} />
          {label}
        </button>
      ))}
    </nav>
  );
}
