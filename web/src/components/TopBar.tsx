/**
 * components/TopBar.tsx
 * 48px top navigation bar.
 */
import React from 'react';
import { Map, AlertTriangle, BarChart3, Activity, Sun, Moon, Clock } from 'lucide-react';
import { useUIStore } from '@/store/uiStore';
import type { AppScreen } from '@/store/uiStore';
import { formatFullDateTime } from '@/lib/timeHelpers';

const tabs: Array<{ id: AppScreen; label: string; Icon: React.FC<{ size?: number }> }> = [
  { id: 'map',      label: 'Map',      Icon: ({ size }) => <Map size={size} /> },
  { id: 'warnings', label: 'Warnings', Icon: ({ size }) => <AlertTriangle size={size} /> },
  { id: 'skill',    label: 'Skill',    Icon: ({ size }) => <BarChart3 size={size} /> },
  { id: 'status',   label: 'Status',   Icon: ({ size }) => <Activity size={size} /> },
];

function useNow(tz: 'UTC' | 'IST') {
  const [now, setNow] = React.useState(() => new Date().toISOString());
  React.useEffect(() => {
    const id = setInterval(() => setNow(new Date().toISOString()), 1000);
    return () => clearInterval(id);
  }, []);
  return formatFullDateTime(now, tz);
}

export function TopBar() {
  const { screen, setScreen, theme, toggleTheme, timezone, toggleTimezone } = useUIStore();
  const nowStr = useNow(timezone);

  return (
    <header
      className="flex items-center h-12 px-3 gap-3 border-b shrink-0"
      style={{ background: 'var(--panel)' }}
      role="banner"
    >
      {/* Product name */}
      <div className="flex items-center gap-2 min-w-fit">
        <span
          className="font-bold tracking-tight text-sm"
          style={{ color: 'var(--accent)' }}
        >
          Nowcast Console
        </span>
        <span
          className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded"
          style={{ background: 'var(--raised)', color: 'var(--muted)', border: '1px solid var(--border)' }}
        >
          SIH26072
        </span>
        <span
          className="text-[10px] font-mono px-1.5 py-0.5 rounded"
          style={{ background: 'var(--raised)', color: 'var(--muted)', border: '1px solid var(--border)' }}
        >
          Odisha
        </span>
      </div>

      {/* Nav tabs */}
      <nav className="hidden md:flex items-center gap-1 ml-2" role="navigation" aria-label="Main navigation">
        {tabs.map(({ id, label, Icon }) => (
          <button
            key={id}
            id={`nav-tab-${id}`}
            onClick={() => setScreen(id)}
            aria-current={screen === id ? 'page' : undefined}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-semibold transition-ui"
            style={{
              background: screen === id ? 'var(--accent)' : 'transparent',
              color: screen === id ? '#fff' : 'var(--muted)',
            }}
          >
            <Icon size={13} />
            {label}
          </button>
        ))}
      </nav>

      <div className="flex-1" />

      {/* Clock */}
      <button
        id="timezone-toggle"
        onClick={toggleTimezone}
        className="flex items-center gap-1.5 text-xs font-mono px-2 py-1 rounded transition-ui"
        style={{ color: 'var(--muted)', background: 'var(--raised)', border: '1px solid var(--border)' }}
        title="Toggle UTC/IST"
        aria-label={`Current time: ${nowStr}. Click to toggle timezone.`}
      >
        <Clock size={11} />
        {nowStr}
      </button>

      {/* Theme toggle */}
      <button
        id="theme-toggle"
        onClick={toggleTheme}
        className="p-1.5 rounded transition-ui"
        style={{ color: 'var(--muted)', background: 'var(--raised)', border: '1px solid var(--border)' }}
        aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
      >
        {theme === 'dark' ? <Sun size={14} /> : <Moon size={14} />}
      </button>
    </header>
  );
}
