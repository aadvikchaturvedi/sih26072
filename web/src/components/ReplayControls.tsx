/**
 * components/ReplayControls.tsx
 * Play/pause, step back/forward, speed, jump to now.
 */
import React, { useEffect, useRef } from 'react';
import { Play, Pause, SkipBack, SkipForward, ChevronsRight } from 'lucide-react';
import { useUIStore } from '@/store/uiStore';
import { getTimeline, useTimeline } from '@/data/timeline';
import type { ReplaySpeed } from '@/store/uiStore';

const SPEEDS: ReplaySpeed[] = [1, 2, 4, 8];
const OBS_FRAME = 8;

export function ReplayControls() {
  const {
    isReplaying, setIsReplaying,
    replayFrame, setReplayFrame, stepFrame,
    replaySpeed, setReplaySpeed,
  } = useUIStore();

  const numFrames = useTimeline().times.length;
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    if (isReplaying) {
      const baseInterval = 1000; // 1 second at 1x
      const ms = baseInterval / replaySpeed;
      intervalRef.current = setInterval(() => {
        setReplayFrame(
          useUIStore.getState().replayFrame < getTimeline().times.length - 1
            ? useUIStore.getState().replayFrame + 1
            : 0,
        );
      }, ms);
    } else {
      if (intervalRef.current) clearInterval(intervalRef.current);
    }
    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [isReplaying, replaySpeed, setReplayFrame]);

  const btnStyle = {
    background: 'var(--raised)',
    border: '1px solid var(--border)',
    color: 'var(--text)',
    borderRadius: 4,
    cursor: 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 4,
    fontSize: 12,
  };
  const btnClass = "min-h-[44px] min-w-[44px] md:min-h-0 md:min-w-0 md:px-2 md:py-1";

  return (
    <div
      className="flex items-center gap-2 px-4 py-2 border-t"
      style={{ background: 'var(--panel)' }}
      role="toolbar"
      aria-label="Replay controls"
    >
      {/* Mode indicator */}
      <span
        className="chip mr-1"
        style={{
          color: isReplaying ? '#22C55E' : 'var(--muted)',
          background: isReplaying ? '#22C55E18' : 'var(--raised)',
          border: `1px solid ${isReplaying ? '#22C55E40' : 'var(--border)'}`,
        }}
      >
        {isReplaying ? '▶ REPLAY' : '⏸ PAUSED'}
      </span>

      {/* Step back */}
      <button
        id="replay-step-back"
        onClick={() => stepFrame(-1, numFrames)}
        className={btnClass}
        style={btnStyle}
        aria-label="Step back one frame"
        title="Step back"
      >
        <SkipBack size={13} />
      </button>

      {/* Play/Pause */}
      <button
        id="replay-play-pause"
        onClick={() => setIsReplaying(!isReplaying)}
        className={btnClass}
        style={{
          ...btnStyle,
          background: isReplaying ? 'var(--accent)' : 'var(--raised)',
          color: isReplaying ? '#fff' : 'var(--text)',
          border: `1px solid ${isReplaying ? 'var(--accent)' : 'var(--border)'}`,
        }}
        aria-label={isReplaying ? 'Pause replay' : 'Play replay'}
      >
        {isReplaying ? <Pause size={13} /> : <Play size={13} />}
      </button>

      {/* Step forward */}
      <button
        id="replay-step-forward"
        onClick={() => stepFrame(1, numFrames)}
        className={btnClass}
        style={btnStyle}
        aria-label="Step forward one frame"
        title="Step forward"
      >
        <SkipForward size={13} />
      </button>

      {/* Speed selector */}
      <div className="flex items-center gap-1 ml-2">
        <span className="label">SPEED</span>
        {SPEEDS.map((s) => (
          <button
            key={s}
            id={`replay-speed-${s}x`}
            onClick={() => setReplaySpeed(s)}
            className={btnClass}
            style={{
              ...btnStyle,
              background: replaySpeed === s ? 'var(--accent)' : 'var(--raised)',
              color: replaySpeed === s ? '#fff' : 'var(--muted)',
              border: `1px solid ${replaySpeed === s ? 'var(--accent)' : 'var(--border)'}`,
              padding: '3px 7px',
              fontSize: 11,
            }}
            aria-pressed={replaySpeed === s}
            aria-label={`${s}x speed`}
          >
            {s}×
          </button>
        ))}
      </div>
    </div>
  );
}
