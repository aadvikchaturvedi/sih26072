/**
 * data/api/ApiDataSource.ts
 * DataSource implementation that connects to a live FastAPI backend.
 * Switch via VITE_DATA_SOURCE=api and VITE_API_BASE_URL.
 * UI components never import this directly — always use the factory.
 */
import type { DataSource } from '../DataSource';
import type {
  ForecastResponse, Cell, Warning, SkillReport, HealthStatus,
  LiveTick, LightningStroke, DistrictForecast,
} from '../types';
import {
  ForecastResponseSchema, CellSchema, WarningSchema, SkillReportSchema,
  HealthStatusSchema, LiveTickSchema, LightningStrokeSchema, DistrictForecastSchema,
  TimelineSchema,
} from '../schemas';
import type { Timeline } from '../timeline';
import { z } from 'zod';

const BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000/api/v1/console';
const WS_URL = import.meta.env.VITE_WS_URL ?? `${BASE.replace(/^http/, 'ws')}/live`;

async function fetchJson<T>(schema: z.ZodSchema<T>, url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status} from ${url}`);
  const data = await res.json();
  return schema.parse(data);
}

export class ApiDataSource implements DataSource {
  readonly live = true;

  async getTimeline(): Promise<Timeline> {
    return fetchJson(TimelineSchema, `${BASE}/timeline`);
  }

  getObservedRadarUrl(t: string): string {
    return `${BASE}/radar/${encodeURIComponent(t)}`;
  }

  async getForecast(t0: string): Promise<ForecastResponse> {
    return fetchJson(ForecastResponseSchema, `${BASE}/forecast/${encodeURIComponent(t0)}`);
  }

  async getCells(t0: string): Promise<Cell[]> {
    return fetchJson(z.array(CellSchema), `${BASE}/cells/${encodeURIComponent(t0)}`);
  }

  async getLightningStrokes(t0: string): Promise<LightningStroke[]> {
    return fetchJson(z.array(LightningStrokeSchema), `${BASE}/lightning/${encodeURIComponent(t0)}`);
  }

  async getDistricts(t0: string): Promise<DistrictForecast[]> {
    return fetchJson(z.array(DistrictForecastSchema), `${BASE}/districts/${encodeURIComponent(t0)}`);
  }

  async getWarnings(t0: string): Promise<Warning[]> {
    return fetchJson(z.array(WarningSchema), `${BASE}/warnings/${encodeURIComponent(t0)}`);
  }

  async getSkill(): Promise<SkillReport> {
    return fetchJson(SkillReportSchema, `${BASE}/skill`);
  }

  async getHealth(t0?: string): Promise<HealthStatus> {
    const url = t0 ? `${BASE}/health/${encodeURIComponent(t0)}` : `${BASE}/health`;
    return fetchJson(HealthStatusSchema, url);
  }

  async exportCap(warningId: string): Promise<{ xml: string; json: object }> {
    const res = await fetch(`${BASE}/cap/${encodeURIComponent(warningId)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status} fetching CAP for ${warningId}`);
    return res.json() as Promise<{ xml: string; json: object }>;
  }

  subscribe(onTick: (msg: LiveTick) => void): () => void {
    let ws: WebSocket | null = null;
    let reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
    let closed = false;

    const connect = () => {
      ws = new WebSocket(WS_URL);
      ws.onmessage = (event) => {
        try {
          const raw = JSON.parse(event.data as string);
          const tick = LiveTickSchema.parse(raw);
          onTick(tick);
        } catch (e) {
          console.warn('Invalid live tick:', e);
        }
      };
      ws.onclose = () => {
        if (!closed) reconnectTimeout = setTimeout(connect, 3000);
      };
    };

    connect();

    return () => {
      closed = true;
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      ws?.close();
    };
  }
}
