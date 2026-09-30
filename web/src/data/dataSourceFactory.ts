/**
 * data/dataSourceFactory.ts
 * Returns the correct DataSource implementation based on VITE_DATA_SOURCE env var.
 * UI components import from here — never directly from mock/ or api/.
 */
import type { DataSource } from './DataSource';

let instance: DataSource | null = null;

export async function getDataSource(): Promise<DataSource> {
  if (instance) return instance;

  const mode = import.meta.env.VITE_DATA_SOURCE ?? 'mock';

  if (mode === 'api') {
    const { ApiDataSource } = await import('./api/ApiDataSource');
    instance = new ApiDataSource();
  } else {
    const { MockDataSource } = await import('./mock/MockDataSource');
    instance = new MockDataSource();
  }

  return instance;
}

/** Synchronous factory — only works after first async call */
export function getDataSourceSync(): DataSource {
  if (!instance) throw new Error('DataSource not yet initialized. Call getDataSource() first.');
  return instance;
}
