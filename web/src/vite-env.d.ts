// vite-env.d.ts - augment for GeoJSON imports
/// <reference types="vite/client" />

declare module '*.geojson' {
  const value: object;
  export default value;
}
