/**
 * features/map/MapCanvas.tsx
 * MapLibre GL JS canvas with deck.gl overlay for cell tracks,
 * lightning strokes, district polygons.
 *
 * Constraints:
 * - 2D only (pitch 0, rotation disabled)
 * - Locally bundled GeoJSON basemap (no external tiles)
 * - All data from TanStack Query hooks
 */
import React, { useEffect, useRef, useState } from 'react';
import * as maplibregl from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { MapLibreOverlay } from '@deck.gl/maplibre';
import { ScatterplotLayer, PathLayer, TextLayer } from '@deck.gl/layers';
import { PathStyleExtension } from '@deck.gl/extensions';
import { useUIStore } from '@/store/uiStore';
import { useCells, useForecast, useLightningStrokes, useDistricts } from '@/data/hooks';
import { getDataSourceSync } from '@/data/dataSourceFactory';
import { addMinutes, parseISO } from 'date-fns';
import { getTimeline } from '@/data/timeline';

/** District fill colours. MapLibre paint properties cannot resolve CSS variables. */
const LEVEL_HEX: Record<string, string> = {
  red: '#EF4444',
  orange: '#F97316',
  yellow: '#EAB308',
  green: '#22C55E',
};
import type { Cell, DistrictForecast } from '@/data/types';
import odishaDistricts from '@/assets/geo/odisha_districts.json';

interface TooltipState {
  x: number;
  y: number;
  district: DistrictForecast;
}

const MAPLIBRE_STYLE: maplibregl.StyleSpecification = {
  version: 8,
  name: 'Nowcast-Local',
  glyphs: 'https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf',
  sources: {},
  layers: [
    {
      id: 'background',
      type: 'background',
      paint: { 'background-color': '#0E1217' },
    },
  ],
};

export function MapCanvas() {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const overlayRef = useRef<MapLibreOverlay | null>(null);
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  // The data effects below need the style's sources; re-run them once the map has loaded.
  const [mapLoaded, setMapLoaded] = useState(false);

  const {
    replayFrame, layers, selectedCellId, setSelectedCellId,
    selectedDistrictId, setSelectedDistrictId,
    layerPanelOpen, selectedLead
  } = useUIStore();

  const { data: cells } = useCells(replayFrame);
  const { data: forecast } = useForecast(replayFrame);
  const { data: strokes } = useLightningStrokes(replayFrame);
  const { data: districts } = useDistricts(replayFrame);

  // Initialize map
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: mapContainerRef.current,
      style: MAPLIBRE_STYLE,
      center: [86.0, 21.0],
      zoom: 6.8,
      minZoom: 5,
      maxZoom: 12,
      maxPitch: 0,
      dragRotate: false,
      touchZoomRotate: false,
      attributionControl: false,
    });

    mapRef.current = map;
    // Extent the raster overlays are stretched over (fixed for the session).
    const DOMAIN_BBOX = getTimeline().bbox;

    overlayRef.current = new MapLibreOverlay({
      interleaved: true,
      layers: []
    });
    map.addControl(overlayRef.current as any);

    map.on('load', () => {
      // Add district boundaries GeoJSON source
      map.addSource('districts', {
        type: 'geojson',
        data: odishaDistricts as maplibregl.GeoJSONSourceSpecification['data'],
      });

      // District fill layer (warnings)
      map.addLayer({
        id: 'district-fills',
        type: 'fill',
        source: 'districts',
        paint: {
          'fill-color': '#22C55E',
          'fill-opacity': 0.25,
        },
      });

      // District outlines
      map.addLayer({
        id: 'district-outlines',
        type: 'line',
        source: 'districts',
        paint: {
          'line-color': '#263140',
          'line-width': 1.5,
        },
      });

      // District labels
      map.addLayer({
        id: 'district-labels',
        type: 'symbol',
        source: 'districts',
        layout: {
          'text-field': ['get', 'name'],
          'text-size': 11,
          'text-font': ['Open Sans Regular'],
        },
        paint: {
          'text-color': '#E6EBF1',
          'text-halo-color': '#151B22',
          'text-halo-width': 2,
        },
      });

      const EMPTY_IMAGE = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';

      // Radar image source (placeholder - updated per frame)
      map.addSource('radar', {
        type: 'image',
        url: EMPTY_IMAGE,
        coordinates: [
          [DOMAIN_BBOX.west, DOMAIN_BBOX.north],
          [DOMAIN_BBOX.east, DOMAIN_BBOX.north],
          [DOMAIN_BBOX.east, DOMAIN_BBOX.south],
          [DOMAIN_BBOX.west, DOMAIN_BBOX.south],
        ],
      });

      map.addLayer({
        id: 'radar-layer',
        type: 'raster',
        source: 'radar',
        paint: { 'raster-opacity': 0.8 },
      });

      // Probability layers
      map.addSource('prob30', {
        type: 'image',
        url: EMPTY_IMAGE,
        coordinates: [
          [DOMAIN_BBOX.west, DOMAIN_BBOX.north],
          [DOMAIN_BBOX.east, DOMAIN_BBOX.north],
          [DOMAIN_BBOX.east, DOMAIN_BBOX.south],
          [DOMAIN_BBOX.west, DOMAIN_BBOX.south],
        ],
      });

      map.addLayer({
        id: 'prob30-layer',
        type: 'raster',
        source: 'prob30',
        paint: { 'raster-opacity': 0.7 },
        layout: { visibility: 'none' },
      });

      map.addSource('prob60', {
        type: 'image',
        url: EMPTY_IMAGE,
        coordinates: [
          [DOMAIN_BBOX.west, DOMAIN_BBOX.north],
          [DOMAIN_BBOX.east, DOMAIN_BBOX.north],
          [DOMAIN_BBOX.east, DOMAIN_BBOX.south],
          [DOMAIN_BBOX.west, DOMAIN_BBOX.south],
        ],
      });

      map.addLayer({
        id: 'prob60-layer',
        type: 'raster',
        source: 'prob60',
        paint: { 'raster-opacity': 0.7 },
        layout: { visibility: 'none' },
      });

      map.addSource('first-flash', {
        type: 'image',
        url: EMPTY_IMAGE,
        coordinates: [
          [DOMAIN_BBOX.west, DOMAIN_BBOX.north],
          [DOMAIN_BBOX.east, DOMAIN_BBOX.north],
          [DOMAIN_BBOX.east, DOMAIN_BBOX.south],
          [DOMAIN_BBOX.west, DOMAIN_BBOX.south],
        ],
      });

      map.addLayer({
        id: 'first-flash-layer',
        type: 'raster',
        source: 'first-flash',
        paint: {
          'raster-opacity': 0.8,
          // Hatched styling requires mapbox custom shaders which maplibre supports via raster-color
          // But since it's an image overlay, the image itself should ideally have the hatched pattern
          // We will render it as is for now.
        },
        layout: { visibility: 'none' },
      });

      // District hover + click
      map.on('mousemove', 'district-fills', (e) => {
        if (e.features && e.features.length > 0) {
          map.getCanvas().style.cursor = 'pointer';
        }
      });

      map.on('mouseleave', 'district-fills', () => {
      map.getCanvas().style.cursor = '';
      });

      map.on('click', 'district-fills', (e) => {
        if (e.features && e.features.length > 0) {
        const props = e.features[0].properties as { id: string; name: string } | null;
        if (props) setSelectedDistrictId(props.id);
        }
      });

      setMapLoaded(true);
    });

    // Scale bar
    map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-right');
    // Navigation (zoom only)
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'top-right');

    return () => {
      map.remove();
      mapRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Update radar image source
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded || !forecast) return;

    try {
      const src = map.getSource('radar') as maplibregl.ImageSource | undefined;
      if (src && layers.observedRadar) {
        let url = '';
        if (selectedLead <= 0) {
          const t = addMinutes(parseISO(forecast.attrs.t0), selectedLead).toISOString().replace('.000', '');
          url = getDataSourceSync().getObservedRadarUrl(t);
        } else {
          url = forecast.reflectivityUrlByLead[selectedLead] ?? '';
        }
        if (url) src.updateImage({ url });
      }

      const layer = map.getLayer('radar-layer');
      if (layer) {
        map.setLayoutProperty('radar-layer', 'visibility',
          layers.observedRadar ? 'visible' : 'none');
      }

      // Probability layers
      const src30 = map.getSource('prob30') as maplibregl.ImageSource | undefined;
      if (src30 && forecast.lightningProb30Url) {
        src30.updateImage({ url: forecast.lightningProb30Url });
      }
      map.setLayoutProperty('prob30-layer', 'visibility',
        layers.lightningProb30 ? 'visible' : 'none');

      const src60 = map.getSource('prob60') as maplibregl.ImageSource | undefined;
      if (src60 && forecast.lightningProb60Url) {
        src60.updateImage({ url: forecast.lightningProb60Url });
      }
      map.setLayoutProperty('prob60-layer', 'visibility',
        layers.lightningProb60 ? 'visible' : 'none');

      const srcFirstFlash = map.getSource('first-flash') as maplibregl.ImageSource | undefined;
      if (srcFirstFlash && forecast.firstFlashUrl) {
        srcFirstFlash.updateImage({ url: forecast.firstFlashUrl });
      }
      map.setLayoutProperty('first-flash-layer', 'visibility',
        layers.firstFlash ? 'visible' : 'none');
    } catch (_e) {
      // Map may not be fully loaded yet
    }
  }, [mapLoaded, forecast, layers.observedRadar, layers.lightningProb30, layers.lightningProb60, layers.firstFlash, selectedLead]);

  // Update district warning colors
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoaded || !districts) return;

    try {
      if (!map.getSource('districts') || !map.getLayer('district-fills')) return;

      const colorEntries = districts.flatMap((d) => [d.districtId, LEVEL_HEX[d.warningLevel]]);
      const colorMap = ['match', ['get', 'id'], ...colorEntries, '#22C55E'];

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      map.setPaintProperty('district-fills', 'fill-color', colorMap as any);

      map.setPaintProperty('district-fills', 'fill-opacity',
        layers.districtWarnings ? 0.3 : 0);
      map.setPaintProperty('district-outlines', 'line-opacity',
        layers.districtWarnings ? 1 : 0);
      map.setLayoutProperty('district-labels', 'visibility',
        layers.districtLabels ? 'visible' : 'none');
    } catch (_e) {
      // ignore
    }
  }, [mapLoaded, districts, layers.districtWarnings, layers.districtLabels]);


  // Update deck.gl overlay
  useEffect(() => {
    if (!overlayRef.current) return;

    const deckLayers = [];

    if (layers.lightningStrokes && strokes) {
      deckLayers.push(
        new ScatterplotLayer({
          id: 'strokes',
          data: strokes,
          getPosition: (d: any) => [d.lon, d.lat],
          getFillColor: [255, 200, 0, 200],
          getRadius: 1000,
          radiusUnits: 'meters',
        })
      );
    }

    if (layers.cellTracks && cells) {
      // Cell markers
      deckLayers.push(
        new ScatterplotLayer({
          id: 'cells',
          data: cells,
          getPosition: (d: any) => [d.lon, d.lat],
          getFillColor: (d: any): [number, number, number, number] => {
            const color = d.lightningJumpFlag ? [239, 68, 68] :
                          d.status === 'growing' ? [249, 115, 22] :
                          d.status === 'mature' ? [234, 179, 8] : [34, 197, 94];
            return [...color, 40] as [number, number, number, number];
          },
          getLineColor: (d: any): [number, number, number, number] => {
            if (d.id === selectedCellId) return [255, 255, 255, 255];
            const color = d.lightningJumpFlag ? [239, 68, 68] :
                          d.status === 'growing' ? [249, 115, 22] :
                          d.status === 'mature' ? [234, 179, 8] : [34, 197, 94];
            return [...color, 255] as [number, number, number, number];
          },
          getLineWidth: (d: any) => d.id === selectedCellId ? 3 : 2,
          lineWidthUnits: 'pixels',
          getRadius: 24,
          radiusUnits: 'pixels',
          stroked: true,
          pickable: true,
          onClick: (info: any) => {
            if (info.object) setSelectedCellId(info.object.id);
          }
        }),
        new TextLayer({
          id: 'cell-labels',
          data: cells,
          getPosition: (d: any) => [d.lon, d.lat],
          getText: (d: any) => d.id.toString(),
          getSize: 12,
          getColor: [255, 255, 255, 255],
          fontFamily: 'var(--font-mono)'
        }),
        new PathLayer({
          id: 'cell-paths',
          data: cells,
          getPath: (d: any) => [[d.lon, d.lat], ...d.forecastPath.map((p: any) => [p.lon, p.lat])],
          getColor: [255, 255, 255, 120],
          getWidth: 2,
          widthUnits: 'pixels',
          getDashArray: [4, 4],
          dashJustified: true,
          extensions: [new PathStyleExtension({dash: true})]
        }),
        new ScatterplotLayer({
          id: 'cell-path-nodes',
          data: cells.flatMap((c: any) => c.forecastPath),
          getPosition: (d: any) => [d.lon, d.lat],
          getFillColor: [255, 255, 255, 255],
          getRadius: 3,
          radiusUnits: 'pixels',
        }),
        new TextLayer({
          id: 'cell-path-labels',
          data: cells.flatMap((c: any) => c.forecastPath),
          getPosition: (d: any) => [d.lon, d.lat],
          getText: (d: any) => `+${d.leadMin}m`,
          getSize: 10,
          getColor: [255, 255, 255, 200],
          getPixelOffset: [0, -12],
          fontFamily: "'JetBrains Mono', monospace"
        })
      );
    }

    overlayRef.current.setProps({ layers: deckLayers });
  }, [cells, strokes, layers.lightningStrokes, layers.cellTracks, selectedCellId, setSelectedCellId]);

  return (
    <div className="relative flex-1 overflow-hidden" style={{ background: 'var(--bg)' }}>
      <div
        ref={mapContainerRef}
        style={{ width: '100%', height: '100%' }}
        aria-label="Nowcast map - Odisha region"
        role="img"
      />

      {/* District summary strip */}
      <DistrictSummaryStrip districts={districts} />

      {/* Tooltip */}
      {tooltip && (
        <div
          className="absolute pointer-events-none"
          style={{
            left: tooltip.x + 12,
            top: tooltip.y - 60,
            background: 'var(--raised)',
            border: '1px solid var(--border)',
            borderRadius: 6,
            padding: '8px 12px',
            fontSize: 12,
            zIndex: 100,
          }}
        >
          <div className="font-bold">{tooltip.district.name}</div>
          <div className="text-muted text-xs">
            P(30m): {Math.round(tooltip.district.lightningProb30 * 100)}% |
            P(60m): {Math.round(tooltip.district.lightningProb60 * 100)}% |
            Max dBZ: {Math.round(tooltip.district.maxDbz)}
          </div>
        </div>
      )}

      {/* dBZ legend */}
      <DbzLegend />
    </div>
  );
}

function DistrictSummaryStrip({ districts }: { districts: DistrictForecast[] | undefined }) {
  const { setWarningLevelFilter, warningLevelFilter } = useUIStore();

  if (!districts) return null;

  const counts = {
    red: districts.filter((d) => d.warningLevel === 'red').length,
    orange: districts.filter((d) => d.warningLevel === 'orange').length,
    yellow: districts.filter((d) => d.warningLevel === 'yellow').length,
    green: districts.filter((d) => d.warningLevel === 'green').length,
  };

  const levels = [
    { level: 'red' as const, label: 'RED', color: 'var(--warn-red)' },
    { level: 'orange' as const, label: 'ORANGE', color: 'var(--warn-orange)' },
    { level: 'yellow' as const, label: 'YELLOW', color: 'var(--warn-yellow)' },
    { level: 'green' as const, label: 'GREEN', color: 'var(--warn-green)' },
  ];

  return (
    <div
      className="absolute top-3 left-3 flex gap-2 items-center"
      style={{ zIndex: 50 }}
      role="status"
      aria-live="polite"
      aria-label="District warning summary"
    >
      {levels.map(({ level, label, color }) => (
        <button
          key={level}
          id={`district-filter-${level}`}
          onClick={() => setWarningLevelFilter(warningLevelFilter === level ? null : level)}
          className="chip transition-ui"
          style={{
            color,
            background: warningLevelFilter === level ? `${color}20` : 'var(--panel)',
            border: `1px solid ${warningLevelFilter === level ? color : 'var(--border)'}`,
            opacity: counts[level] === 0 ? 0.4 : 1,
          }}
          aria-pressed={warningLevelFilter === level}
          title={`Filter to ${label} districts (${counts[level]})`}
        >
          <span style={{ fontWeight: 800 }}>{counts[level]}</span>
          <span style={{ fontSize: 9 }}>{label}</span>
        </button>
      ))}
    </div>
  );
}

function DbzLegend() {
  const entries = [
    { dbz: '15-20', color: '#B4F0FF' },
    { dbz: '20-25', color: '#64C8FF' },
    { dbz: '25-30', color: '#00FF64' },
    { dbz: '30-35', color: '#50DC00' },
    { dbz: '35-40', color: '#FFE600' },
    { dbz: '40-45', color: '#FF8C00' },
    { dbz: '45-50', color: '#FF2828' },
    { dbz: '50-55', color: '#C800C8' },
    { dbz: '55+',   color: '#8C008C' },
  ];

  return (
    <div
      className="absolute bottom-12 right-12 flex flex-col gap-px"
      style={{
        background: 'var(--panel)',
        border: '1px solid var(--border)',
        borderRadius: 6,
        padding: '6px 8px',
        zIndex: 50,
        fontSize: 10,
      }}
      role="img"
      aria-label="dBZ colorscale legend"
    >
      <div className="label mb-1">dBZ</div>
      {entries.map((e) => (
        <div key={e.dbz} className="flex items-center gap-2">
          <div style={{ width: 14, height: 8, background: e.color, borderRadius: 2 }} />
          <span className="mono text-muted">{e.dbz}</span>
        </div>
      ))}
    </div>
  );
}
