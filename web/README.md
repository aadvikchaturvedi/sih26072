# Nowcast Console (web)

Operator console for the thunderstorm and lightning nowcasting system: map with observed and
forecast radar, lightning, storm-cell tracks and district warnings; warnings feed with CAP
export; skill page; system status.

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # type-check + production build
```

## Data source

All data goes through one interface, `src/data/DataSource.ts`, chosen by `VITE_DATA_SOURCE`
in `.env`:

- `api` (default): the FastAPI backend in `../backend` (`/api/v1/console`). Start it with
  `make demo` in the repository root or in `../backend`. The replay timeline and the map
  extent come from the backend (`GET /timeline`), and a websocket (`/live`) announces new
  forecasts, on which the console extends the timeline and refetches.
- `mock`: the procedural scenario in `src/data/mock`, no backend needed.

Response shapes are defined in `src/data/types.ts` and validated in `src/data/schemas.ts`.

## Note on Boundaries
The district boundaries used in this demo are sourced from community GeoJSON for presentation purposes. In production, these should be replaced by official Survey of India / IMD shapefiles.
