# Thunderstorm & lightning nowcasting (SIH 2026, PS 26072): ml/ + backend/ + web/
.PHONY: install demo backend web test lint

install:
	$(MAKE) -C ml install
	$(MAKE) -C backend install
	cd web && npm install

# The complete application on dummy data. The backend trains a small model on synthetic
# storms the first time (~2 min) and replays a synthetic afternoon over Odisha; the web
# console opens on http://localhost:5173. Ctrl-C stops both.
demo:
	@trap 'kill 0' INT TERM EXIT; \
	$(MAKE) -C backend demo & \
	(cd web && npm run dev) & \
	wait

backend:
	$(MAKE) -C backend demo

web:
	cd web && npm run dev

test:
	$(MAKE) -C ml test
	$(MAKE) -C backend test
	cd web && npx tsc -b

lint:
	$(MAKE) -C ml lint
	$(MAKE) -C backend lint
	cd web && npm run lint
