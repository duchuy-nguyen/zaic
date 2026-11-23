#!/bin/sh
set -eu
cd /code
exec jupyter lab --port 9777 --ip 0.0.0.0 --ServerApp.token=zac2026 --allow-root --no-browser
