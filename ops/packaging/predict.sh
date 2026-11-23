#!/bin/sh
set -eu
cd /code
exec python3 /code/predict.py "$@"
