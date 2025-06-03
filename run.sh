#!/bin/bash

cd api-server

rm -rf .venv
uv venv -p 3.12
uv pip install -e '.'

# Set default port if not provided
SYFTBOX_ASSIGNED_PORT=8000
uv run python main.py
