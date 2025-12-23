#!/bin/bash
# Shell wrapper for uvicorn to ensure proper signal handling
# This allows environment variable expansion while maintaining JSON form for CMD

exec uvicorn main:app \
    --host "${API_HOST}" \
    --port "${API_PORT}" \
    --reload \
    "$@"
