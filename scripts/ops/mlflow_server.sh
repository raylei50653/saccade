#!/bin/bash
set -euo pipefail

DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_USER="${DB_USER:-saccade}"
DB_PASSWORD="${DB_PASSWORD:-saccade}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-./storage/mlflow_artifacts}"

BACKEND_URI="postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/mlflow"

echo "Starting MLflow tracking server..."
echo "  Backend:  ${BACKEND_URI}"
echo "  Artifacts: ${ARTIFACT_ROOT}"
echo "  UI:       http://localhost:${MLFLOW_PORT}"
echo ""

exec mlflow server \
    --backend-store-uri "${BACKEND_URI}" \
    --default-artifact-root "${ARTIFACT_ROOT}" \
    --host 0.0.0.0 \
    --port "${MLFLOW_PORT}"
