#!/usr/bin/env bash
set -euo pipefail

BUCKET="${S3_BUCKET:-retino-datasets-cezario}"
DATASETS="${DATASETS:-oia-ddr}"
DATA_ROOT="${DATA_ROOT:-/app/data/raw}"

echo "[entrypoint] bucket=$BUCKET"
echo "[entrypoint] datasets=$DATASETS"
echo "[entrypoint] data_root=$DATA_ROOT"

mkdir -p "$DATA_ROOT"

for ds in $DATASETS; do
    local_path="$DATA_ROOT/$ds"
    marker="$local_path/.sync_complete"

    if [ -f "$marker" ]; then
        echo "[entrypoint] $ds já sincronizado (marker presente), pulando"
        continue
    fi

    echo "[entrypoint] sincronizando $ds de s3://$BUCKET/raw/$ds/ ..."
    mkdir -p "$local_path"
    aws s3 sync "s3://$BUCKET/raw/$ds/" "$local_path/"
    touch "$marker"
    echo "[entrypoint] $ds sincronizado"
done

echo "[entrypoint] iniciando: $*"
exec "$@"
