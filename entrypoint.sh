#!/usr/bin/env bash
set -euo pipefail

BUCKET="${S3_BUCKET:-retino-datasets-cezario}"
DATASETS="${DATASETS:-oia-ddr}"
DATA_ROOT="${DATA_ROOT:-/app/data/raw}"

echo "[entrypoint] bucket=$BUCKET"
echo "[entrypoint] datasets=$DATASETS"
echo "[entrypoint] data_root=$DATA_ROOT"

mkdir -p "$DATA_ROOT"

# 1) Sincroniza o(s) dataset(s) do S3 (pulando o que já foi baixado).
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

# 2) Converte as anotações do DDR (Pascal VOC -> YOLO) se ainda não foi feito.
#    Gera o pool data/yolo/ consumido pelo treino. Idempotente via manifest.csv.
if [ ! -f /app/data/yolo/manifest.csv ]; then
    echo "[entrypoint] convertendo anotações (scripts/01_convert_annotations.py) ..."
    python scripts/01_convert_annotations.py
else
    echo "[entrypoint] pool YOLO já existe (data/yolo/manifest.csv), pulando conversão"
fi

echo "[entrypoint] iniciando: $*"
exec "$@"
