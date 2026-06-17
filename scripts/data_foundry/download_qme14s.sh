#!/usr/bin/env bash
# Download QMe14S dataset from Figshare.
# Uses wget with resume support. Falls back to curl if wget unavailable.
#
# Usage:
#   bash download_qme14s.sh
#
# Server download is preferred. If server cannot download, use local fallback.

set -euo pipefail

# Configuration — override via environment variables
DATASET_ID="qme14s"
FIGSHARE_URL="https://figshare.com/s/889262a4e999b5c9a5b3"
SERVER_DATA_ROOT="${SERVER_DATA_ROOT:-/data/home/scwc008/run/xxy/MTO/data/external}"
LOCAL_FALLBACK_ROOT="${LOCAL_FALLBACK_ROOT:-/mnt/e/Ar-MTO-data-foundry}"
RAW_SUBDIR="raw"
LOG_FILE="download_${DATASET_ID}.log"

# Detect environment
if [ -d "${SERVER_DATA_ROOT}" ]; then
    DEST_DIR="${SERVER_DATA_ROOT}/${DATASET_ID}/${RAW_SUBDIR}"
    echo "[INFO] Running on server. Destination: ${DEST_DIR}"
else
    DEST_DIR="${LOCAL_FALLBACK_ROOT}/${DATASET_ID}"
    echo "[INFO] Running locally. Destination: ${DEST_DIR}"
fi

mkdir -p "${DEST_DIR}"
cd "${DEST_DIR}"

# Log start
{
    echo "=== QMe14S Download Log ==="
    echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Hostname: $(hostname)"
    echo "Whoami: $(whoami)"
    echo "PWD: $(pwd)"
    echo "Destination: ${DEST_DIR}"
    echo ""
} | tee "${LOG_FILE}"

# Step 1: Fetch Figshare article metadata to get file list
echo "[INFO] Fetching Figshare article metadata..." | tee -a "${LOG_FILE}"

FIGSHARE_API="https://api.figshare.com/v2/articles"

# The Figshare URL is a collection; resolve to article IDs
# For the URL pattern https://figshare.com/s/889262a4e999b5c9a5b3,
# we need to resolve the private sharing link
# Try curl to follow redirects and find actual article
ARTICLE_URL=$(curl -sI -L -o /dev/null -w '%{url_effective}' "${FIGSHARE_URL}" 2>/dev/null || echo "")
echo "[INFO] Resolved URL: ${ARTICLE_URL}" | tee -a "${LOG_FILE}"

# Extract article ID from URL if possible
if [[ "${ARTICLE_URL}" =~ articles/([0-9]+) ]]; then
    ARTICLE_ID="${BASH_REMATCH[1]}"
    echo "[INFO] Article ID: ${ARTICLE_ID}" | tee -a "${LOG_FILE}"

    # Fetch article details
    curl -sL "https://api.figshare.com/v2/articles/${ARTICLE_ID}" \
        -o "figshare_article_${ARTICLE_ID}.json" 2>&1 | tee -a "${LOG_FILE}"
    echo "[INFO] Saved article metadata to figshare_article_${ARTICLE_ID}.json" | tee -a "${LOG_FILE}"

    # List files
    curl -sL "https://api.figshare.com/v2/articles/${ARTICLE_ID}/files" \
        -o "figshare_files_${ARTICLE_ID}.json" 2>&1 | tee -a "${LOG_FILE}"
    echo "[INFO] Saved file list to figshare_files_${ARTICLE_ID}.json" | tee -a "${LOG_FILE}"

    # Download each file
    if command -v python3 &>/dev/null; then
        python3 -c "
import json, subprocess, sys
with open('figshare_files_${ARTICLE_ID}.json') as f:
    files = json.load(f)
for f in files:
    name = f['name']
    url = f['download_url']
    size = f.get('size', 0)
    print(f'Downloading {name} ({size} bytes)...')
    # Use wget -c for resume
    ret = subprocess.run(['wget', '-c', '-q', '--show-progress', url, '-O', name])
    if ret.returncode != 0:
        print(f'wget failed for {name}, trying curl...')
        ret = subprocess.run(['curl', '-L', '-C', '-', '-o', name, url])
    if ret.returncode == 0:
        print(f'  Done: {name}')
    else:
        print(f'  FAILED: {name}', file=sys.stderr)
" 2>&1 | tee -a "${LOG_FILE}"
    else
        echo "[WARN] python3 not available, please download files manually from: ${FIGSHARE_URL}" | tee -a "${LOG_FILE}"
        echo "[INFO] File list saved to figshare_files_${ARTICLE_ID}.json" | tee -a "${LOG_FILE}"
    fi
else
    echo "[WARN] Could not resolve Figshare article ID." | tee -a "${LOG_FILE}"
    echo "[INFO] URL: ${FIGSHARE_URL}" | tee -a "${LOG_FILE}"
    echo "[INFO] Please download manually and place files in: ${DEST_DIR}" | tee -a "${LOG_FILE}"
fi

# Final report
echo "" | tee -a "${LOG_FILE}"
echo "=== Download Summary ===" | tee -a "${LOG_FILE}"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG_FILE}"
echo "Files in ${DEST_DIR}:" | tee -a "${LOG_FILE}"
ls -lah "${DEST_DIR}" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Total size:" | tee -a "${LOG_FILE}"
du -sh "${DEST_DIR}" | tee -a "${LOG_FILE}"
