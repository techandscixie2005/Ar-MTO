#!/usr/bin/env bash
# Locate and download nmrshiftdb2 public data.
# nmrshiftdb2 data may be available via SourceForge or the project website.
# This script locates the public archive and downloads available files.
#
# Usage:
#   bash download_nmrshiftdb2.sh
#
# IMPORTANT: Only download clearly public bulk files.
# Do not attempt to scrape the web interface.

set -euo pipefail

DATASET_ID="nmrshiftdb2"
SOURCEFORGE_URL="https://sourceforge.net/projects/nmrshiftdb2/"
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

{
    echo "=== nmrshiftdb2 Download Log ==="
    echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Hostname: $(hostname)"
    echo "Whoami: $(whoami)"
    echo "PWD: $(pwd)"
    echo ""
} | tee "${LOG_FILE}"

# Step 1: Check SourceForge project page for file releases
echo "[INFO] Checking SourceForge project: ${SOURCEFORGE_URL}" | tee -a "${LOG_FILE}"

# SourceForge file listing API
SF_API="https://sourceforge.net/projects/nmrshiftdb2/files/latest/download"
SF_RSS="https://sourceforge.net/projects/nmrshiftdb2/rss"

echo "[INFO] Fetching SourceForge RSS feed..." | tee -a "${LOG_FILE}"
curl -sL "${SF_RSS}" -o "sourceforge_rss.xml" 2>&1 | tee -a "${LOG_FILE}" || {
    echo "[WARN] Could not fetch SourceForge RSS feed." | tee -a "${LOG_FILE}"
}

# Step 2: Try nmrshiftdb2 direct download endpoints
NMRDB2_BASE="https://nmrshiftdb.nmr.uni-koeln.de"

echo "[INFO] nmrshiftdb2 homepage: ${NMRDB2_BASE}" | tee -a "${LOG_FILE}"
echo "[INFO] Checking for data download links on nmrshiftdb2..." | tee -a "${LOG_FILE}"

# Fetch homepage to find download links
curl -sL "${NMRDB2_BASE}/" -o "nmrshiftdb2_homepage.html" 2>&1 | tee -a "${LOG_FILE}" || {
    echo "[WARN] Could not fetch nmrshiftdb2 homepage." | tee -a "${LOG_FILE}"
}

# Step 3: Check for NMReDATA SD files
# NMReDATA format SD files are the most valuable resource
echo "[INFO] Looking for known download paths..." | tee -a "${LOG_FILE}"

# Common known paths for nmrshiftdb2 data
KNOWN_PATHS=(
    "https://sourceforge.net/projects/nmrshiftdb2/files/latest"
    "${NMRDB2_BASE}/download"
    "${NMRDB2_BASE}/data"
    "${NMRDB2_BASE}/export"
)

for path in "${KNOWN_PATHS[@]}"; do
    echo "[INFO] Trying: ${path}" | tee -a "${LOG_FILE}"
    HTTP_CODE=$(curl -sI -o /dev/null -w '%{http_code}' "${path}" 2>/dev/null || echo "000")
    echo "  HTTP ${HTTP_CODE}" | tee -a "${LOG_FILE}"
done

# Step 4: Report findings
echo "" | tee -a "${LOG_FILE}"
echo "=== nmrshiftdb2 Download Attempt Summary ===" | tee -a "${LOG_FILE}"
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "[STATUS] nmrshiftdb2 public bulk download endpoint not yet confirmed." | tee -a "${LOG_FILE}"
echo "[ACTION] Check the following manually:" | tee -a "${LOG_FILE}"
echo "  1. ${SOURCEFORGE_URL} — SourceForge project files" | tee -a "${LOG_FILE}"
echo "  2. ${NMRDB2_BASE} — nmrshiftdb2 home page" | tee -a "${LOG_FILE}"
echo "  3. Contact nmrshiftdb2 maintainers for bulk data access" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Files in ${DEST_DIR}:" | tee -a "${LOG_FILE}"
ls -lah "${DEST_DIR}" | tee -a "${LOG_FILE}"
