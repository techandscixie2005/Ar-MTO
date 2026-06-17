#!/usr/bin/env bash
# Download QM7-X dataset from Zenodo record 3905361.
# Uses the robust_downloader.py for resumable downloads with retry.
#
# Usage:
#   bash download_qm7x.sh [--sample]
#
# The dataset consists of 8 XZ-compressed HDF5 files + README + createDB.py
# (~9.62 GiB compressed, ~40+ GiB extracted).
# If full download is too large, use --sample to get README + smallest file only.
#
# NOTE: The HPC server (bjhpc_xxy_1) has DNS issues resolving zenodo.org.
# Preferred approach: download locally then rsync to server.

set -euo pipefail

DATASET_ID="qm7x"
ZENODO_RECORD="3905361"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOADER="${SCRIPT_DIR}/robust_downloader.py"

SERVER_DATA_ROOT="${SERVER_DATA_ROOT:-/data/home/scwc008/run/xxy/MTO/data/external}"
LOCAL_FALLBACK_ROOT="${LOCAL_FALLBACK_ROOT:-/mnt/e/Ar-MTO-data-foundry}"
RAW_SUBDIR="raw"
LOG_FILE="download_${DATASET_ID}.log"

SAMPLE_MODE=false
if [ "${1:-}" = "--sample" ]; then
    SAMPLE_MODE=true
fi

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
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "=== QM7-X Download Log ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Hostname: $(hostname)"
echo "Destination: ${DEST_DIR}"
echo "Sample mode: ${SAMPLE_MODE}"
echo ""

# Zenodo file metadata (from record 3905361)
# Format: filename size_bytes md5_hash
declare -A QM7X_FILES
QM7X_FILES=(
    ["README.txt"]="3126|md5:649cb964abef05011939a5dcd380b173"
    ["createDB.py"]="3038|md5:29f0cf5803e5f5568de52ad8045962d3"
    ["8000.xz"]="89426884|md5:c893ae88b8f5c32541c3f024fc1daa45"
    ["1000.xz"]="715361604|md5:b50c6a5d0a4493c274368cf22285503e"
    ["7000.xz"]="1104141872|md5:5ecce00a188410d06b747cb683d8d347"
    ["5000.xz"]="1135294700|md5:85ac444596b87812aaa9e48d203d0b70"
    ["2000.xz"]="1043743376|md5:4418a813daf5e0d44aa5a26544249ee6"
    ["4000.xz"]="1461283500|md5:26819601705ef8c14080fa7fc69decd4"
    ["6000.xz"]="2016065508|md5:787fc4a9036af0e67c034a30adc54c07"
    ["3000.xz"]="2052043036|md5:f7b5aac39a745f11436047c12d1eb24e"
)

# Build download URL
download_url() {
    local filename="$1"
    echo "https://zenodo.org/records/${ZENODO_RECORD}/files/${filename}?download=1"
}

# Sort files by size (smallest first) for progressive download
download_order=(
    "README.txt" "createDB.py" "8000.xz" "1000.xz" "7000.xz"
    "5000.xz" "2000.xz" "4000.xz" "6000.xz" "3000.xz"
)

if [ "${SAMPLE_MODE}" = true ]; then
    download_order=("README.txt" "createDB.py" "8000.xz")
    echo "[SAMPLE] Will download only: ${download_order[*]}"
fi

downloaded=0
failed=0
for filename in "${download_order[@]}"; do
    IFS='|' read -r size md5 <<< "${QM7X_FILES[${filename}]}"
    url=$(download_url "${filename}")

    echo ""
    echo "--- ${filename} (${size} bytes) ---"

    # Check if already downloaded
    if [ -f "${filename}" ]; then
        actual_size=$(stat -c%s "${filename}" 2>/dev/null || stat -f%z "${filename}" 2>/dev/null || echo 0)
        if [ "${actual_size}" = "${size}" ]; then
            echo "[SKIP] Already downloaded, size matches"
            downloaded=$((downloaded + 1))
            continue
        else
            echo "[RESUME] Partial download (${actual_size} / ${size} bytes)"
        fi
    fi

    # Download using robust_downloader.py
    if [ -f "${DOWNLOADER}" ]; then
        python3 "${DOWNLOADER}" \
            --url "${url}" \
            --dest "${filename}" \
            --expected-size "${size}" \
            --expected-md5 "${md5}" \
            --max-retries 10 \
            --timeout 300
        ret=$?
    else
        # Fallback to wget
        echo "[WARN] robust_downloader.py not found, using wget..."
        wget -c -q --show-progress --timeout=300 --tries=10 \
            "${url}" -O "${filename}"
        ret=$?
    fi

    if [ ${ret} -eq 0 ]; then
        echo "[OK] ${filename} downloaded"
        downloaded=$((downloaded + 1))
    else
        echo "[FAIL] ${filename} download failed"
        failed=$((failed + 1))
    fi
done

# Final report
echo ""
echo "=== Download Summary ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Downloaded: ${downloaded} / $((downloaded + failed))"
echo "Files in ${DEST_DIR}:"
ls -lah "${DEST_DIR}"
echo ""
echo "Total size:"
du -sh "${DEST_DIR}"

if [ ${failed} -gt 0 ]; then
    echo ""
    echo "[WARN] ${failed} file(s) failed. Re-run this script to resume."
    exit 1
fi
