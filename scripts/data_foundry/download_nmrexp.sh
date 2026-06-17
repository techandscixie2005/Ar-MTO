#!/usr/bin/env bash
# Download NMRexp dataset from Zenodo record 17296666.
# Uses the robust_downloader.py for resumable downloads with retry.
#
# Usage:
#   bash download_nmrexp.sh [--sample]
#
# Dataset: 3.37 million experimental NMR spectra (1H, 13C, 19F, 31P, 11B, 29Si).
# Total: ~3.3 GB (10 files).
# Paper: DOI: 10.1038/s41597-025-06245-5
# Zenodo: https://zenodo.org/records/17296666
# License: CC BY 4.0
#
# NOTE: HPC server has DNS issues resolving zenodo.org.
# Preferred: download locally then rsync to server.
# Small validation files (<1 MB each) can be downloaded directly on HPC.

set -euo pipefail

DATASET_ID="nmrexp"
ZENODO_RECORD="17296666"
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

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "=== NMRexp Download Log ==="
echo "Timestamp: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "Hostname: $(hostname)"
echo "Destination: ${DEST_DIR}"
echo "Sample mode: ${SAMPLE_MODE}"
echo "Zenodo record: ${ZENODO_RECORD}"
echo ""

# File metadata from Zenodo record 17296666
# Format: filename size_bytes md5_hash
declare -A NMREXP_FILES
NMREXP_FILES=(
    ["NMRexp_10to24_1_0811.py"]="218133|md5:37e7212e578ee8039dfbea9e071ef486"
    ["F_50_checked.csv"]="20152|md5:2215638aa53c5ac10a3ce10afadfdf09"
    ["hetero_200_checked.csv"]="73922|md5:612f248f288b59b52a96806cde4d3c7c"
    ["test_300_checked.csv"]="254418|md5:57acad1d418599194ecacd3d4dd7bcf8"
    ["Si_50_checked.csv"]="18343|md5:705548706c9e66c141a3be5e9d04d509"
    ["P_50_checked.csv"]="18746|md5:a2765d3a2457dd5a7d486d01688a8517"
    ["B_50_checked.csv"]="17758|md5:29f849f72aa82a28f03021db2e084ffd"
    ["NMRexp_10to24_1_1004_sc_less_than_1.parquet"]="528502118|md5:9a51f69f554b77ba5120ec2ac93f65fd"
    ["NMRexp_10to24_1_1004.parquet"]="661259287|md5:2f9ed8bc533364dfb9dce96c83703dd4"
    ["NMRexp_10to24_1_1004.csv"]="2143071209|md5:4e77af655abf917fdc9a07ff97ad6b55"
)

download_url() {
    local filename="$1"
    echo "https://zenodo.org/records/${ZENODO_RECORD}/files/${filename}?download=1"
}

# Download order: small validation files first, then large data files
download_order=(
    "NMRexp_10to24_1_0811.py"
    "F_50_checked.csv"
    "hetero_200_checked.csv"
    "test_300_checked.csv"
    "Si_50_checked.csv"
    "P_50_checked.csv"
    "B_50_checked.csv"
    "NMRexp_10to24_1_1004_sc_less_than_1.parquet"
    "NMRexp_10to24_1_1004.parquet"
    "NMRexp_10to24_1_1004.csv"
)

if [ "${SAMPLE_MODE}" = true ]; then
    download_order=("NMRexp_10to24_1_0811.py" "test_300_checked.csv" "hetero_200_checked.csv")
    echo "[SAMPLE] Will download only validation files"
fi

downloaded=0
failed=0
for filename in "${download_order[@]}"; do
    IFS='|' read -r size md5 <<< "${NMREXP_FILES[${filename}]}"
    url=$(download_url "${filename}")

    size_mb=$(echo "scale=1; ${size} / 1048576" | bc 2>/dev/null || echo "${size}")
    echo ""
    echo "--- ${filename} (${size_mb} MB) ---"

    # Check if already downloaded
    if [ -f "${filename}" ]; then
        actual_size=$(stat -c%s "${filename}" 2>/dev/null || stat -f%z "${filename}" 2>/dev/null || echo 0)
        if [ "${actual_size}" = "${size}" ]; then
            echo "[SKIP] Already downloaded, size matches"
            downloaded=$((downloaded + 1))
            continue
        fi
    fi

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
    echo "[WARN] ${failed} file(s) failed. Re-run to resume."
    exit 1
fi
