#!/usr/bin/env bash
# verify_mto_core.sh — Run all mandatory MTO core tests and exit 0 only when all pass.
#
# Usage:
#   bash scripts/verify_mto_core.sh              # run all tests
#   bash scripts/verify_mto_core.sh --quick      # skip slow DetaNet-dependent tests
#   bash scripts/verify_mto_core.sh --verbose    # verbose output
#
# Exit codes:
#   0 — all tests passed
#   1 — one or more tests failed
#   2 — test discovery or environment error

set -euo pipefail

cd "$(dirname "$0")/.."

VERBOSE=""
QUICK=""

for arg in "$@"; do
    case "$arg" in
        --verbose|-v) VERBOSE="-v" ;;
        --quick|-q)   QUICK="1" ;;
        *)            echo "Unknown arg: $arg"; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# Collect test files
# ---------------------------------------------------------------------------
ALL_TESTS=(
    tests/test_tensor_split_reconstruct.py
    tests/test_tensor_equivariance.py
    tests/test_tensor_adapter.py
    tests/test_signed_routing.py
    tests/test_mto_equivariance.py
    tests/test_cg_coupling.py
    tests/test_tensor_gate.py
    tests/test_translation.py
    tests/test_permutation.py
    tests/test_mto_forward_backward.py
)

if [ "$QUICK" != "1" ]; then
    ALL_TESTS+=(
        tests/test_detanet_import.py
        tests/test_detanet_forward.py
        tests/test_detanet_tensor_layout.py
    )
fi

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
echo "=============================================="
echo " MTO Core Verification Suite"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo " Python: $(python --version 2>&1)"
echo " Pytest: $(python -m pytest --version 2>&1)"
echo " Test files: ${#ALL_TESTS[@]}"
echo "=============================================="
echo ""

# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------
PASS_COUNT=0
FAIL_COUNT=0
FAILED_TESTS=()

for test_file in "${ALL_TESTS[@]}"; do
    if [ ! -f "$test_file" ]; then
        echo "  SKIP  $test_file (not found)"
        continue
    fi

    printf "  ...   %s" "$test_file"
    if python -m pytest "$test_file" -q --tb=line $VERBOSE > /tmp/mto_test_out.$$ 2>&1; then
        echo -e "\r  PASS  $test_file"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo -e "\r  FAIL  $test_file"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        FAILED_TESTS+=("$test_file")
        # Print condensed failure info
        tail -20 /tmp/mto_test_out.$$
    fi
done

rm -f /tmp/mto_test_out.$$

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo " Results: $PASS_COUNT passed, $FAIL_COUNT failed"
echo "=============================================="

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo ""
    echo "Failed test files:"
    for f in "${FAILED_TESTS[@]}"; do
        echo "  - $f"
    done
    echo ""
    echo "To re-run failures:"
    echo "  python -m pytest ${FAILED_TESTS[*]} -v"
    exit 1
fi

echo "All MTO core tests passed."
exit 0
