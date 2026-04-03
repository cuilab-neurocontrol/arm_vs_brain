#!/bin/bash

# Subspace analysis batch submit for all 6 conditions

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$SCRIPT_DIR/.."
PREPROCESSING_DIR="$BASE_DIR/preprocessing"
OUTPUT_BASE="$BASE_DIR/preprocessing"
BATCH_SCRIPT="$OUTPUT_BASE/batch.sh"

conditions=("bohr_bc" "bohr_hand" "bohr_feedback" "leb_bc" "leb_hand" "leb_feedback")

for cond in "${conditions[@]}"; do
    INPUT_DIR="$PREPROCESSING_DIR/$cond"
    OUTPUT_DIR="$OUTPUT_BASE/$cond/subspace"

    # Create output directory if not exists
    mkdir -p "$OUTPUT_DIR/log"

    echo "=== Submitting $cond ==="

    # Count npz files
    count=$(ls "$INPUT_DIR"/*.npz 2>/dev/null | wc -l)
    echo "  Found $count npz files"

    # Submit jobs for each npz file
    for file in "$INPUT_DIR"/*.npz; do
        if [[ -f "$file" ]]; then
            filename=$(basename "$file")
            echo "  Submitting: $filename"
            sbatch --job-name="sub_${cond}" \
                   --output="$OUTPUT_DIR/log/%j.out" \
                   "$BATCH_SCRIPT" "$filename" "$INPUT_DIR" "$OUTPUT_DIR"
        fi
    done
done

echo ""
echo "All jobs submitted. Check with: squeue -u $USER"
