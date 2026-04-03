#!/bin/bash

# Submit 8 CCA jobs in parallel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="$SCRIPT_DIR"
PYTHON="python"
SCRIPT="${OUTPUT_DIR}/cca_analysis.py"

for monkey in bohr leb; do
    for condition in feedforward feedback; do
        for prep in "" "--prep"; do
            if [ -z "$prep" ]; then
                epoch="exec"
            else
                epoch="prep"
            fi
            job_name="cca_${monkey}_${condition}_${epoch}"

            sbatch --job-name="$job_name" \
                   --output="${OUTPUT_DIR}/${job_name}_%j.out" \
                   --partition=cn2 \
                   --ntasks=1 \
                   --cpus-per-task=4 \
                   --wrap="cd ${OUTPUT_DIR} && ${PYTHON} ${SCRIPT} --monkey ${monkey} --condition ${condition} ${prep}"

            echo "Submitted: $job_name"
        done
    done
done

echo "All 8 jobs submitted!"
