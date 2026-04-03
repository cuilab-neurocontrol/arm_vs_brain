#!/bin/bash

#SBATCH -J cca_fig4c
#SBATCH -o cca_%j.out
#SBATCH -n 1
#SBATCH -p cn2
#SBATCH -c 8

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
python cca_analysis.py --parallel 8
