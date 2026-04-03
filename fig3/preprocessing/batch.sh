#!/bin/bash

#SBATCH -J subspace
#SBATCH -o %x_%j.out
#SBATCH -n 1
#SBATCH -p cn2
#SBATCH -c 1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/../subspace_numpy.py" -f $1 -p $2 -o $3
