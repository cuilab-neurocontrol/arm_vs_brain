# Preserved Neural Dynamics across Arm and Brain-controlled Movements

## Directory Structure

```
├── demo/                    # Demo (single session, see below)
│   ├── demo.ipynb           # Demo notebook
│   └── demo_data/
│       └── demo_bmi.nwb     # Demo NWB file (29 MB)
├── fig1/                    # BMI performance analysis
│   ├── e/                   # Sliding window success rate
│   └── f/                   # Success rate, trial time, path efficiency
├── fig2/                    # Preferred direction analysis
│   ├── a/                   # R2 proportion analysis
│   ├── c/                   # PD shift analysis
│   ├── d/                   # Violin plot analysis
│   └── preprocessing/       # PD tuning curve fitting (per condition)
├── fig3/                    # Neural subspace analysis
│   ├── b/                   # Alignment index
│   ├── c/                   # Variance analysis
│   ├── d/                   # AI time series
│   ├── e/                   # VMPS time series
│   ├── preprocessing/       # PCA trajectory & subspace computation
│   └── subspace_numpy.py
└── fig4/                    # CCA analysis
    └── c/                   # Feedforward/feedback CCA
```

## 1. System Requirements

### Software dependencies

- Python 3.11+
- numpy 2.3.3
- pandas 2.x
- scipy 1.16.x
- scikit-learn 1.7.x
- pynapple 0.9.2
- h5py 3.x
- tqdm 4.x
- matplotlib 3.x
- pymanopt 2.x
- autograd 1.x

A `requirements.txt` is provided for convenience.

### External dependencies (fig4 CCA analysis)

The CCA analysis in `fig4/c/cca_analysis.py` depends on [BeNeuroLab/2022-preserved-dynamics](https://github.com/BeNeuroLab/2022-preserved-dynamics). To run it:

```bash
# Clone the repo
git clone https://github.com/BeNeuroLab/2022-preserved-dynamics.git

# Install dependencies (pyaldata is not on PyPI)
pip install git+https://github.com/BeNeuroLab/pyaldata.git
pip install torch matplotlib

# Symlink tools and params into fig4/c/
ln -s /path/to/2022-preserved-dynamics/tools fig4/c/tools
ln -s /path/to/2022-preserved-dynamics/params.py fig4/c/params.py

# Add the repo root to PYTHONPATH (params.py imports monkey, mouse, etc.)
export PYTHONPATH="/path/to/2022-preserved-dynamics:$PYTHONPATH"

# Run
python fig4/c/cca_analysis.py --monkey bohr --condition feedforward
```

### Operating systems

Tested on:
- macOS 14 (Apple Silicon)
- Linux (CentOS 7, SLURM HPC cluster)

### Hardware

No non-standard hardware required. HPC with SLURM is recommended for preprocessing (fig2/preprocessing, fig3/preprocessing) due to computation time.

## 2. Installation Guide

```bash
pip install -r requirements.txt
```

Typical install time: < 2 minutes.

## 3. Demo

A demo notebook is provided in `demo/demo.ipynb`. It loads a single BMI session (Monkey Bohr, 115 units, ~400 trials) and visualizes peri-event neural activity aligned to movement onset.

**To run:**

```bash
cd demo
jupyter notebook demo.ipynb
```

**Expected output:** A heatmap of normalized firing rates across all units, sorted by peak firing time, aligned to movement onset.

**Expected run time:** < 30 seconds on a standard desktop computer.

## 4. Instructions for Use

### Running analysis scripts

All scripts use relative paths from the repository root. Each figure requires NWB data in `nwb_kilosort/` (not included in this repository; available separately).

### Estimated run times per figure

Tested on Apple M-series (macOS) and Linux HPC. Times scale with dataset size.

| Figure | Step | Script | Time per run |
|--------|------|--------|-------------|
| **Fig 1** | Analysis | `fig1/e/*.py`, `fig1/f/*.py` | ~2-5 min each |
| **Fig 2** | Preprocessing | `fig2/preprocessing/*/*.py` | ~1-4 min each (24 scripts) |
| | Analysis | `fig2/a/*.py`, `fig2/c/*.py`, `fig2/d/*.py` | < 1 min each |
| **Fig 3** | PCA trajectory | `fig3/preprocessing/*/make_pca_traj.py` | ~5-10 min each |
| | Subspace (HPC) | `fig3/subspace_numpy.py` | ~2-3 min per session |
| | Analysis | `fig3/b/*.py`, `fig3/c-e/*.py` | < 1 min each |
| **Fig 4** | CCA | `fig4/c/cca_analysis.py` | ~12 min per condition (8 conditions) |

**Total estimated time:** ~4-5 hours for all figures (can be parallelized on HPC).

### Reproduction

Each figure requires running preprocessing first, then the analysis script. For example:

```bash
# Fig 2: preprocessing (one condition)
python fig2/preprocessing/bohr_bc/pd_ana_movement_onset.py

# Fig 2: analysis
python fig2/c/pd_shift_analysis.py
```

For Fig 3 subspace analysis on HPC:

```bash
sbatch fig3/preprocessing/batch.sh <input_file> <input_dir> <output_dir>
```