#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PD Shift Analysis for Fig2c - All 6 conditions
Calculates proportion of cells with PD shift > 90 degrees between exec and delay periods
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def _pick_nearest_timepoint(df: pd.DataFrame, target_t: float) -> pd.DataFrame:
    tmp = df.copy()
    tmp['__dist__'] = (tmp['t'] - target_t).abs()
    idx = tmp.groupby(['cell', 'dataset'])['__dist__'].idxmin()
    return df.loc[idx].reset_index(drop=True)

def compute_pd_differences(
    df: pd.DataFrame,
    delay_time: float = -0.4,
    prep_time: float = -0.2,
    exec_time: float = 0.2,
    r2_threshold: float = 0.7,
):
    delay_df = _pick_nearest_timepoint(df, delay_time)
    prep_df = _pick_nearest_timepoint(df, prep_time)
    exec_df = _pick_nearest_timepoint(df, exec_time)

    delay_df = delay_df.rename(columns={col: f"{col}_delay" for col in delay_df.columns if col not in ['cell', 'dataset']})
    merged_df = prep_df.merge(exec_df, on=['cell', 'dataset'], suffixes=('_prep', '_exec'))
    merged_df = merged_df.merge(delay_df, on=['cell', 'dataset'])

    filt = (
        (merged_df['R2_exec'] > r2_threshold) &
        (merged_df['R2_delay'] > r2_threshold)
    )
    result_df = merged_df[filt]

    def _angle_diff(a, b):
        d = np.abs(a - b)
        d[d > np.pi] = np.abs(2 * np.pi - d[d > np.pi])
        return d

    theta1 = np.array(result_df.PD_prep.to_list())
    theta1[theta1 < 0] += 2 * np.pi

    phi1 = np.array(result_df.PD_delay.to_list())
    phi1[phi1 < 0] += 2 * np.pi

    diff_pd1 = _angle_diff(theta1, phi1)

    theta2 = np.array(result_df.PD_exec.to_list())
    theta2[theta2 < 0] += 2 * np.pi

    diff_pd2 = _angle_diff(theta2, phi1)
    return diff_pd1, diff_pd2, result_df


# Parameters
DELAY_TIME = -0.5
PREP_TIME = -0.2
R2_THRESHOLD = 0.7

# Compute submission root relative to this script's location
SUBMISSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Paths
base_path = os.path.join(SUBMISSION_ROOT, 'fig2')
preprocessing_path = f'{base_path}/preprocessing'
output_path = f'{base_path}/c'

# Six conditions
conditions = {
    'bohr_bc': f'{preprocessing_path}/bohr_bc/movement_onset_pd.csv',
    'bohr_hand': f'{preprocessing_path}/bohr_hand/movement_onset_pd.csv',
    'bohr_feedback': f'{preprocessing_path}/bohr_feedback/movement_onset_pd.csv',
    'leb_bc': f'{preprocessing_path}/leb_bc/movement_onset_pd.csv',
    'leb_hand': f'{preprocessing_path}/leb_hand/movement_onset_pd.csv',
    'leb_feedback': f'{preprocessing_path}/leb_feedback/movement_onset_pd.csv',
}

for cond_name, input_file in conditions.items():
    print(f"\n=== Processing {cond_name} ===")

    if not os.path.exists(input_file):
        print(f"  Input file not found: {input_file}")
        continue

    df_all = pd.read_csv(input_file)
    print(f"  Loaded {len(df_all)} rows, {df_all['cell'].nunique()} cells")

    # Get time range
    time = df_all.t
    time = time[(time > -0.5) & (time < 0.5)]
    time = np.sort(np.unique(time))

    if df_all.empty:
        print("  No data")
        continue

    pd_shift = []
    for t in tqdm(time, desc=f"  {cond_name}"):
        try:
            diff_pd1, diff_pd2, result_df = compute_pd_differences(
                df_all,
                delay_time=DELAY_TIME,
                prep_time=PREP_TIME,
                exec_time=t,
                r2_threshold=R2_THRESHOLD,
            )
            if len(diff_pd2) > 0:
                pd_shift.append(sum(diff_pd2 > (np.pi / 2)) / len(diff_pd2))
            else:
                pd_shift.append(np.nan)
        except Exception:
            pd_shift.append(np.nan)

    pd_shift = np.array(pd_shift)
    out_df = pd.DataFrame({'x': time, 'condition': cond_name, 'y': pd_shift})

    # Create output directory for this condition
    cond_output_dir = f'{output_path}/{cond_name}'
    os.makedirs(cond_output_dir, exist_ok=True)

    output_file = f'{cond_output_dir}/exec_delay_pd_timeseries.csv'
    out_df.to_csv(output_file, index=False)
    print(f"  Saved to: {output_file}")

print("\n=== All conditions completed ===")
