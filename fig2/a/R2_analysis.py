#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R2 proportion analysis for all conditions (bohr_bc, bohr_hand, leb_bc, leb_hand)

This script calculates the proportion of cells with R2 > 0.7 at each time point
for different time windows: prep, exec, delay, iti.

Author: lichenyang
"""

import os
import pandas as pd
import numpy as np
from scipy import stats


def calculate_r2_proportion(data, threshold=0.7):
    """Calculate proportion of cells with R2 > threshold at each time point"""
    grouped = data.groupby(['t', 'dataset']).agg({
        'R2': lambda x: (x > threshold).sum() / len(x)
    }).reset_index()

    time_grouped = grouped.groupby('t')['R2'].agg(['mean', 'std', 'count']).reset_index()

    confidence_level = 0.95
    alpha = 1 - confidence_level

    time_grouped['CI_left'] = time_grouped.apply(
        lambda row: row['mean'] - stats.t.ppf(1-alpha/2, row['count']-1) * row['std'] / np.sqrt(row['count'])
        if row['count'] > 1 else row['mean'], axis=1
    )
    time_grouped['CI_right'] = time_grouped.apply(
        lambda row: row['mean'] + stats.t.ppf(1-alpha/2, row['count']-1) * row['std'] / np.sqrt(row['count'])
        if row['count'] > 1 else row['mean'], axis=1
    )

    result_df = pd.DataFrame({
        'x': time_grouped['t'],
        'y': time_grouped['mean'],
        'CI_left': time_grouped['CI_left'],
        'CI_right': time_grouped['CI_right']
    })

    return result_df


# Compute submission root relative to this script's location
SUBMISSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Base paths
preprocessing_base = os.path.join(SUBMISSION_ROOT, 'fig2', 'preprocessing')
output_base = os.path.join(SUBMISSION_ROOT, 'fig2', 'a')
os.makedirs(output_base, exist_ok=True)

# Process all conditions
conditions = ['bohr_bc', 'bohr_hand', 'bohr_feedback', 'leb_bc', 'leb_hand', 'leb_feedback']

for condition in conditions:
    print(f"\n=== Processing {condition} ===")
    data_path = os.path.join(preprocessing_base, condition)
    output_path = os.path.join(output_base, condition)
    os.makedirs(output_path, exist_ok=True)

    # R2_prep: from GO_pd.csv, prep window (t>=0 & t<0.3)
    try:
        go_pd = pd.read_csv(f'{data_path}/GO_pd.csv')
        prep_window = go_pd[(go_pd.t >= 0) & (go_pd.t < 0.3)]
        prep_proportion = calculate_r2_proportion(prep_window, threshold=0.7)
        prep_proportion.to_csv(f'{output_path}/prep_window_r2_proportion.csv', index=False)
        print(f"  prep_window: {len(prep_window)} rows, saved")
    except Exception as e:
        print(f"  prep_window error: {e}")

    # R2_exec: from movement_onset_pd.csv, exec window (t>=0 & t<0.3)
    try:
        mo_pd = pd.read_csv(f'{data_path}/movement_onset_pd.csv')
        exec_window = mo_pd[(mo_pd.t >= 0) & (mo_pd.t < 0.3)]
        exec_proportion = calculate_r2_proportion(exec_window, threshold=0.7)
        exec_proportion.to_csv(f'{output_path}/exec_window_r2_proportion.csv', index=False)
        print(f"  exec_window: {len(exec_window)} rows, saved")
    except Exception as e:
        print(f"  exec_window error: {e}")

    # R2_delay_iti: from target_on_pd.csv
    try:
        target_pd = pd.read_csv(f'{data_path}/target_on_pd.csv')

        # ITI window (t>=-0.3 & t<0)
        iti_window = target_pd[(target_pd.t >= -0.3) & (target_pd.t < 0)]
        iti_proportion = calculate_r2_proportion(iti_window, threshold=0.7)
        iti_proportion.to_csv(f'{output_path}/iti_window_r2_proportion.csv', index=False)
        print(f"  iti_window: {len(iti_window)} rows, saved")

        # Delay window (t>=0 & t<0.3)
        delay_window = target_pd[(target_pd.t >= 0) & (target_pd.t < 0.3)]
        delay_proportion = calculate_r2_proportion(delay_window, threshold=0.7)
        delay_proportion.to_csv(f'{output_path}/delay_window_r2_proportion.csv', index=False)
        print(f"  delay_window: {len(delay_window)} rows, saved")
    except Exception as e:
        print(f"  delay/iti window error: {e}")

print("\n=== All R2 analyses completed ===")
