#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Violin Plot Analysis for Fig2d - 4 conditions (no feedback)
Calculates PD differences between prep/exec and delay periods
"""

import numpy as np
import pandas as pd
import os
from scipy import stats

def compute_violin_data(df, delay_time=-0.4, prep_time=-0.2, exec_time=0.2, r2_threshold=0.7):
    """Compute violin plot data for a single condition"""

    # Pick nearest timepoints
    delay_df = df.iloc[(df['t'] - delay_time).abs().argsort()].groupby(['cell', 'dataset']).head(1)
    prep_df = df.iloc[(df['t'] - prep_time).abs().argsort()].groupby(['cell', 'dataset']).head(1)
    exec_df = df.iloc[(df['t'] - exec_time).abs().argsort()].groupby(['cell', 'dataset']).head(1)

    # Rename and merge
    delay_df = delay_df.rename(columns={col: f"{col}_delay" for col in delay_df.columns if col not in ['cell', 'dataset']})
    merged_df = prep_df.merge(exec_df, on=['cell', 'dataset'], suffixes=('_prep', '_exec'))
    merged_df = merged_df.merge(delay_df, on=['cell', 'dataset'])

    # Filter by R2
    filtered_df = merged_df[
        (merged_df['R2_prep'] > r2_threshold) &
        (merged_df['R2_exec'] > r2_threshold) &
        (merged_df['R2_delay'] > r2_threshold)
    ]

    if len(filtered_df) == 0:
        return None, None, None

    # Compute prep-delay difference
    theta = np.array(filtered_df.PD_prep.to_list())
    theta[theta < 0] += 2 * np.pi
    phi = np.array(filtered_df.PD_delay.to_list())
    phi[phi < 0] += 2 * np.pi
    diff_pd1 = np.abs(theta - phi)
    diff_pd1[diff_pd1 > np.pi] = np.abs(2 * np.pi - diff_pd1[diff_pd1 > np.pi])

    # Compute exec-delay difference
    theta = np.array(filtered_df.PD_exec.to_list())
    theta[theta < 0] += 2 * np.pi
    phi = np.array(filtered_df.PD_delay.to_list())
    phi[phi < 0] += 2 * np.pi
    diff_pd2 = np.abs(theta - phi)
    diff_pd2[diff_pd2 > np.pi] = np.abs(2 * np.pi - diff_pd2[diff_pd2 > np.pi])

    return diff_pd1, diff_pd2, filtered_df


# Compute submission root relative to this script's location
SUBMISSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Paths
base_path = os.path.join(SUBMISSION_ROOT, 'fig2')
preprocessing_path = f'{base_path}/preprocessing'
output_path = f'{base_path}/d'
os.makedirs(output_path, exist_ok=True)

# Four conditions (no feedback)
conditions = {
    'bohr_bc': f'{preprocessing_path}/bohr_bc/movement_onset_pd.csv',
    'bohr_hand': f'{preprocessing_path}/bohr_hand/movement_onset_pd.csv',
    'leb_bc': f'{preprocessing_path}/leb_bc/movement_onset_pd.csv',
    'leb_hand': f'{preprocessing_path}/leb_hand/movement_onset_pd.csv',
}

all_stats = []

for cond_name, input_file in conditions.items():
    print(f"\n=== Processing {cond_name} ===")

    if not os.path.exists(input_file):
        print(f"  Input file not found: {input_file}")
        continue

    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df)} rows, {df['cell'].nunique()} cells")

    # Compute violin data
    diff_pd1, diff_pd2, result_df = compute_violin_data(df)

    if diff_pd1 is None:
        print(f"  No data after filtering")
        continue

    print(f"  Filtered cells: {len(result_df)}")

    # Save violin plot data
    violin_data = pd.DataFrame({
        'group_value': np.concatenate([diff_pd1, diff_pd2]),
        'group_name': ['prep.'] * len(diff_pd1) + ['exec.'] * len(diff_pd2)
    })

    cond_output_dir = f'{output_path}/{cond_name}'
    os.makedirs(cond_output_dir, exist_ok=True)
    violin_data.to_csv(f'{cond_output_dir}/absolute_delay_difference.csv', index=False)
    print(f"  Saved violin data: {len(diff_pd1)} prep, {len(diff_pd2)} exec")

    # T-test
    t_statistic, p_value = stats.ttest_ind(diff_pd1, diff_pd2)

    stats_result = {
        'condition': cond_name,
        'comparison': 'prep. vs exec.',
        'group1_name': 'prep.',
        'group2_name': 'exec.',
        'group1_mean': np.mean(diff_pd1),
        'group2_mean': np.mean(diff_pd2),
        'group1_std': np.std(diff_pd1, ddof=1),
        'group2_std': np.std(diff_pd2, ddof=1),
        'group1_n': len(diff_pd1),
        'group2_n': len(diff_pd2),
        't_statistic': t_statistic,
        'p_value': p_value,
        'significant': 'Yes' if p_value < 0.05 else 'No'
    }
    all_stats.append(stats_result)

    # Save individual stats
    pd.DataFrame([stats_result]).to_csv(f'{cond_output_dir}/ttest_result.csv', index=False)

    print(f"  prep. mean: {np.mean(diff_pd1):.4f} ± {np.std(diff_pd1, ddof=1):.4f} (n={len(diff_pd1)})")
    print(f"  exec. mean: {np.mean(diff_pd2):.4f} ± {np.std(diff_pd2, ddof=1):.4f} (n={len(diff_pd2)})")
    print(f"  t={t_statistic:.4f}, p={p_value:.4e}, significant: {'Yes' if p_value < 0.05 else 'No'}")

# Save combined stats
if all_stats:
    combined_stats = pd.DataFrame(all_stats)
    combined_stats.to_csv(f'{output_path}/all_ttest_results.csv', index=False)
    print(f"\n=== Combined stats saved to {output_path}/all_ttest_results.csv ===")

print("\n=== All conditions completed ===")
