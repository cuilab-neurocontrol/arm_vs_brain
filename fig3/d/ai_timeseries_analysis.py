#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alignment Index Timeseries Analysis for Fig3d
Calculates mean alignment_index vs exec_t (with perp_t fixed at -0.3)
"""

import pandas as pd
import numpy as np
import os

def round_columns(df, columns, decimals=2):
    df[columns] = df[columns].round(decimals)
    return df

def process_condition(merged_file, output_dir, condition_name):
    """Process a single condition and save timeseries CSV"""
    df = pd.read_csv(merged_file)

    # Round and filter
    filtered_df_ = round_columns(df.copy(), ['perp_t', 'exec_t'], decimals=2)
    filtered_df_ = filtered_df_[(filtered_df_['perp_t'] == -0.3) & (filtered_df_['exec_t'] > -0.3)]
    filtered_df_ = filtered_df_.drop_duplicates(subset=['dataset', 'exec_t'])

    if filtered_df_.empty:
        print(f"  {condition_name}: No data")
        return

    # Aggregate by exec_t
    stats_df = filtered_df_.groupby('exec_t')['alignment_index'].agg(['mean', 'count', 'sem']).reset_index()

    # Calculate 95% CI
    stats_df['ci95'] = 1.96 * stats_df['sem']
    stats_df['ci_lower'] = stats_df['mean'] - stats_df['ci95']
    stats_df['ci_upper'] = stats_df['mean'] + stats_df['ci95']

    # Save to CSV
    final_stats_df = stats_df[['exec_t', 'mean', 'ci_lower', 'ci_upper']].rename(
        columns={'exec_t': 'x', 'mean': 'y', 'ci_lower': 'CI_left', 'ci_upper': 'CI_right'}
    )
    output_csv = f'{output_dir}/ai_timeseries_{condition_name}.csv'
    final_stats_df.to_csv(output_csv, index=False)

    # Print summary
    n_datasets = filtered_df_['dataset'].nunique()
    n_timepoints = len(stats_df)
    print(f"  {condition_name}: {n_datasets} datasets, {n_timepoints} timepoints")
    print(f"    x range: [{final_stats_df['x'].min():.2f}, {final_stats_df['x'].max():.2f}]")
    print(f"    y range: [{final_stats_df['y'].min():.4f}, {final_stats_df['y'].max():.4f}]")


# Compute submission root relative to this script's location
SUBMISSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Paths
base_path = os.path.join(SUBMISSION_ROOT, 'fig3')
preprocessing_path = f'{base_path}/preprocessing'
output_path = f'{base_path}/d'

# Six conditions (including feedback)
conditions = ['bohr_bc', 'bohr_hand', 'bohr_feedback', 'leb_bc', 'leb_hand', 'leb_feedback']

print("=== Alignment Index Timeseries Analysis for Fig3d ===\n")

for cond in conditions:
    merged_file = f'{preprocessing_path}/{cond}/subspace/merged_file.csv'
    if os.path.exists(merged_file):
        process_condition(merged_file, output_path, cond)
    else:
        print(f"  {cond}: merged_file.csv not found")

print("\n=== Done ===")
