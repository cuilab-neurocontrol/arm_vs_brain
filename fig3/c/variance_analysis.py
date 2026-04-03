#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variance Analysis for Fig3c
Extracts vpps, vmps, vmms, vpms at perp_t=-0.3, exec_t=0.2 for 4 conditions
"""

import numpy as np
import pandas as pd
import os

def round_columns(df, columns, decimals=2):
    df[columns] = df[columns].round(decimals)
    return df

def mean_ci(arr):
    arr = np.asarray(arr, dtype=float).ravel()
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    m = float(np.mean(arr))
    if arr.size == 1:
        return (m, m, m)
    se = np.std(arr, ddof=1) / np.sqrt(arr.size)
    ci = 1.96 * se
    return (m, m - ci, m + ci)

def process_condition(merged_file, output_dir, condition_name):
    """Process a single condition and save variance CSV files"""
    df = pd.read_csv(merged_file)

    # Clean and filter data
    df_cleaned = df[~df['dataset'].str.contains('Shuffle', case=False, na=False)]
    filtered_df_ = round_columns(df_cleaned.copy(), ['perp_t', 'exec_t'], decimals=2)
    filtered_df = filtered_df_[(filtered_df_['perp_t'] == -0.3) & (filtered_df_['exec_t'] == 0.2)]

    if filtered_df.empty:
        print(f"  {condition_name}: No data at perp_t=-0.3, exec_t=0.2")
        return

    # Get unique datasets
    result_unique = filtered_df.drop_duplicates(subset=['dataset'])
    n_datasets = len(result_unique)
    print(f"  {condition_name}: {n_datasets} datasets")

    # Extract variance values
    bar_list = result_unique[['vpps', 'vmps', 'vmms', 'vpms']].to_numpy()
    vpps_vals = bar_list[:, 0]  # prep in prep space
    vmps_vals = bar_list[:, 1]  # exec in prep space
    vmms_vals = bar_list[:, 2]  # exec in exec space
    vpms_vals = bar_list[:, 3]  # prep in exec space

    # 1. manopt_plot_data.csv
    data = {
        'sum': list(vpps_vals) + list(vmps_vals) + list(vmms_vals) + list(vpms_vals),
        'Group type': ['prep.'] * n_datasets + ['move.'] * n_datasets + ['prep.'] * n_datasets + ['move.'] * n_datasets,
        'space type': ['prep.'] * n_datasets + ['prep.'] * n_datasets + ['move.'] * n_datasets + ['move.'] * n_datasets
    }

    manopt_rows = []
    for val, gt, st in zip(data['sum'], data['Group type'], data['space type']):
        manopt_rows.append([st, gt, float(val)])
    pd.DataFrame(manopt_rows, columns=['space_type', 'group_type', 'value']).to_csv(
        os.path.join(output_dir, f'manopt_plot_data_{condition_name}.csv'), index=False)

    # 2. prep_exec_prep_prep.csv (prep space data)
    rows_csv1 = []
    for name, vals in [("prep,prep", vpps_vals), ("prep,exec", vpms_vals)]:
        mean_, low_, high_ = mean_ci(vals)
        rows_csv1.append([name, mean_, low_, high_])
    pd.DataFrame(rows_csv1, columns=['group_name', 'group_value', 'ci_low', 'ci_high']).to_csv(
        os.path.join(output_dir, f'prep_exec_prep_prep_{condition_name}.csv'), index=False)

    # 3. exec_prep_exec_exec.csv (exec space data)
    rows_csv2 = []
    for name, vals in [("exec,prep", vmps_vals), ("exec,exec", vmms_vals)]:
        mean_, low_, high_ = mean_ci(vals)
        rows_csv2.append([name, mean_, low_, high_])
    pd.DataFrame(rows_csv2, columns=['group_name', 'group_value', 'ci_low', 'ci_high']).to_csv(
        os.path.join(output_dir, f'exec_prep_exec_exec_{condition_name}.csv'), index=False)

    # Print summary
    print(f"    vpps: {np.mean(vpps_vals):.2f}%, vmps: {np.mean(vmps_vals):.2f}%")
    print(f"    vmms: {np.mean(vmms_vals):.2f}%, vpms: {np.mean(vpms_vals):.2f}%")


# Compute submission root relative to this script's location
SUBMISSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Paths
base_path = os.path.join(SUBMISSION_ROOT, 'fig3')
preprocessing_path = f'{base_path}/preprocessing'
output_path = f'{base_path}/c'

# Four conditions (no feedback)
conditions = ['bohr_bc', 'bohr_hand', 'leb_bc', 'leb_hand']

print("=== Variance Analysis for Fig3c ===\n")

for cond in conditions:
    merged_file = f'{preprocessing_path}/{cond}/subspace/merged_file.csv'
    if os.path.exists(merged_file):
        process_condition(merged_file, output_path, cond)
    else:
        print(f"  {cond}: merged_file.csv not found")

print("\n=== Done ===")
