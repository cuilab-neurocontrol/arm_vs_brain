#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alignment Index Analysis for Fig3b
Extracts alignment_index at perp_t=-0.3, exec_t=0.2 for all conditions
"""

import os
import pandas as pd
import numpy as np
from scipy.io import loadmat

def round_columns(df, columns, decimals=2):
    df[columns] = df[columns].round(decimals)
    return df

def process_condition(merged_file, output_dir, condition_name, monkey_name):
    """Process a single condition and save alignment_index CSV"""
    df = pd.read_csv(merged_file)

    # Round time columns
    df = round_columns(df, ['perp_t', 'exec_t'], decimals=2)

    # Filter for specific time window
    filtered_df = df[(df['perp_t'] == -0.3) & (df['exec_t'] == 0.2)]

    if filtered_df.empty:
        print(f"  {condition_name}: No data at perp_t=-0.3, exec_t=0.2")
        return None

    # Get unique alignment_index per dataset
    unique_df = filtered_df.drop_duplicates(subset=['dataset'])
    vals = unique_df['alignment_index'].tolist()

    # Save CSV
    csv_data = [[monkey_name, float(v)] for v in vals]
    output_file = f'{output_dir}/alignment_{condition_name}.csv'
    pd.DataFrame(csv_data, columns=['group_name', 'group_value']).to_csv(output_file, index=False)

    print(f"  {condition_name}: {len(vals)} datasets, mean={np.mean(vals):.4f}")
    return vals

# Compute submission root relative to this script's location
SUBMISSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Paths
base_path = os.path.join(SUBMISSION_ROOT, 'fig3')
preprocessing_path = f'{base_path}/preprocessing'
output_path = f'{base_path}/b'

# Conditions
conditions = {
    'bohr_bc': ('Monkey B', 'bohr_bc'),
    'bohr_hand': ('Monkey B', 'bohr_hand'),
    'bohr_feedback': ('Monkey B', 'bohr_feedback'),
    'leb_bc': ('Monkey L', 'leb_bc'),
    'leb_hand': ('Monkey L', 'leb_hand'),
    'leb_feedback': ('Monkey L', 'leb_feedback'),
}

print("=== Alignment Index Analysis ===\n")

all_results = {}
for cond_key, (monkey_name, cond_name) in conditions.items():
    merged_file = f'{preprocessing_path}/{cond_key}/subspace/merged_file.csv'
    vals = process_condition(merged_file, output_path, cond_name, monkey_name)
    if vals:
        all_results[cond_key] = vals

# Load control data if available
control_file_bohr = os.path.join(SUBMISSION_ROOT, 'fig3', 'control_data', 'Bohr', 'rand1.mat')  # NOTE: control data files need to be placed here
control_file_leb = os.path.join(SUBMISSION_ROOT, 'fig3', 'control_data', 'Leb', 'rand1.mat')  # NOTE: control data files need to be placed here

print("\n=== Control Data ===")
for monkey, control_file in [('bohr', control_file_bohr), ('leb', control_file_leb)]:
    try:
        data = loadmat(control_file)
        control_vals = data['randIx'].squeeze().ravel()

        # Random sample 20 if more
        if len(control_vals) > 20:
            control_vals = np.random.choice(control_vals, size=20, replace=False)

        # Save control CSV
        monkey_name = 'Monkey B' if monkey == 'bohr' else 'Monkey L'
        control_csv = [['Control', float(v)] for v in control_vals]
        output_file = f'{output_path}/alignment_{monkey}_control.csv'
        pd.DataFrame(control_csv, columns=['group_name', 'group_value']).to_csv(output_file, index=False)
        print(f"  {monkey}_control: {len(control_vals)} samples, mean={np.mean(control_vals):.4f}")
    except Exception as e:
        print(f"  {monkey}_control: Not found or error - {e}")

print("\n=== Done ===")
