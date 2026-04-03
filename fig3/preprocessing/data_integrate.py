#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Integration for Fig3b Subspace Analysis
Merges all CSV results for each condition
"""

import os
import pandas as pd

SUBMISSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

base_path = os.path.join(SUBMISSION_ROOT, 'fig3', 'preprocessing')
conditions = ['bohr_bc', 'bohr_hand', 'bohr_feedback', 'leb_bc', 'leb_hand', 'leb_feedback']

for cond in conditions:
    folder_path = os.path.join(base_path, cond, 'subspace')
    output_file = os.path.join(folder_path, 'merged_file.csv')

    # Get all subspace CSV files
    csv_files = [f for f in os.listdir(folder_path) if f.startswith('subspace_') and f.endswith('.csv')]

    if not csv_files:
        print(f"{cond}: No CSV files found")
        continue

    # Merge all CSV files
    merged_data = pd.DataFrame()
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        data = pd.read_csv(file_path)
        merged_data = pd.concat([merged_data, data], ignore_index=True)

    # Save merged data
    merged_data.to_csv(output_file, index=False)
    print(f"{cond}: Merged {len(csv_files)} files -> {len(merged_data)} rows")

print("\nDone!")
