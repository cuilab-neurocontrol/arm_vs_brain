#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Success Rate analysis for Monkey Leb (Brain Control)

This script calculates the success rate for trials where training flag is 0.
Success is defined as trials containing marker 20.

Author: lichenyang
"""

import os
import numpy as np
import pandas as pd
import pynapple as nap

# Compute submission root relative to this script's location
SUBMISSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data folder path (kilosort processed data)
data_folder = os.path.join(SUBMISSION_ROOT, 'nwb_kilosort', 'leb', 'Brain_control')

# Output folder
output_folder = os.path.join(SUBMISSION_ROOT, 'fig1', 'f')
os.makedirs(output_folder, exist_ok=True)

pf_dict = {}
pf_dict['group_name'] = []
pf_dict['group_value'] = []
pf_dict['dataset_name'] = []
trial_num = []

for raw_dirname in os.listdir(data_folder):
    # Filter for Semi2DBC120 datasets only
    if not 'Semi2DBC120' == raw_dirname.split('_')[1]:
        continue

    filename_ = os.path.join(data_folder, raw_dirname, 'formatted_data', 'standard_data_bmi.nwb')
    if not os.path.exists(filename_):
        continue

    print(f"Processing: {raw_dirname}")

    # Load NWB file
    time_series_data = nap.load_file(filename_)
    coeff_pd = time_series_data['training flag']
    pos_pd = time_series_data['pos']
    events = time_series_data['BehaviorMarkers']

    # Handle marker values (convert negative values)
    marker = events.d[:] + 65280 if events.d[0] < 0 else events.d[:]

    # Find trial boundaries
    marker_24_pos = np.where(marker == 24)[0]
    marker_5_pos = np.where(marker == 5)[0]
    marker_24_time = events.t[np.where(marker == 24)[0]]
    marker_5_time = events.t[np.where(marker == 5)[0]]

    # Extract position and coefficient data for each trial
    trial_pos = [pos_pd.d[np.where(pos_pd.t[:] < j * (pos_pd.t[:] > i))[0]]
                 for i, j in zip(marker_24_time, marker_5_time)]
    trial_coeff = [coeff_pd.d[np.where(coeff_pd.t[:] < j * (coeff_pd.t[:] > i))[0]]
                   for i, j in zip(marker_24_time, marker_5_time)]
    trial = [marker[i:j+1] for i, j in zip(marker_24_pos, marker_5_pos)]

    # Select trials where training flag is 0 throughout
    select_trial = [1 if sum(i) == 0 else 0 for i in trial_coeff]

    # Count successful trials (marker 20 present) with training flag = 0
    select_pos = [i for i, j, z in zip(trial_pos, select_trial, trial) if (j == 1) and (20 in z)]
    all_select_trial = [i for i, j in zip(trial_pos, select_trial) if (j == 1)]

    # Calculate success rate
    success_rate = len(select_pos) / len(all_select_trial)
    pf_dict['group_name'].append('L')
    pf_dict['group_value'].append(success_rate)
    pf_dict['dataset_name'].append(raw_dirname)
    trial_num.append(len(all_select_trial))

# Save results
data = pd.DataFrame(pf_dict)
output_path = os.path.join(output_folder, 'success_leb.csv')
data.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")
print(f"Total sessions: {len(data)}")
