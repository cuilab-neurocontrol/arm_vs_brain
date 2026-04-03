#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time To Target (TTT) analysis for Monkey Leb (Brain Control)

This script calculates the mean time to reach target for successful trials
where training flag is 0 throughout the trial.

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
    marker_start_pos = np.where(marker == 24)[0]
    marker_end_pos = np.where(marker == 5)[0]
    marker_start_time = events.t[np.where(marker == 3)[0]]
    marker_end_time = events.t[np.where(marker == 5)[0]]

    # Extract coefficient data for each trial
    trial_coeff = [coeff_pd.d[np.where(coeff_pd.t[:] < j * (coeff_pd.t[:] > i))[0]]
                   for i, j in zip(marker_start_time, marker_end_time)]
    trial = [events[i:j+1] for i, j in zip(marker_start_pos, marker_end_pos)]

    # Select trials where training flag is 0 throughout
    select_trial = [1 if sum(i) == 0 else 0 for i in trial_coeff]

    # Get successful trials (marker 20 present) with training flag = 0
    success_trial = [z for j, z in zip(select_trial, trial) if (j == 1) and (20 in z)]

    # Calculate time to target: from marker 3 to second-to-last marker
    times = [i.t[-2] - i.t[3] for i in success_trial]

    pf_dict['group_value'].append(np.mean(times))
    pf_dict['group_name'].append('L')

# Save results
data = pd.DataFrame(pf_dict)
output_path = os.path.join(output_folder, 'TTT_leb.csv')
data.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")
print(f"Total sessions: {len(data)}")
