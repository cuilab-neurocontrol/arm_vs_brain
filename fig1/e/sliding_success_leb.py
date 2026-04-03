#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sliding window success rate analysis for Monkey Leb (Brain Control)

This script calculates the sliding window success rate for BMI performance
using a 75-trial window and computes 95% confidence intervals.

Author: lichenyang
"""

import os
import numpy as np
import pandas as pd
import pynapple as nap
from scipy import stats

# Compute submission root relative to this script's location
SUBMISSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Data folder path (kilosort processed data)
data_folder = os.path.join(SUBMISSION_ROOT, 'nwb_kilosort', 'leb', 'Brain_control')

# Output folder
output_folder = os.path.join(SUBMISSION_ROOT, 'fig1', 'e')
os.makedirs(output_folder, exist_ok=True)

data = []

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

    # Find trial start (marker 24) and end (marker 5) positions
    marker_24_pos = np.where(marker == 24)[0]
    marker_5_pos = np.where(marker == 5)[0]
    marker_24_time = events.t[np.where(marker == 24)[0]]
    marker_5_time = events.t[np.where(marker == 5)[0]]

    # Extract trial events
    trial_event = [events[i:j+1] for i, j in zip(marker_24_pos, marker_5_pos)]

    # Extract position and coefficient data for each trial
    trial_pos = [pos_pd.d[np.where(pos_pd.t[:] < j * (pos_pd.t[:] > i))[0]] for i, j in zip(marker_24_time, marker_5_time)]
    trial_coeff = [coeff_pd.d[np.where(coeff_pd.t[:] < j * (coeff_pd.t[:] > i))[0]] for i, j in zip(marker_24_time, marker_5_time)]

    # Get trial data
    trial = [events[i:j+1] for i, j in zip(marker_24_pos, marker_5_pos)]
    duration = [i.t[-1] - i.t[3] for i in trial]

    # Check if trial was successful (marker 20 present)
    trial_success = [20 in i for i in trial]

    # Calculate trial time relative to first trial (in minutes)
    trial_time = [(i.t[-1] - trial[0].t[0]) / 60 for i in trial]

    # Get training flag data
    flag_times = coeff_pd.t[:]
    training_flag = coeff_pd.d[:]
    trial_flag = [training_flag[np.where(j[0] < flag_times * (j[-1] > flag_times))[0]] for j in trial]
    trial_flag_times = [flag_times[np.where(j[0] < flag_times * (j[-1] > flag_times))[0]] for j in trial]
    trial_flag_times = [(i - trial[0].t[0]).squeeze() / 60 for i in trial_flag_times]

    # Calculate sliding window statistics
    mean_success = []
    sliding_time = []
    mean_duration = []
    flag = []
    epoch = []
    paradigm_start = trial[0].t[0]

    # Use 75-trial sliding window
    for i in range(0, len(trial) - 75):
        mean_success.append(trial_success[i:i+75].count(True) / 75)
        mean_duration.append(np.mean(duration[i:i+75]))
        sliding_time.append(round(trial_time[i+75]))
        epoch.append(i)

    # Create dataframe
    labels = ['Success rate'] * len(mean_success)
    value = mean_success
    time = sliding_time
    df = pd.DataFrame(zip(value, time, labels),
                    columns=['Percentage', 'Time (min)', 'labels'])
    df['dataset'] = raw_dirname
    data.append(df)

# Concatenate all data
if len(data) > 0:
    data = pd.concat(data)
    data.index = range(0, len(data))

    # Group by time and calculate statistics
    grouped = data.groupby('Time (min)')['Percentage'].agg(['mean', 'std', 'count']).reset_index()

    # Calculate 95% confidence interval
    confidence_level = 0.95
    alpha = 1 - confidence_level

    grouped['CI_left'] = grouped.apply(
        lambda row: row['mean'] - stats.t.ppf(1-alpha/2, row['count']-1) * row['std'] / np.sqrt(row['count']),
        axis=1
    )
    grouped['CI_right'] = grouped.apply(
        lambda row: row['mean'] + stats.t.ppf(1-alpha/2, row['count']-1) * row['std'] / np.sqrt(row['count']),
        axis=1
    )

    # Create result dataframe
    result_df = pd.DataFrame({
        'x': grouped['Time (min)'],
        'y': grouped['mean'],
        'CI_left': grouped['CI_left'],
        'CI_right': grouped['CI_right']
    })

    # Save to CSV
    output_path = os.path.join(output_folder, 'sliding_success_leb.csv')
    result_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print(f"Total time points: {len(result_df)}")
else:
    print("No data found!")
