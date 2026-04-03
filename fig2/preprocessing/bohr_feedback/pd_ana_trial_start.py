#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PD analysis aligned to trial start (marker 24) for Monkey Bohr Feedback (2DBC)

This script calculates preferred direction (PD) metrics for each unit
at each time bin relative to trial start.

Author: lichenyang
"""

import os
import pynapple as nap
import pandas as pd
import numpy as np
from sklearn import linear_model
from tqdm import tqdm


def DiscreatePD(ConditionVariable, ConditionFiringRate):
    """Calculate preferred direction using linear regression"""
    ConditionVariable = np.array([[np.cos(i), np.sin(i)] for i in ConditionVariable])
    PDModel = linear_model.LinearRegression()
    PDResults = {}
    PDModel.fit(ConditionVariable, ConditionFiringRate)
    PDResults['R2'] = PDModel.score(ConditionVariable, np.array(ConditionFiringRate))
    PDResults['BaseLine'] = PDModel.intercept_
    PDResults['ModulationDepth'] = np.linalg.norm(PDModel.coef_)
    PDResults['PD'] = np.arctan2(PDModel.coef_[1], PDModel.coef_[0])
    return PDResults


# Compute submission root relative to this script's location
SUBMISSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Data paths (using kilosort processed data)
data_folder = os.path.join(SUBMISSION_ROOT, 'nwb_kilosort', 'bohr', 'Brain_control')
base_path = os.path.join(SUBMISSION_ROOT, 'fig2', 'preprocessing', 'bohr_feedback')

all_df = []

for raw_dirname in tqdm(os.listdir(data_folder)):
    # Filter for 2DBC datasets only
    if not '2DBC' == raw_dirname.split('_')[1]:
        continue

    data_path = os.path.join(data_folder, raw_dirname, 'formatted_data', 'standard_data_bmi.nwb')
    if not os.path.exists(data_path):
        continue

    try:
        time_series_data = nap.load_file(data_path)
    except:
        continue

    print(f"Processing: {raw_dirname}")

    # Load data
    units = time_series_data['units']
    markers = time_series_data['BehaviorMarkers']
    pos = time_series_data['pos']
    training_flag = time_series_data['training flag']

    # Handle marker values
    marker_data = markers.d[:]
    if marker_data[0] < 0:
        marker_data = marker_data + 65280

    # Find trial boundaries
    trial_start = np.sort(np.where(marker_data == 24)[0])
    trial_end = np.sort(np.where(marker_data == 5)[0][0:len(trial_start)])

    # Extract trial data
    trial_marker = [markers[i:j+1] for i, j in zip(trial_start, trial_end)]
    trial_flag = [training_flag.get(i.t[0], i.t[-1]) for i in trial_marker]
    trial_pos = [pos.get(i.t[0], i.t[-1]) if sum(j.d[:]) == 0 else None
                 for i, j in zip(trial_marker, trial_flag)]

    # Build trial info dataframe
    trial_info = {
        'markers': trial_marker,
        'flags': trial_flag,
        'brain_control_pos': trial_pos,
    }
    trial_info_df = pd.DataFrame(trial_info)

    # Select successful trials with training flag = 0
    df = trial_info_df.dropna().loc[trial_info_df['markers'].apply(lambda x: 20 in x.d[:])]
    if len(df) == 0:
        continue

    marker_list = df['markers'].to_list()

    # Alignment time: trial start (marker 24)
    mo_times = np.array([i.t[np.where(i.d[:] == 24)[0][0]] for i in df['markers'].values])
    mo_times_ts = nap.Ts(t=mo_times, time_units='s')

    # Compute PETH
    brain_control_epochs = nap.IntervalSet(start=-1, end=1, time_units="s")
    peth0 = nap.compute_perievent(units, mo_times_ts, minmax=(-2, 2), time_unit="s")

    unit_index = np.array(list(peth0.keys()))
    if len(unit_index) == 0:
        continue

    is_array = [peth0[i].count(0.02).smooth(
        std=0.05, size_factor=5, time_units='s', norm=True
    ).restrict(brain_control_epochs) for i in unit_index]

    bin_time = is_array[0].t
    is_array = np.array([i.as_array() / 0.02 for i in is_array]).transpose((1, 2, 0))
    is_tsd = nap.TsdTensor(t=bin_time, d=is_array)

    # Align trial count: compute_perievent may drop boundary trials
    n_trials = is_tsd.d.shape[1]
    df = df.iloc[:n_trials]

    # Get target positions and calculate angles
    pos_array = np.array([i[-1] for i in df['brain_control_pos'].to_list()])
    angles = np.arctan2(pos_array[:, 1], pos_array[:, 0]) * 180 / np.pi
    groups = np.digitize(angles % 360, bins=np.linspace(0, 360, num=9))

    # Calculate PD for each cell at each time point
    x = (np.array(sorted(np.unique(groups))) * 45 - 22.5) / 180 * np.pi
    cells_ps = {}

    for ind, cell_ind in enumerate(unit_index):
        pd_list = []
        for t in is_tsd.t:
            fr_t_c = is_tsd.get(t)[:, ind]
            slice_conditions = [groups == group for group in sorted(np.unique(groups))]
            y = [fr_t_c[condition].mean() for condition in slice_conditions]
            pd3 = DiscreatePD(x, y)
            pd3['t'] = t
            pd_list.append(pd3)
        cells_ps[cell_ind] = pd_list

    # Create dataframe
    df_result = pd.DataFrame([
        {'cell': key, **subdict}
        for key, values in cells_ps.items()
        for subdict in values
    ])
    df_result['dataset'] = raw_dirname
    df_result.sort_values(by='t', ascending=True, inplace=True)
    all_df.append(df_result)

# Save results
if len(all_df) > 0:
    df_combined = pd.concat(all_df, axis=0, ignore_index=True)
    output_path = f'{base_path}/trial_start_pd.csv'
    df_combined.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print(f"Total cells: {df_combined['cell'].nunique()}")
else:
    print("No data processed!")
