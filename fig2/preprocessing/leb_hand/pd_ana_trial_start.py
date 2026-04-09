#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PD analysis aligned to trial start for Monkey Leb Manual Control

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
data_folder = os.path.join(SUBMISSION_ROOT, 'nwb_kilosort', 'leb', 'Data_recording')
base_path = os.path.join(SUBMISSION_ROOT, 'fig2', 'preprocessing', 'leb_hand')

all_df = []

for raw_dirname in tqdm(os.listdir(data_folder)):
    data_path = os.path.join(data_folder, raw_dirname, 'formatted_data', 'standard_data_manual.nwb')
    if not os.path.exists(data_path):
        continue

    try:
        time_series_data = nap.load_file(data_path)
    except:
        continue

    print(f"Processing: {raw_dirname}")

    # Load data
    units = time_series_data['units']
    trials = time_series_data['trials']

    # Select successful trials
    success_trials = trials[trials['result_label'] == 'success']
    if len(success_trials) == 0:
        continue

    # Alignment time: trial start
    mo_times = success_trials['start']
    mo_times_ts = nap.Ts(t=mo_times.tolist(), time_units='s')

    # Compute PETH
    behavior_epochs = nap.IntervalSet(start=-1, end=1, time_units="s")
    # Filter to kilosort2.5 M1+PMd units (Leb)
    units = units[units['sorter'] == 'kilosort2.5']
    units = units[(units['_location']=='M1')+(units['_location']=='PMd')]

    peth0 = nap.compute_perievent(units, mo_times_ts, minmax=(-2, 2), time_unit="s")

    unit_index = np.array(list(peth0.keys()))
    if len(unit_index) == 0:
        continue

    is_array = [peth0[i].count(0.02).smooth(
        std=0.05, size_factor=5, time_units='s', norm=True
    ).restrict(behavior_epochs) for i in unit_index]

    bin_time = is_array[0].t
    is_array = np.array([i.as_array() / 0.02 for i in is_array]).transpose((1, 2, 0))
    is_tsd = nap.TsdTensor(t=bin_time, d=is_array)

    # Align trial count: compute_perievent may drop boundary trials
    n_trials = is_tsd.d.shape[1]
    # df = df.iloc[:n_trials]

    # Get target positions and calculate angles
    pos_array = np.array(success_trials['feedback_pos'].to_list())
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
    df = pd.DataFrame([
        {'cell': key, **subdict}
        for key, values in cells_ps.items()
        for subdict in values
    ])
    df['dataset'] = raw_dirname
    df.sort_values(by='t', ascending=True, inplace=True)
    all_df.append(df)

# Save results
if len(all_df) > 0:
    df_combined = pd.concat(all_df, axis=0, ignore_index=True)
    output_path = f'{base_path}/trial_start_pd.csv'
    df_combined.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    print(f"Total cells: {df_combined['cell'].nunique()}")
else:
    print("No data processed!")
