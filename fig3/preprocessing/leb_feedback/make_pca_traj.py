#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA Trajectory Preprocessing for Fig3 - Leb Feedback (2DBC)
"""

import os
import pynapple as nap
import json
import pandas as pd
import numpy as np

def tsd_json_to_tsdframe(tsd):
    timestamps = tsd.index
    json_data = tsd.values
    parsed_data = [json.loads(entry) for entry in json_data]
    df = pd.json_normalize(parsed_data, sep='_')
    df.index = timestamps
    return df

def process_dataset(data_path, output_path):
    """Process a single dataset and save npz"""
    try:
        time_series_data = nap.load_file(data_path)

        units = time_series_data['units']
        markers = time_series_data['BehaviorMarkers']
        pos = time_series_data['pos']
        training_flag = time_series_data['training flag']

        # Slice data for analysis
        trial_start = np.sort(np.where(markers.d[:]==24)[0])
        trial_end = np.sort(np.where(markers.d[:]==5)[0][0:len(trial_start)])

        trial_marker = [markers[i:j+1] for i,j in zip(trial_start,trial_end)]
        trial_flag = [training_flag.get(i.t[0],i.t[-1]) for i in trial_marker]
        trial_pos = [pos.get(i.t[0],i.t[-1]) if sum(j.d[:])==0 else None
                     for i,j in zip(trial_marker, trial_flag)]

        def calculate_norms_and_mo_times(pos_segment):
            norms = np.linalg.norm(pos_segment.to_numpy(), axis=1)
            mo_indices = np.where(norms > 1.5)[0][0]
            return float(pos_segment.t[mo_indices])

        trial_mo = [calculate_norms_and_mo_times(i) if i is not None else None for i in trial_pos]

        trial_info = {
            'markers': trial_marker,
            'flags': trial_flag,
            'brain_control_pos': trial_pos,
            'movement_onset_times': trial_mo
        }
        trial_info_df = pd.DataFrame(trial_info)
        df = trial_info_df.dropna().loc[trial_info_df['markers'].apply(lambda x: 20 in x.d[:])]

        if len(df) == 0:
            return False, "No valid trials"

        marker_list = df['markers'].to_list()
        mo_times = df['movement_onset_times'].values
        mo_times_ts = nap.Ts(t=mo_times, time_units='s')

        left_border = -0.4
        right_border = 0.4
        brain_control_epochs = nap.IntervalSet(start=left_border-0.4, end=right_border, time_units="s")

        # Filter to kilosort2.5 M1+PMd units (Leb)
        units = units[units['sorter'] == 'kilosort2.5']
        units = units[(units['_location']=='M1')+(units['_location']=='PMd')]

        peth0 = nap.compute_perievent(units, mo_times_ts,
                                      minmax=(left_border-1, right_border+1), time_unit="s")
        unit_index = np.array(list(peth0.keys()))

        if len(unit_index) == 0:
            return False, "No units"

        is_array = [peth0[i].count(0.02).smooth(
                        std=0.05, size_factor=5, time_units='s', norm=False
                    ).restrict(brain_control_epochs) for i in unit_index]
        bin_time = is_array[0].t
        ntk_array = np.array([i.as_array()/0.02 for i in is_array]).transpose((2, 0, 1))
        pos_array = np.array([i[-1] for i in df['brain_control_pos'].to_list()])

        # PCA projection - normalize and center
        neuralData = ntk_array.copy()
        angles = np.arctan2(pos_array[:,1], pos_array[:,0]) * 180 / np.pi
        groups = np.digitize(angles % 360, bins=np.linspace(0, 360, num=9))
        ranges = np.max(neuralData, axis=(0, 2)) - np.min(neuralData, axis=(0, 2)) + 5

        normalized_neuralData = neuralData / ranges[None, :, None]
        mean_across_trials = np.mean(normalized_neuralData, axis=0, keepdims=True)
        normalized_neuralData_centered = normalized_neuralData - mean_across_trials
        group_means = np.array([np.mean(normalized_neuralData_centered[groups == g], axis=0)
                                for g in np.unique(groups)]).transpose((2, 0, 1))

        is_tsd = nap.TsdTensor(t=bin_time, d=group_means)
        is_tsd.save(output_path)

        return True, f"{len(unit_index)} units, {len(df)} trials"
    except Exception as e:
        return False, str(e)


# Compute submission root relative to this script's location
SUBMISSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Main
nwb_base = os.path.join(SUBMISSION_ROOT, 'nwb_kilosort', 'leb', 'Brain_control')
output_base = os.path.join(SUBMISSION_ROOT, 'fig3', 'preprocessing', 'leb_feedback')

# Get 2DBC datasets (feedback condition)
datasets = [d for d in os.listdir(nwb_base) if '2DBC' == d.split('_')[1]]
print(f"Found {len(datasets)} 2DBC datasets")

success_count = 0
for dataset in sorted(datasets):
    data_path = os.path.join(nwb_base, dataset, 'formatted_data', 'standard_data_bmi.nwb')
    output_path = os.path.join(output_base, f'{dataset}.npz')

    if not os.path.exists(data_path):
        print(f"  {dataset}: NWB not found")
        continue

    success, msg = process_dataset(data_path, output_path)
    status = "OK" if success else "FAIL"
    print(f"  {dataset}: {status} - {msg}")
    if success:
        success_count += 1

print(f"\nDone! {success_count}/{len(datasets)} datasets processed")
