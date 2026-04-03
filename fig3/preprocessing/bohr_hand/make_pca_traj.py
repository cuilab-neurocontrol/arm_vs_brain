#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA Trajectory Preprocessing for Fig3 - Bohr Hand (Interception)
"""

import os
import pynapple as nap
import pandas as pd
import numpy as np

def process_dataset(data_path, output_path, time_offset=-0.108):
    """Process a single dataset and save npz"""
    try:
        time_series_data = nap.load_file(data_path)

        trials = time_series_data['trials']
        success_trial = trials[trials.result_label == 'success']

        if len(success_trial) == 0:
            return False, "No success trials"

        left_border = -0.5
        right_border = 0.5
        brain_control_epochs = nap.IntervalSet(start=left_border, end=right_border, time_units="s")

        units = time_series_data['units']
        # Apply time offset for hand data
        mo_times_ts = nap.Ts(t=np.array(success_trial.movement_onset_time.to_list()) + time_offset, time_units='s')

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

        pos_array = np.array(success_trial.target_pos_touch.to_list()).squeeze()
        angles = np.arctan2(pos_array[:,1], pos_array[:,0]) * 180 / np.pi
        groups = np.digitize(angles % 360, bins=np.linspace(0, 360, num=9))

        # PCA projection - normalize and center
        neuralData = ntk_array
        ranges = np.max(neuralData, axis=(0, 2)) - np.min(neuralData, axis=(0, 2)) + 5

        normalized_neuralData = neuralData / ranges[None, :, None]
        mean_across_trials = np.mean(normalized_neuralData, axis=0, keepdims=True)
        normalized_neuralData_centered = normalized_neuralData - mean_across_trials
        group_means = np.array([np.mean(normalized_neuralData_centered[groups == g], axis=0)
                                for g in np.unique(groups)]).transpose((2, 0, 1))

        is_tsd = nap.TsdTensor(t=bin_time, d=group_means)
        is_tsd.save(output_path)

        return True, f"{len(unit_index)} units, {len(success_trial)} trials"
    except Exception as e:
        return False, str(e)


# Compute submission root relative to this script's location
SUBMISSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# Main
nwb_base = os.path.join(SUBMISSION_ROOT, 'nwb_kilosort', 'bohr', 'Data_recording')
output_base = os.path.join(SUBMISSION_ROOT, 'fig3', 'preprocessing', 'bohr_hand')

# Get Interception datasets (exclude Semi2DBC120 and 2DBC)
datasets = [d for d in os.listdir(nwb_base)
            if 'Interception' in d or 'interception' in d]
print(f"Found {len(datasets)} Interception datasets")

success_count = 0
for dataset in sorted(datasets):
    data_path = os.path.join(nwb_base, dataset, 'formatted_data', 'standard_data_manual.nwb')
    output_path = os.path.join(output_base, f'{dataset}.npz')

    if not os.path.exists(data_path):
        print(f"  {dataset}: NWB not found")
        continue

    success, msg = process_dataset(data_path, output_path, time_offset=-0.1)
    status = "OK" if success else "FAIL"
    print(f"  {dataset}: {status} - {msg}")
    if success:
        success_count += 1

print(f"\nDone! {success_count}/{len(datasets)} datasets processed")
