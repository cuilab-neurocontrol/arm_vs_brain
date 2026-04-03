#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCA Analysis for Fig4c
Supports feedforward/feedback, prep/exec epochs
Uses multiprocessing for acceleration
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from multiprocessing import Pool, cpu_count
import argparse

# Compute submission root relative to this script's location
SUBMISSION_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# NOTE: tools/ directory with ccaTools and dataTools must be in the Python path
sys.path.insert(0, os.path.join(SUBMISSION_ROOT, 'fig4', 'c'))

import pynapple as nap
from scipy.stats import wilcoxon
from tools import dataTools as dt
from tools.ccaTools import get_ccs, get_ccs_lower_bound_monkey, get_ccs_upper_bound
import params

monkey_defs = params.monkey_defs


def load_bc_data(dataset_path, dataset_name, filter_location=False):
    """Load brain control dataset"""
    data_path = os.path.join(dataset_path, dataset_name, 'formatted_data', 'standard_data_bmi.nwb')
    if not os.path.exists(data_path):
        return None

    time_series_data = nap.load_file(data_path)
    units = time_series_data['units']
    markers = time_series_data['BehaviorMarkers']
    pos = time_series_data['pos']
    training_flag = time_series_data['training flag']

    trial_start = np.sort(np.where(markers.d[:]==24)[0])
    trial_end = np.sort(np.where(markers.d[:]==5)[0][0:len(trial_start)])

    trial_marker = [markers[i:j+1] for i,j in zip(trial_start,trial_end)]
    trial_flag = [training_flag.get(i.t[0],i.t[-1]) for i in trial_marker]
    trial_pos = [pos.get(i.t[0],i.t[-1]) if sum(j.d[:])==0 else None
                 for i,j in zip(trial_marker, trial_flag)]

    def calculate_mo_times(pos_segment):
        norms = np.linalg.norm(pos_segment.to_numpy(), axis=1)
        mo_indices = np.where(norms > 1.5)[0][0]
        return float(pos_segment.t[mo_indices])

    trial_mo = [calculate_mo_times(i) if i is not None else None for i in trial_pos]

    trial_info_df = pd.DataFrame({
        'markers': trial_marker,
        'flags': trial_flag,
        'brain_control_pos': trial_pos,
        'movement_onset_times': trial_mo
    })
    df = trial_info_df.dropna().loc[trial_info_df['markers'].apply(lambda x: 20 in x.d[:])]

    if len(df) == 0:
        return None

    # Sort by movement onset times (as in original Leb scripts)
    df = df.sort_values(by='movement_onset_times')

    mo_times = df['movement_onset_times'].values
    mo_times_ts = nap.Ts(t=mo_times, time_units='s')

    # Compute dynamic window from GO signal and success marker
    marker_list = df['markers'].to_list()
    t_left = np.array([m.t[m.d[:]==3] for m in marker_list]).squeeze()
    t_right = np.array([m.t[m.d[:]==20] for m in marker_list]).squeeze()
    left_border = -np.mean(mo_times - t_left)
    right_border = np.mean(t_right - mo_times)

    kilosort_unit = units[units['sorter'] == "kilosort2.5"]
    if filter_location:
        kilosort_unit = kilosort_unit[(kilosort_unit['_location']=='M1')+(kilosort_unit['_location']=='PMd')]
    behavior_epochs = nap.IntervalSet(start=-0.3, end=0.4, time_units="s")
    peth0 = nap.compute_perievent(kilosort_unit, mo_times_ts, minmax=(left_border-1, right_border+1), time_unit="s")

    unit_index = np.array(list(peth0.keys()))
    is_array = [peth0[i].count(0.02).smooth(std=0.06, size_factor=10, time_units='s', norm=False
                ).restrict(behavior_epochs) for i in unit_index]
    bin_time = is_array[0].t
    is_array = np.array([i.as_array()/0.02 for i in is_array]).transpose((1, 2, 0))
    is_tsd = nap.TsdTensor(t=bin_time, d=is_array)

    # Align trial count: compute_perievent may drop boundary trials
    n_trials = is_tsd.d.shape[1]
    df = df.iloc[:n_trials]

    pos_array = np.array([i[-1] for i in df['brain_control_pos'].to_list()])
    angles = np.arctan2(pos_array[:,1], pos_array[:,0]) * 180 / np.pi
    groups = np.digitize(angles % 360, bins=np.linspace(0, 360, num=9))

    return pd.DataFrame({
        'MCx_rates': list(is_tsd.d.transpose(1,0,-1)),
        'target_id': groups,
        'trial_id': np.arange(is_tsd.d.shape[1]),
        'idx_movement_on': [np.argmin(np.abs(is_tsd.t-0))] * is_tsd.d.shape[1],
        'monkey': ['BC'] * is_tsd.d.shape[1]
    })


def load_mc_data(dataset_path, dataset_name, time_offset=-0.1, filter_location=False):
    """Load manual control dataset"""
    data_path = os.path.join(dataset_path, dataset_name, 'formatted_data', 'standard_data_manual.nwb')
    if not os.path.exists(data_path):
        return None

    time_series_data = nap.load_file(data_path)
    units = time_series_data['units']
    trials = time_series_data['trials']
    success_trials = trials[trials['result_label']=='success']

    if len(success_trials) == 0:
        return None

    mo_times = success_trials['movement_onset_time'] + time_offset
    mo_times_ts = nap.Ts(t=mo_times.to_list(), time_units='s')

    kilosort_unit = units[units['sorter'] == "kilosort2.5"]
    if filter_location:
        kilosort_unit = kilosort_unit[(kilosort_unit['_location']=='M1')+(kilosort_unit['_location']=='PMd')]
    behavior_epochs = nap.IntervalSet(start=-0.3, end=0.4, time_units="s")
    peth0 = nap.compute_perievent(kilosort_unit, mo_times_ts, minmax=(-2, 2), time_unit="s")

    unit_index = np.array(list(peth0.keys()))
    is_array = [peth0[i].count(0.02).smooth(std=0.06, size_factor=5, time_units='s', norm=False
                ).restrict(behavior_epochs) for i in unit_index]
    bin_time = is_array[0].t
    is_array = np.array([i.as_array()/0.02 for i in is_array]).transpose((1, 2, 0))
    is_tsd = nap.TsdTensor(t=bin_time, d=is_array)

    # Align trial count: compute_perievent may drop boundary trials
    n_trials = is_tsd.d.shape[1]

    pos_array = np.array(success_trials['feedback_pos'].to_list())[:n_trials]
    angles = np.arctan2(pos_array[:,1], pos_array[:,0]) * 180 / np.pi
    groups = np.digitize(angles % 360, bins=np.linspace(0, 360, num=9))

    return pd.DataFrame({
        'MCx_rates': list(is_tsd.d.transpose(1,0,-1)),
        'target_id': groups,
        'trial_id': np.arange(is_tsd.d.shape[1]),
        'idx_movement_on': [np.argmin(np.abs(is_tsd.t-0))] * is_tsd.d.shape[1],
        'monkey': ['MC'] * is_tsd.d.shape[1]
    })


def run_cca_analysis(allDFs, prep=False):
    """Run CCA analysis"""
    # Reset index for all DataFrames to ensure index starts from 0
    allDFs = [df.reset_index(drop=True) for df in allDFs]

    defs = monkey_defs
    pairFileList = dt.get_paired_files_monkey(allDFs)

    pair_side1df = [allDFs[i] for i,_ in pairFileList]
    pair_side2df = [allDFs[j] for _,j in pairFileList]
    side1df = allDFs

    if prep:
        len_trial = int(np.round(np.diff(defs.WINDOW_prep)/defs.BIN_SIZE))
        epoch = defs.prep_epoch
    else:
        len_trial = int(np.round(np.diff(defs.WINDOW_exec)/defs.BIN_SIZE))
        epoch = defs.exec_epoch

    n_components = defs.n_components
    area = defs.areas[2]

    allCCs = get_ccs(pair_side1df, pair_side2df, epoch, area, n_components, use_procrustes=False)
    CCsL = get_ccs_lower_bound_monkey(pair_side1df, pair_side2df, area, n_components, len_trial, use_procrustes=False)
    CCsU = get_ccs_upper_bound(side1df, epoch, area, n_components, use_procrustes=False)

    return allCCs, CCsL, CCsU


def export_results(allCCs, CCsL, CCsU, output_dir, suffix=''):
    """Export CCA results to CSV"""
    # CCA results
    records = []
    for name, data_arr in [('allCCs', allCCs), ('CCsL', CCsL), ('CCsU', CCsU)]:
        n_comps = min(10, data_arr.shape[0])
        for i in range(n_comps):
            row_data = data_arr[i, :]
            mean_val = np.mean(row_data)
            sem = stats.sem(row_data)
            ci = 1.96 * sem
            records.append({
                'x': i + 1,
                'condition': name,
                'y': mean_val,
                'CI_left': mean_val - ci,
                'CI_right': mean_val + ci
            })

    csv_path = os.path.join(output_dir, f'cca_results{suffix}.csv')
    pd.DataFrame(records).to_csv(csv_path, index=False)

    # Diff results
    diff_vals = np.abs(np.diff(allCCs.T).T)
    diff_vals_5 = diff_vals[:5, :]

    diff_records = []
    for i in range(diff_vals_5.shape[0]):
        row_data = diff_vals_5[i, :]
        mean_val = np.mean(row_data)
        sem = stats.sem(row_data)
        ci = 1.96 * sem
        diff_records.append({
            'x': i + 1,
            'condition': 'diff_allCCs',
            'y': mean_val,
            'CI_left': mean_val - ci,
            'CI_right': mean_val + ci
        })

    diff_csv_path = os.path.join(output_dir, f'diff_allCCs_results{suffix}.csv')
    pd.DataFrame(diff_records).to_csv(diff_csv_path, index=False)

    return csv_path, diff_csv_path


def get_datasets_by_pattern(base_path, pattern):
    """Get dataset names matching pattern from directory"""
    datasets = []
    if os.path.exists(base_path):
        for d in os.listdir(base_path):
            nwb_path = os.path.join(base_path, d, 'formatted_data')
            if os.path.isdir(nwb_path) and pattern in d:
                datasets.append(d)
    return sorted(datasets)


def process_condition(args):
    """Process a single condition (for multiprocessing)"""
    condition, monkey, prep, output_dir = args

    print(f"Processing: {monkey}_{condition}_{'prep' if prep else 'exec'}")

    # Base path for nwb_kilosort data
    nwb_base = os.path.join(SUBMISSION_ROOT, 'nwb_kilosort')

    if monkey == 'bohr':
        bc_path = f'{nwb_base}/bohr/Brain_control'
        mc_path = f'{nwb_base}/bohr/Data_recording'
        time_offset = -0.1
    else:  # leb
        bc_path = f'{nwb_base}/leb/Brain_control'
        mc_path = f'{nwb_base}/leb/Data_recording'
        time_offset = -0.1

    df_all = []

    # Load BC data based on condition
    if condition == 'feedforward':
        # Feedforward BC (Semi2DBC120 pattern)
        bc_datasets = get_datasets_by_pattern(bc_path, 'Semi2DBC120')
        print(f"  Found {len(bc_datasets)} feedforward BC datasets")
    else:  # feedback
        # Feedback BC (2DBC but not Semi)
        all_bc = [d for d in os.listdir(bc_path) if os.path.isdir(os.path.join(bc_path, d))]
        bc_datasets = sorted([d for d in all_bc if '2DBC' in d and 'Semi' not in d])
        print(f"  Found {len(bc_datasets)} feedback BC datasets")

    # Leb uses location filter (M1+PMd) and minimum trial counts
    filter_loc = (monkey == 'leb')
    min_bc_trials = 150 if monkey == 'leb' else 0
    min_mc_trials = 200 if monkey == 'leb' else 0

    for ds in bc_datasets:
        df = load_bc_data(bc_path, ds, filter_location=filter_loc)
        if df is not None and len(df) >= min_bc_trials:
            df_all.append(df)

    # Both conditions load MC data (Interception pattern)
    mc_datasets = get_datasets_by_pattern(mc_path, 'Interception')
    print(f"  Found {len(mc_datasets)} MC datasets")
    for ds in mc_datasets:
        df = load_mc_data(mc_path, ds, time_offset, filter_location=filter_loc)
        if df is not None and len(df) >= min_mc_trials:
            df_all.append(df)

    print(f"  Loaded {len(df_all)} total dataframes")
    if len(df_all) < 2:
        print(f"  Not enough data for {monkey}_{condition}")
        return None

    # Leb feedforward: optimize target groups (drop group with fewest trials)
    if monkey == 'leb' and condition == 'feedforward':
        group_ids = range(1, 9)
        dataset_counts = [df['target_id'].value_counts().to_dict() for df in df_all]

        best_group_to_drop = None
        max_retained = -1
        for drop_g in group_ids:
            retained = 0
            for counts in dataset_counts:
                if all(counts.get(g, 0) >= 5 for g in group_ids if g != drop_g):
                    retained += 1
            if retained > max_retained:
                max_retained = retained
                best_group_to_drop = drop_g

        print(f"  Group optimization: drop group {best_group_to_drop}, retain {max_retained}/{len(df_all)} datasets")

        final_df_all = []
        for df, counts in zip(df_all, dataset_counts):
            if all(counts.get(g, 0) >= 5 for g in group_ids if g != best_group_to_drop):
                df_filtered = df[df['target_id'] != best_group_to_drop].reset_index(drop=True)
                final_df_all.append(df_filtered)
        df_all = final_df_all
        print(f"  After optimization: {len(df_all)} dataframes")

    # Run CCA
    try:
        allCCs, CCsL, CCsU = run_cca_analysis(df_all, prep=prep)

        # Export - format: _{condition}_{monkey} or _{condition}_prep_{monkey}
        suffix = f"_{condition}"
        if prep:
            suffix += "_prep"
        suffix += f"_{monkey}"

        csv_path, diff_path = export_results(allCCs, CCsL, CCsU, output_dir, suffix)
        print(f"  Saved: {diff_path}")
        return diff_path
    except Exception as e:
        import traceback
        print(f"  Error: {e}")
        traceback.print_exc()
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--monkey', type=str, choices=['bohr', 'leb'], required=True)
    parser.add_argument('--condition', type=str, choices=['feedforward', 'feedback'], required=True)
    parser.add_argument('--prep', action='store_true', help='Use prep epoch instead of exec')
    args = parser.parse_args()

    output_dir = os.path.join(SUBMISSION_ROOT, 'fig4', 'c')
    os.makedirs(output_dir, exist_ok=True)

    print(f"=== CCA Analysis for Fig4c ===")
    print(f"Monkey: {args.monkey}, Condition: {args.condition}, Prep: {args.prep}\n")

    # Run single condition
    result = process_condition((args.condition, args.monkey, args.prep, output_dir))

    print("\n=== Done ===")
    if result:
        print(f"  {result}")
