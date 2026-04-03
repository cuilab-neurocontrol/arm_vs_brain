#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subspace Analysis for Fig3
Computes orthogonal subspaces for prep and execution periods
"""

import pynapple as nap
import numpy as np
import pymanopt
import pymanopt.manifolds
import pymanopt.optimizers
from pymanopt.manifolds import Stiefel
from pymanopt import Problem
from pymanopt.optimizers import TrustRegions
from scipy.linalg import svd
import pandas as pd
import autograd.numpy as anp
import itertools
import os
import argparse

def pair_subspace(t1, t2, width, path):
    is_tsd = nap.load_file(path)
    trial_time = is_tsd.t[0] + width

    if t1 < trial_time:
        return []
    if t2 < trial_time:
        return []
    if t2 < t1:
        return []

    prepData = is_tsd.get(t1 - 0.1, t1).transpose((1, 2, 0))
    moveData = is_tsd.get(t2 - 0.1, t2).transpose((1, 2, 0))

    _, numNeurons, _ = moveData.shape

    prepData = prepData.reshape(numNeurons, -1)
    moveData = moveData.reshape(numNeurons, -1)

    C_prep = np.cov(prepData)
    C_move = np.cov(moveData)

    s_p = svd(C_prep, compute_uv=False)
    s_m = svd(C_move, compute_uv=False)

    dim_p = 6
    dim_m = 6

    n = len(C_prep)

    manifold = Stiefel(n, dim_p + dim_m)

    @pymanopt.function.autograd(manifold)
    def cost(x):
        term1 = np.trace(x[:, :dim_p].T @ C_prep @ x[:, :dim_p]) / np.sum(s_p[:dim_p])
        term2 = np.trace(x[:, dim_p:dim_p + dim_m].T @ C_move @ x[:, dim_p:dim_p + dim_m]) / np.sum(s_m[:dim_m])
        return -(1/2) * (term1 + term2)

    @pymanopt.function.autograd(manifold)
    def egrad(x):
        grad_prep = C_prep @ x[:, :dim_p] / np.sum(s_p[:dim_p])
        grad_move = C_move @ x[:, dim_p:dim_p + dim_m] / np.sum(s_m[:dim_m])
        return -np.hstack((grad_prep, grad_move))

    problem = Problem(manifold, cost, euclidean_gradient=egrad)
    optimizer = TrustRegions(verbosity=0)
    result = optimizer.run(problem)

    Q_prep = result.point[:, 0:dim_p]
    Q_move = result.point[:, dim_p:(dim_p + dim_m)]
    ort = np.linalg.norm(Q_prep.T @ Q_move, ord=2) < 10e-5
    unit1 = np.linalg.norm(Q_prep.T @ Q_prep - np.eye(Q_prep.shape[-1]), ord=2) < 10e-5
    unit2 = np.linalg.norm(Q_move.T @ Q_move - np.eye(Q_move.shape[-1]), ord=2) < 10e-5
    assert ort and unit1 and unit2, print('Q_prep and Q_move are not orthogonal and unit\n')

    vpps = np.trace(Q_prep.T @ C_prep @ Q_prep) / np.sum(s_p[0:dim_p]) * 100
    vmps = np.trace(Q_prep.T @ C_move @ Q_prep) / np.sum(s_m[0:dim_m]) * 100
    vmms = np.trace(Q_move.T @ C_move @ Q_move) / np.sum(s_m[0:dim_m]) * 100
    vpms = np.trace(Q_move.T @ C_prep @ Q_move) / np.sum(s_p[0:dim_p]) * 100

    P = prepData.T
    M = moveData.T

    PM = P @ Q_move
    MP = M @ Q_prep
    PP = P @ Q_prep
    MM = M @ Q_move

    SPM = np.linalg.eig(np.cov(PM.T))
    SMP = np.linalg.eig(np.cov(MP.T))
    SPP = np.linalg.eig(np.cov(PP.T))
    SMM = np.linalg.eig(np.cov(MM.T))

    SPMR = SPM[0][0:5] / np.sum(s_m[0:dim_m]) * 100
    SMPR = SMP[0][0:5] / np.sum(s_p[0:dim_p]) * 100
    SPPR = SPP[0][0:5] / np.sum(s_p[0:dim_p]) * 100
    SMMR = SMM[0][0:5] / np.sum(s_m[0:dim_m]) * 100
    w, _, _ = np.linalg.svd(moveData, full_matrices=False)
    A = np.trace(w[:, 0:10].T @ C_prep @ w[:, 0:10]) / np.sum(s_p[0:10])

    data_all = {}
    data_all['vpps'] = vpps
    data_all['vmps'] = vmps
    data_all['vmms'] = vmms
    data_all['vpms'] = vpms
    data_all['alignment_index'] = A

    data_all['SPMR'] = SPMR
    data_all['SMPR'] = SMPR
    data_all['SPPR'] = SPPR
    data_all['SMMR'] = SMMR
    data_all['pc_index'] = range(len(SMMR))
    data_all['t1'] = t1
    data_all['t2'] = t2

    return data_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Subspace analysis")
    parser.add_argument("-f", "--file", type=str, required=True, help="Input npz file")
    parser.add_argument("-p", "--path", type=str, required=True, help="Input path")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output path")
    args = parser.parse_args()

    p = os.path.basename(args.file)
    path = args.path
    output_path = args.output

    results_list = []

    if 'npz' in p:
        input_path = os.path.join(path, p)
        is_tsd = nap.load_file(input_path)
        pair_combinations = list(itertools.combinations(
            np.arange(round(is_tsd.t[0], 1), round(is_tsd.t[-1], 1) + 0.05, 0.05), 2))
        name = p.split('.')[0]

        print(f"Processing {name}: {len(pair_combinations)} pairs")

        for i, pair in enumerate(pair_combinations):
            results = pair_subspace(pair[0], pair[1], 0.05, input_path)
            if len(results) != 0:
                results['dataset'] = name
                results['perp_t'] = pair[0]
                results['exec_t'] = pair[1]
                results_list.append(results)

        if results_list:
            results = [pd.DataFrame(i) for i in results_list if len(i) != 0]
            final_df = pd.concat(results, ignore_index=True)
            output_file = os.path.join(output_path, f'subspace_{name}.csv')
            final_df.to_csv(output_file, index=False)
            print(f"Saved: {output_file}")
        else:
            print(f"No results for {name}")
