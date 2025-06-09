# -*- coding: utf-8 -*-
"""
# visualize.py
This module provides functions to visualize the degree of redundancy
in the results of experiments, specifically focusing on the expansion factor and its impact on redundancy metrics.
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

def plot_deg(results_path: Path, exp_params:dict, plt_params: dict):
    """
    Plot the degree of redundancy from the results file.
    It plots the expansion factor against the degree of redundancy.

    Args:
        results_path (Path): Path to the results file.
        exp_params (dict): Experiment parameters.
        plt_params (dict): Plotting parameters.
    Returns:
        None
    """
    if not results_path.exists():
        print(f"Results file {results_path} does not exist.")
        return None
    #^
    with open(results_path, 'r') as f:
        results = json.load(f)
    #^
    x = []
    y = []
    for value in results.values():
        params = value["train_config"]
        if params["hookpoints"][0] == exp_params['hkpt'] and \
        params["sae"]["quantization"] == exp_params['qt'] and \
        params["sae"]["quantization_levels"] == exp_params['lvls'] and \
        params["sae"]["k"] == exp_params['k'] and \
        params["batch_size"] == exp_params['b_sz']:
            x.append(params["sae"]["expansion_factor"])
            if exp_params['deg'] == 'r':
                y.append(value["r"])
            elif exp_params['deg'] == 'v':
                y.append(value["v"])
        #^
    #^
    x = np.array(x)
    y = np.array(y)
    ###
    # expansion factor in increasing order
    sort_idx = np.argsort(x)
    x= x[sort_idx]
    y = y[sort_idx]
    ###
    plt.figure(figsize=(10, 5))
    plt.plot(x,y, marker='o', linestyle='-', color='b')
    plt.title(plt_params['title'])
    plt.xlabel(plt_params['xlabel'])
    plt.ylabel(plt_params['ylabel'])
    plt.xticks(plt_params['xticks'])
    plt.xlim(plt_params['xlim'])
    plt.yticks(plt_params['yticks'])
    plt.ylim(plt_params['ylim'])
    plt.grid(plt_params['grid'])
    if plt_params['xscale'][0]:
        plt.xscale(plt_params['xscale'][1], base=plt_params['xscale'][2])
    if plt_params['yscale'][0]: 
        plt.yscale(plt_params['yscale'][1], base=plt_params['yscale'][2])
    plt.tight_layout()
    plt.show()
    # plot_path = results_path.with_suffix('.png')
    # plt.savefig(plot_path)
    plt.close()

    # print(f"Plot saved to {plot_path}")
#^ 

def plot_deg_scan(results_path: Path, exp_params:dict, scan_param:dict, plt_params: dict):
    """
    Plot the degree of redundancy from the results file.
    It scans a parameter and plots the expansion factor against the degree of redundancy.

    Args:
        results_path (Path): Path to the results file.
        exp_params (dict): Experiment parameters.
        scan_param (dict): Scanning parameters.
        plt_params (dict): Plotting parameters.
    Returns:
        None
    """
    if not results_path.exists():
        print(f"Results file {results_path} does not exist.")
        return None
    #^
    with open(results_path, 'r') as f:
        results = json.load(f)
    #^
    
    assert len(scan_param.keys()) == 2, "Only one scanning parameter and its type are allowed."
    scan_key = list(scan_param.keys())[1]
    assert scan_key not in exp_params['train'].keys(), \
              f"Scanning parameter {scan_key} must not be in exp_params['train']."
    assert scan_key not in exp_params['sae'].keys(), \
              f"Scanning parameter {scan_key} must not be in  exp_params['sae']."

    keys = {}
    for key in exp_params['train'].keys():
        keys[key] = 'train'
    for key in exp_params['sae'].keys():
        keys[key] = 'sae'
    #^
    x_ = -1*np.ones((len(scan_param[scan_key]),1))
    y_ = -1*np.ones((len(scan_param[scan_key]),1))
    for value in results.values():
        params = value["train_config"]
        lis = []
        exp_list = []
        for key in keys:
            if keys[key] == 'train':
                val = params[key]
                exp_val = exp_params['train'][key] 
            elif keys[key] == 'sae':
                val = params["sae"][key]
                exp_val = exp_params['sae'][key]
            lis.append(val)
            exp_list.append(exp_val)
        #^
        if lis != exp_list: continue
        #^
        if scan_param["type"] == 'train':
            if params[scan_key] in scan_param[scan_key]:
                idx = scan_param[scan_key].index(params[scan_key])
                _x = -1*np.ones((len(scan_param[scan_key]),1))
                _x[idx] = params["sae"]["expansion_factor"]
                x_ = np.hstack((x_,_x)) 
                if exp_params['deg'] == 'r':
                    _y = -1*np.ones((len(scan_param[scan_key]),1))
                    _y[idx] = value["r"]
                    y_ = np.hstack((y_,_y))
                elif exp_params['deg'] == 'v':
                    _y = -1*np.ones((len(scan_param[scan_key]),1))
                    _y[idx] = value["v"]
                    y_ = np.hstack((y_,_y))
        elif scan_param["type"] == 'sae':
            if params['sae'][scan_key] in scan_param[scan_key]:
                idx = scan_param[scan_key].index(params['sae'][scan_key])
                _x = -1*np.ones((len(scan_param[scan_key]),1))
                _x[idx] = params["sae"]["expansion_factor"]
                x_ = np.hstack((x_,_x)) 
                if exp_params['deg'] == 'r':
                    _y = -1*np.ones((len(scan_param[scan_key]),1))
                    _y[idx] = value["r"]
                    y_ = np.hstack((y_,_y))
                elif exp_params['deg'] == 'v':
                    _y = -1*np.ones((len(scan_param[scan_key]),1))
                    _y[idx] = value["v"]
                    y_ = np.hstack((y_,_y))  
    #^
    # get the number of non -1 for any row (all rows should have the same number)
    mask = (x_[0,:] != -1)
    x = np.empty((len(scan_param[scan_key]), np.sum(mask)))
    y = np.empty((len(scan_param[scan_key]), np.sum(mask)))
    for i in range(len(scan_param[scan_key])):
        mask = (x_[i,:] != -1)
        assert np.sum(mask) == np.sum(x_[0,:] != -1), \
              f"Row {i} has a different number of non -1 values than row 0."
        #^
        assert np.sum(mask) == np.sum(y_[0,:] != -1), \
              f"Row {i} has a different number of non -1 values than row 0."
        assert np.sum(mask) > 0, \
              f"Row {i} has no non -1 values."
        #^
        assert np.sum(y_[i,:] != -1) > 0, \
              f"Row {i} has no non -1 values."
        #^
        x[i,:] = x_[i,mask]
        y[i,:] = y_[i,mask]
        # expansion factor in increasing order
        sort_idx = np.argsort(x[i,:])
        x[i,:] = x[i,sort_idx]
        y[i,:] = y[i,sort_idx]
    #^
    ### plotting
    
    plt.figure(figsize=(10, 5))
    for i in range(len(scan_param[scan_key])):
        plt.plot(x[i,:], y[i,:], marker='o', linestyle='-', label=f"{scan_key}={scan_param[scan_key][i]}")
    #^
    plt.title(plt_params['title'])
    plt.xlabel(plt_params['xlabel'])
    plt.ylabel(plt_params['ylabel'])
    plt.xticks(plt_params['xticks'])
    plt.xlim(plt_params['xlim'])
    plt.yticks(plt_params['yticks'])
    plt.ylim(plt_params['ylim'])
    plt.grid(plt_params['grid'])
    if plt_params['xscale'][0]:
        plt.xscale(plt_params['xscale'][1], base=plt_params['xscale'][2])
    if plt_params['yscale'][0]: 
        plt.yscale(plt_params['yscale'][1], base=plt_params['yscale'][2])
    if plt_params['legend']:
        plt.legend()
    plt.tight_layout()
    plt.show()
    # plot_path = results_path.with_suffix('.png')
    # plt.savefig(plot_path)
    plt.close()

    # print(f"Plot saved to {plot_path}")
#^ 
