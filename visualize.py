import matplotlib
# matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Optional
import json
import os

def plot_deg(results_path: Path, hookpoint:str, quantization:bool, lvl: int, batch_size: int,plt_params: dict):
    """
    Plot the degree of redundancy from the results file.

    Args:
        results_path (Path): Path to the results file.

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
        if params["hookpoints"][0] == hookpoint and \
            params["sae"]["quantization"] == quantization and \
           params["sae"]["quantization_levels"] == lvl and \
           params["batch_size"] == batch_size:
            x.append(params["sae"]["expansion_factor"])
            y.append(value["r"])
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
    plt.tight_layout()
    # plt.show()
    # plot_path = results_path.with_suffix('.png')
    # plt.savefig(plot_path)
    # plt.close()

    # print(f"Plot saved to {plot_path}")
#^ 
