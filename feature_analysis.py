"""
This module contains functions to check for precentage of feature splitting or 
composition amongst the novel latents in a bigger SAE compared to a smaller SAE. 
Is the feature splitting or composition more prevalent in the larger SAE?
It is useful for understanding how the SAE size (dictionary size) affects the feature 
extraction process. 
"""

from pathlib import Path
from typing import Optional
import orjson
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from sklearn.metrics import roc_auc_score, roc_curve
import json
from delphi.delphi.log.result_analysis import load_data
import numpy as np
import matplotlib.pyplot as plt


def load_explanations(path: Path, modules: list) -> pd.DataFrame:
    """
    Load explanations from files in the specified path for the given modules.
    Args:
        path (Path): The directory path where explanation files are stored.
        modules (list): List of SAE hookpoints.
    Returns:
        pd.DataFrame: A DataFrame containing the latent index and 
        explanation for each latent.
    """
    # Collect per-latent explanations
    explanation_dfs = []
    for module in modules:
        for file in path.glob(f"*{module}*"):
            latent_idx = int(file.stem.split("latent")[-1])
            explanation = orjson.loads(file.read_bytes())
            explanation_dfs.append(pd.DataFrame(
                [{"latent_idx": latent_idx, "explanation": explanation}]
            ))
    return pd.concat(explanation_dfs, ignore_index=True)

def _check_detection_non_activating(latent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for the number of non-zero activations for non-activating examples of 
    Detection scoring in the latent DataFrame.

    Args:
        latent_df (pd.DataFrame): DataFrame containing latent data.
        run_cfg (Optional[dict]): Configuration for the run, if any.
        
    Returns:
        counts of non-activating examples with activations greater than zero.
    """
    # Filter for non-activating examples
    data = latent_df[(latent_df["activating"] == False) & 
                     (latent_df["score_type"] == "detection")]
    
    # Calculate sum of activations
    norm_1 = data.activations.apply(lambda x: np.array(x).sum())

    return norm_1[norm_1 > 0].count()

def _load_sae_cfg(
    root_dir: Path,
    run_cfg: Optional[dict] = None,
) -> dict:
    """
    Load the SAE configuration from a JSON file.
    
    Args:
        root_dir (Path): The root directory where the SAE configuration is stored.
        run_cfg (Optional[dict]): Configuration for the run, if any.
        
    Returns:
        dict: The SAE configuration loaded from the JSON file.
    """
    sae_cfg_path = root_dir / "checkpoints" / run_cfg["name"] / run_cfg["hookpoints"][0] / "cfg.json"
    with open(sae_cfg_path, "r") as f:
        sae_cfg = json.load(f)
    
    return sae_cfg

def find_activating_tokens(df: pd.DataFrame, sae_cfg: dict, 
                           threshold: float = 1.0) -> np.ndarray:
    """
    Takes the latent data of the activating examples from detection scoring and returns 
    a padded array of activating tokens for each latent.
    
    Args:
        df (pd.DataFrame): DataFrame containing the latent data
        of the activating examples from detection scoring.
        sae_cfg (dict): Configuration for the SAE.
        threshold (float): Threshold for activation filtering.
        
    Returns:
        np.ndarray: A padded array of activating tokens for each latent.
    """
    width = sae_cfg["expansion_factor"] * sae_cfg["d_in"]
    act_tokens_list = []
    # Iterate over each latent index   
    for i in range(width):
        data_i = df[(df.latent_idx == i) & (df.correct == True)]
        _act_tokens = []
        for raw in data_i.itertuples():
            acts = np.array(raw.activations)
            tokens = np.array(raw.tokens)
            idx = acts > threshold
            _act_tokens.extend(tokens[idx].flatten().tolist())
        # Collect activation tokens for the current latent index
        act_tokens_list.append(np.array(_act_tokens))
    # Pad the activation tokens to the same length
    max_len = max(map(len, act_tokens_list))
    act_tokens = np.array([np.pad(arr, (0, max_len - len(arr)), constant_values=np.nan) 
                           for arr in act_tokens_list])
    
    return act_tokens

def common_tokens(lat1: np.array, lat2: np.array) -> int:
    """
    The common tokens between two arrays.

    Args:
        lat1 (np.array): tokens of the first latent.
        lat2 (np.array): tokens of the second latent.

    Returns:
        set: Common tokens between the two latents.
    """
    # ignore the NaN values in the latents
    lat1 = lat1[~np.isnan(lat1)]
    lat2 = lat2[~np.isnan(lat2)]
    # convert to sets and find the intersection
    set1 = set(lat1)
    set2 = set(lat2)
    return set1.intersection(set2)