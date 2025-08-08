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
#^
def _check_dection_non_activating(
    latent_df: pd.DataFrame,
    run_cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Check the detection of non-activating examples in the latent DataFrame.
    
    Args:
        latent_df (pd.DataFrame): DataFrame containing latent data.
        run_cfg (Optional[dict]): Configuration for the run, if any.
        
    Returns:
        counts of non-activating examples with activations greater than zero.
    """
    # Filter for non-activating examples
    data = latent_df[(latent_df["activating"] == False) & (latent_df["score_type"] == "detection")]
    
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