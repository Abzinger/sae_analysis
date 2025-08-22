"""
This module contains functions to check for precentage of feature splitting or 
composition amongst the novel latents in a bigger SAE compared to a smaller SAE. 
Is the feature splitting or composition more prevalent in the larger SAE?
It is useful for understanding how the SAE size (dictionary size) affects the feature 
extraction process. 
"""

from pathlib import Path
from typing import Optional, Literal
import orjson
import pandas as pd
import torch
import json
from delphi.delphi.log.result_analysis import load_data
import numpy as np
from sparsify.sparsify import Sae
from tqdm import tqdm

def process_raw_explanations(path: Path, modules: list) -> pd.DataFrame:
    """
    Load explanations from a directory of files (one per latent), 
    save them into a pd.DataFrame, and delete the files.
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
                [{"latent_idx": latent_idx, 
                  "explanation": explanation,
                  "module": module}]
            ))
    
    if explanation_dfs:
        explanations_df = pd.concat(explanation_dfs, ignore_index=True)
        explanations_df.to_csv(path / "explanations.csv", index=False)
    else:
        assert False, 'processing seemed to be done before'
    # test whether the saved file is identical to original
    explanations_js = load_explanations(path)
    assert explanations_js.equals(explanations_df), "DataFrames are different"
    # remove all the files in that directory
    for file in path.glob(f"*{module}*"):
        file.unlink()
    return explanations_df

def load_explanations(path: Path) -> pd.DataFrame:
    """
    Load explanations from a CSV file.
    Args:
        path (Path): The path to the CSV file.
    Returns:
        pd.DataFrame: A DataFrame containing the explanations.
    """
    path = path / "explanations.csv"
    return pd.read_csv(path)

def process_raw_scores(path: Path, modules: list) -> pd.DataFrame:
    """
    Load raw scores from a directory of files (one per latent), 
    save them into a pd.DataFrame, and delete the files.
    Args:
        path (Path): The directory path where score files are stored.
    Returns:
        pd.DataFrame: A DataFrame containing the raw scores.
    """
    # Collect per-latent scores
    scores_df, counts = load_data(path, modules)
    if not scores_df.empty:
        # save scores_df to disk (preserve the dtypes for loading)
        dtypes = scores_df.dtypes.apply(lambda x: x.name).to_dict()
        with open(path / "scores_dtypes.json", "w") as f:
            json.dump(dtypes, f)
        
        scores_df.to_json(path / "scores.json", 
                        index=False,
                        orient="records",
                        double_precision=15)
    else:
        assert False, 'processing seemed to be done before'
    
    # test whether the saved file is identical to original
    scores_js = load_scores(path)
    assert scores_js.equals(scores_df), "DataFrames are different"
    # remove all the files in that directory
    del counts  # not needed anymore
    for score_type_dir in path.iterdir():
        for module in modules:
            for file in score_type_dir.glob(f"*{module}*"):
                file.unlink()
    return scores_df

def load_scores(path: Path) -> pd.DataFrame:
    """
    Load scores from a JSON file.
    Args:
        path (Path): The path to the JSON file.
    Returns:
        pd.DataFrame: A DataFrame containing the scores.
    """
    f_path = path / "scores.json"
    df = pd.read_json(f_path)
    # load a json dict dtypes
    with open(path / "scores_dtypes.json", "r") as f:
        dtypes = json.load(f)
    df = df.astype(dtypes)
    return df
def load_firing_counts(path: Path) -> pd.DataFrame:
    """
    Load firing counts from a JSON file.
    Args:
        path (Path): The path to the JSON file.
    Returns:
        pd.DataFrame: A DataFrame containing the firing counts.
    """
    f_path = path / "log" / "hookpoint_firing_counts.pt"
    counts = torch.load(f_path, weights_only=True)
    return counts 
#^ 

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

class TokenSimilarity:
    """
    Class to compute the similarity between the activating tokens of 
    the latents of a smaller and larger SAE.
    """

    def __init__(self, df1: pd.DataFrame, 
                 df2: pd.DataFrame,
                 sae_cfg1: dict,
                 sae_cfg2: dict,
                 threshold: float = 1.0,
                #  sorted_indices1: Optional[np.array] = None,
                #  sorted_indices2: Optional[np.array] = None,
                 common_cache: Optional[np.array] = None):
        """
        Initialize the TokenSimilarity with two SAEs.

        Args:
            df1 (pd.DataFrame): DataFrame containing the latent data
            of the activating examples from detection scoring of the smaller SAE.
            df2 (pd.DataFrame): DataFrame containing the latent data
            of the activating examples from detection scoring of the larger SAE.
            sae_cfg1 (dict): Configuration for the smaller SAE.
            sae_cfg2 (dict): Configuration for the larger SAE.
        """
        self.df1 = df1
        self.df2 = df2
        self.sae_cfg1 = sae_cfg1
        self.sae_cfg2 = sae_cfg2
        self.threshold = threshold
        self.sorted_indices1 = None
        self.sorted_indices2 = None
        self.sim_type: str = None
        if common_cache is not None:
            self.common_cache = common_cache
        else:
            width1 = self.sae_cfg1["expansion_factor"] * self.sae_cfg1["d_in"]
            width2 = self.sae_cfg2["expansion_factor"] * self.sae_cfg2["d_in"]
            self.common_cache: np.array = np.zeros((width1, width2), dtype=float)

    def find_activating_tokens(self, df: pd.DataFrame, sae_cfg: dict) -> np.ndarray:
        """
        Takes the latent data of the activating examples from detection scoring and 
        returns a padded array of activating tokens for each latent.
        
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
                tokens = np.array(raw.text)
                idx = acts > self.threshold
                _act_tokens.extend(tokens[idx].flatten().tolist())
            # Collect activation tokens for the current latent index
            act_tokens_list.append(np.array(_act_tokens))
        # Pad the activation tokens to the same length
        max_len = max(map(len, act_tokens_list))
        act_tokens = np.array([np.pad(arr, (0, max_len - len(arr)),
                                       constant_values="-1.0") 
                            for arr in act_tokens_list])
        
        return act_tokens
    #^
    def order_activating_tokens(self, act_tokens: np.array) -> np.array:
        """
        Order the activating tokens by their frequency in the latent.

        Args:
            act_tokens (np.array): Array of activating tokens for each latent.

        Returns:
            np.array: Ordered array of activating tokens.
        """
        token_counts = np.array([np.sum(arr == "-1.0") for arr in act_tokens])
        sorted_indices = np.argsort(token_counts)
        width1 = self.sae_cfg1["expansion_factor"] * self.sae_cfg1["d_in"]
        if act_tokens.shape[0] == width1:
            self.sorted_indices1 = sorted_indices
        else:
            self.sorted_indices2 = sorted_indices
        return act_tokens[sorted_indices]
    #^
    def ignore_padding(self, latents: np.array) -> np.array:
        return latents[latents != '-1.0']
    #^
    def common_tokens(self, lat1: np.array, lat2: np.array) -> set:
        """
        The common tokens between two latents.

        Args:
            lat1 (np.array): tokens of the latent from the smaller SAE.
            lat2 (np.array): tokens of the latent from the larger SAE.

        Returns:
            set: Common tokens between the two latents.
        """
        # ignore the "-1.0" values in the latents
        lat1 = self.ignore_padding(lat1)
        lat2 = self.ignore_padding(lat2)
        # convert to sets and find the intersection
        set1 = set(lat1)
        set2 = set(lat2)
        return set1.intersection(set2)
    #^
    def compute_similarity(self, 
                           sim_type: Literal['raw', 'jaccard', 'small', 'big'],
                           sorted: bool = False
                           ) -> np.array:
        """
        Compute the similarity between the tokens of the smaller and larger SAEs.

        Returns:
            latent_common (np.array): The similarity score between all latents in the 
            two SAEs (rows are latents of the smaller SAE).
        """
        # tokens_common = [] (no need to store all tokens)
        self.sim_type = sim_type
        act_tokens1 = self.find_activating_tokens(self.df1, self.sae_cfg1)
        act_tokens2 = self.find_activating_tokens(self.df2, self.sae_cfg2)
        if sorted:
            act_tokens1 = self.order_activating_tokens(act_tokens1)
            act_tokens2 = self.order_activating_tokens(act_tokens2)
        for i in range(len(act_tokens1)):
            # _token_common = []
            for j in range(len(act_tokens2)):
                # if dead latent, set common to NaN
                if np.all(act_tokens1[i] == '-1.0') or np.all(act_tokens2[j] == '-1.0'):
                    self.common_cache[i, j] = np.nan
                    continue
                else:
                    common = self.common_tokens(act_tokens1[i], act_tokens2[j])
                # _token_common.append(common)
                if self.sim_type == 'raw':
                    # Raw similarity
                    self.common_cache[i, j] = len(common)
                elif self.sim_type == 'jaccard':
                    # Jaccard similarity
                    lat1 = self.ignore_padding(act_tokens1[i])
                    lat2 = self.ignore_padding(act_tokens2[j])
                    if len(lat1) + len(lat2) - len(common) == 0:
                        self.common_cache[i, j] = 0
                    else:
                        self.common_cache[i, j] = len(common) / (len(lat1) + len(lat2) - len(common))
                elif sim_type == 'small':
                    # Small similarity
                    lat1 = self.ignore_padding(act_tokens1[i])
                    if len(lat1) == 0:
                        self.common_cache[i, j] = 0
                    else:
                        # Normalize by the smaller latent size
                        self.common_cache[i, j] = len(common) / len(lat1)
                elif sim_type == 'big':
                    # Big similarity
                    lat2 = self.ignore_padding(act_tokens2[j])
                    if len(lat2) == 0:
                        self.common_cache[i, j] = 0
                    else:
                        # Normalize by the smaller latent size
                        self.common_cache[i, j] = len(common) / len(lat2)
                else:
                    raise ValueError(
                        f"Invalid {sim_type}. Choose from 'raw', 'jaccard', 'small', or 'big'."
                        )
            #^
            # tokens_common.append(_token_common)
        #^
        return self.common_cache
    def save_common_cache(self, path: str) -> None:
        """
        Save the common cache to the path 
        """
        # the common cache is a numpy array
        np.save(path + f"_{self.sim_type}.npy", self.common_cache)
    def load_common_cache(self, path: str) -> dict[str, dict[int, list[int]]]:
        """
        Load the common cache from the path 
        """
        return np.load(path)
    #^

class DecoderSimilarity:
    """
    Class to compute the similarity between the latents of a smaller and larger SAE.
    """

    def __init__(self,
                 sae1: Optional[torch.nn.Module] = None,
                 sae2: Optional[torch.nn.Module] = None,
                 number_of_neighbors: int = 10,
                 neighbor_cache: Optional[dict[int, list[tuple[int, float]]]] = None):
        """
        Initialize a DecoderSimilarity.

        Args:
            sae1 (Optional[torch.Module]): The smaller SAE.
            sae2 (Optional[torch.Module]): The larger SAE.
            number_of_neighbors (int): Number of neighbors to consider for similarity.
            neighbor_cache (Optional[dict[int, list[tuple[int, float]]]]):
                Precomputed neighbor cache for the smaller SAE.
        """
        self.sae1 = sae1
        self.sae2 = sae2
        self.number_of_neighbors = number_of_neighbors
        if neighbor_cache is not None:
            self.neighbor_cache = neighbor_cache
        else:
            self.neighbor_cache: dict[int, list[tuple[int, float]]] = {}
        #^
    def _compute_d_similarity(self) -> dict[int, list[tuple[int, float]]]:
        """
        Compute the similarity between the latents of two SAEs based on their decoders.

        Returns:
            dict: the neighbor lists from larger SAE for each latent in sae1
        """
        assert isinstance(
                self.sae1, Sae
            ), "Autoencoder must be a sparsify.Sae for decoder similarity"
        decoder1 = self.sae1.W_dec.data.cuda()  # type:ignore
        weight_matrix_normalized1 = decoder1 / decoder1.norm(dim=1, keepdim=True)

        assert isinstance(
                self.sae2, Sae
            ), "Autoencoder must be a sparsify.Sae for decoder similarity"
        decoder2 = self.sae2.W_dec.data.cuda()  # type:ignore
        weight_matrix_normalized2 = decoder2 / decoder2.norm(dim=1, keepdim=True)
        # compute the similarity between the two weight matrices
        wT = weight_matrix_normalized2.T
        done = False
        batch_size = weight_matrix_normalized1.shape[0]
        number_latents = batch_size

        neighbor_lists = {}
        while not done:
            try:
                for start in tqdm(range(0, number_latents, batch_size)):
                    rows = wT[start : start + batch_size]
                    similarity_matrix = weight_matrix_normalized1 @ rows
                    # remove nan values
                    similarity_matrix = torch.nan_to_num(similarity_matrix, 0)
                    indices, values = torch.topk(
                        similarity_matrix, self.number_of_neighbors + 1, dim=1
                    )
                    neighbor_lists.update(
                        {
                            i
                            + start: list(
                                zip(indices[i].tolist()[1:], values[i].tolist()[1:])
                            )
                            for i in range(len(indices))
                        }
                    )
                    del similarity_matrix
                    torch.cuda.empty_cache()
                done = True
            except RuntimeError:  # Out of memory
                batch_size = batch_size // 2
                if batch_size < 2:
                    raise ValueError(
                        "Batch size is too small to compute similarity matrix. "
                        "You don't have enough memory."
                    )
        #^
        return neighbor_lists

    def save_neighbor_cache(self, path: str) -> None:
        """
        Save the neighbor cache to the path as a json file
        """
        with open(path + f"_decoder_sim.json", "w") as f:
            json.dump(self.neighbor_cache, f)
    #^
    def load_neighbor_cache(self, path: str) -> dict[str, dict[int, list[int]]]:
        """
        Load the neighbor cache from the path as a json file
        """
        with open(path + f"_decoder_sim.json", "r") as f:
            return json.load(f)
    #^