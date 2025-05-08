import torch
from transformers import AutoTokenizer
from delphi.delphi.config import RunConfig, ConstructorConfig, SamplerConfig
from delphi.delphi.latents import LatentDataset
from pathlib import Path
import json


def create_joint_rlz(run_cfg: RunConfig, 
                     constructor_cfg: ConstructorConfig, 
                     sampler_cfg: SamplerConfig,
                     name: str,
                     modules: list[str],
                     raw_dir: Path,
                     save_dir: Path) -> torch.Tensor:
    """
    Create a sparse tensor of the realization (tokens, activations) 
    of the SAE or transcoder.
    The tensor is of shape (n_tokens, width, 2) where the first
    dimension is the token index, the second dimension is the
    activation index, and the third dimension is the value of
    the token and the activation.
    The first value is the token id and the second value is the
    activation value.
    Args:
        run_cfg (RunConfig): The run configuration.
        constructor_cfg (ConstructorConfig): The constructor configuration.
        sampler_cfg (SamplerConfig): The sampler configuration.
        name (str): The name of the sae or transcoder to use.
        modules (list[str]): The list of module (of transformer) to include.
        raw_dir (Path): The root directory where the raw tokens and activations
            are saved.
        save_dir (Path): The root directory where the realization will be saved.
    Returns:
        rlz (torch.tensor): The sparse tensor of the 
        realization (tokens, activations) of the SAE or transcoder.
    """
    
    print("loading Tokinzer...")
    tokenizer = AutoTokenizer.from_pretrained(run_cfg.model)
    
    print("reading raw tokens activation as latent datase...")
    latent_dataset = LatentDataset(
        raw_dir=raw_dir / name,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        tokenizer=tokenizer,
        modules=modules
        )
    # create an empty realization of shape (n_tokens, sae width) 
    with open(run_cfg.sparse_model + "config.json", "r") as f:
        train_config = json.load(f)
    #^
    n_tokens = latent_dataset.tokens.shape[0]*latent_dataset.tokens.shape[1]
    rlz = torch.empty((n_tokens, 2*train_config["sae"]["k"]))

    print("getting coordinates&values for all latent dataset...")
    for i in range(len(latent_dataset.buffers)):
        if i == 0:
            locations, activations, tokens = latent_dataset.buffers[i].load()
            tokens_flat = tokens.flatten() # might get rid of this
            t_coo = locations[:,0]*tokens.shape[1] + locations[:,1]
            id_coo = torch.stack((t_coo, locations[:,2]), dim=0)
            act_coo = torch.stack((t_coo, activations), dim=0)
        else:
            _locations, _activations, _ = latent_dataset.buffers[i].load()
            _t_coo = _locations[:,0]*tokens.shape[1] + _locations[:,1]
            _id_coo = torch.stack((_t_coo, _locations[:,2]), dim=0)
            _act_coo = torch.stack((_t_coo, _activations), dim=0)
            # _coo_check = _coo[:,:5]
            # coo_shape = coo.shape 
            # coo_check = coo[:,coo_shape[1]-5:coo_shape[1]] 
            id_coo = torch.cat((id_coo, _id_coo), dim=1)
            act_coo = torch.cat((act_coo, _act_coo), dim=1)
            # c_coo_shape = coo.shape
            # c_coo_check = coo[:,coo_shape[1]:coo_shape[1]+5]
            # act_shape = activations.shape
            # activations = torch.cat((activations, _activations), dim=0)
            # c_act_shape = activations.shape
        #^
    #^ 
    print("reorder the tokens coordinates...")
    sort_idx = torch.argsort(id_coo[0,:])
    id_coo = id_coo[:, sort_idx]
    act_coo = act_coo[:, sort_idx]
    
    print("getting unique tokens coordinates...")
    _, counts = torch.unique(id_coo[0,:], return_counts=True)
    
    print("filling the rlz tensor...")
    # comment: first neurons idices and then activations 
    lis_counts = counts.tolist()
    print("splitting tokens...")
    _tok = torch.split(id_coo[1,:], lis_counts, dim=0)
    # make sure that the maximum length of the tokens is 32
    _tok +=(-1*torch.ones((train_config["sae"]["k"],), dtype=id_coo[1,:].dtype),)
    print("padding tokens...")
    _tok = torch.nn.utils.rnn.pad_sequence(_tok, batch_first=True, padding_value=-1)
    _tok = _tok[:-1]
    print("splitting activations...")
    _act = torch.split(act_coo[1,:], lis_counts, dim=0)
    # make sure that the maximum length of the activations is 32
    _act +=(-1*torch.ones((train_config["sae"]["k"],), dtype=act_coo[1,:].dtype),)
    print("padding activations...")
    _act = torch.nn.utils.rnn.pad_sequence(_act, batch_first=True, padding_value=-1)
    _act = _act[:-1]
    rlz[:, :32] = _tok
    rlz[:, 32:] = _act
    
    print("saving the rlz tensor...")
    t_name = "rlz_" + name + ".pt"
    torch.save(rlz, save_dir / t_name)

    return rlz
#^

def create_pmf(rlz: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Create a probability mass function (PMF) from the realization tensor.
    The PMF is computed by normalizing the counts of unique values
    in the realization tensor.
    Args:
        rlz (torch.Tensor): The realization tensor of shape (n_tokens, width).
        dim (int): The dimension along which to compute the PMF.
    Returns:
        pmf (torch.Tensor): The PMF of shape (n_tokens, width).
    """
    # compute the unique values and their counts
    _,inverse,counts= torch.unique(rlz, dim=dim, return_inverse=True, return_counts=True)
    return counts[inverse]/(rlz.shape[0])