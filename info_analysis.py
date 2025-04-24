import torch
from transformers import AutoTokenizer
from delphi.delphi.config import RunConfig, ConstructorConfig, SamplerConfig
from delphi.delphi.latents import LatentDataset
from pathlib import Path


def create_rlz(run_cfg: RunConfig, 
               constructor_cfg: ConstructorConfig, 
               sampler_cfg: SamplerConfig, 
               width: int,
               save_dir: Path):
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
        width (int): The width of the SAE or transcoder.
        save_dir (Path): The directory where the raw tokens and activations
            are saved.
    Returns:
        tkn_acts (torch.sparse_coo_tensor): The sparse tensor of the 
        realization (tokens, activations) of the SAE or transcoder.
    """
    
    print("loading Tokinzer")
    tokenizer = AutoTokenizer.from_pretrained(run_cfg.model)
    
    print("reading raw tokens activation as latent dataset")
    latent_dataset = LatentDataset(
        raw_dir=save_dir,
        sampler_cfg=sampler_cfg,
        constructor_cfg=constructor_cfg,
        tokenizer=tokenizer,
        modules=["transformer.h.1.mlp.act"]
        )
    print("get coordinates&values for all latent dataset")
    for i in range(len(latent_dataset.buffers)):
        if i == 0:
            locations, activations, tokens = latent_dataset.buffers[i].load()
            tokens_flat = tokens.flatten()
            t_coo = locations[:,0]*tokens.shape[1] + locations[:,1]
            coo = torch.stack((t_coo, locations[:,2]), dim=0)
            values = torch.stack((tokens_flat[t_coo], activations), dim=1)
        else:
            _locations, _activations, _ = latent_dataset.buffers[i].load()
            _t_coo = _locations[:,0]*tokens.shape[1] + _locations[:,1]
            _coo = torch.stack((_t_coo, _locations[:,2]), dim=0)
            _values = torch.stack((tokens_flat[_t_coo], _activations), dim=1)
            coo = torch.cat((coo, _coo), dim=1)
            values = torch.cat((values, _values), dim=0)
        #^
    #^ 
    # create a sparse tensor (currently tokens are included in the values)
    tkn_acts = torch.sparse_coo_tensor(coo, values,
                                       (tokens_flat.shape[0], width, 2))
    return tkn_acts
#^
