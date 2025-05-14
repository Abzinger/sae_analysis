import torch
from transformers import AutoTokenizer
from delphi.delphi.config import RunConfig, ConstructorConfig, SamplerConfig
from delphi.delphi.latents import LatentDataset
from pathlib import Path
from typing import Optional
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
    rlz[:, :train_config["sae"]["k"]] = _tok
    rlz[:, train_config["sae"]["k"]:] = _act
    
    print("saving the rlz tensor...")
    t_name = "rlz_" + name + ".pt"
    torch.save(rlz, save_dir / t_name)

    return rlz
#^

def create_marginal_rlz(joint_rlz: torch.Tensor, m_rlz_inverse: torch.Tensor,
                        neu_id: int, complementary: bool) -> torch.Tensor:
    """
    Create a marginal realization tensor from the realization tensor.
    The marginal realization is computed by masking the activation values 
    of all neurons but the neu_id-th neuron.
    Args:
        joint_rlz (torch.Tensor): The realization tensor of shape (n_tokens, 2*top_k).
        m_rlz_inverse (torch.Tensor): neurons indices in the joint realization tensor.
        neu_id (int): The dimension along which to compute the marginal (Note: due
            to -1 padding it is ith-neuron + 1).
        complementary (bool): If True, compute the complement marginal realization
            (i.e. all neurons but neu_id-th neuron).
    Returns:
        marginal_rlz (torch.Tensor): The marginal realization of shape 
        (n_tokens, 2*top_k).
    """
    rlz = torch.empty((joint_rlz.shape[0], joint_rlz.shape[1]))
    k = joint_rlz.shape[1]//2
    # compute a mask of when neu_id-th neuron is active
    mask_neu = (m_rlz_inverse == neu_id)
    if complementary:
        # set neu_id index to -1 and keep all other indices
        rlz_idx = torch.mul(joint_rlz[:, :k], ~mask_neu)
        mask_sub = torch.mul(mask_neu, -1)
        rlz_idx = rlz_idx + mask_sub
        # set activations of neu_id to -1 and keep all other activations
        rlz_act = torch.mul(joint_rlz[:, k:], ~mask_neu)
        rlz_act = rlz_act + mask_sub
        # reorder the tensor to have the neu_id-th neuron at the end
        srt_idx = torch.argsort(rlz_idx, dim=1)
        # the correct way to reorder the tensor is to use torch.take_along_dim
        torch.take_along_dim(rlz_idx, srt_idx, axis=1)
        torch.take_along_dim(rlz_act, srt_idx, axis=1)
        rlz[:, :k] = rlz_idx
        rlz[:, k:] = rlz_act
        return rlz
    #^    
    else:
        # Set the activation values of all neurons but neu_id to zeros 
        rlz_act = joint_rlz[:, k:]
        neu_inverse = torch.mul(rlz_act, mask_neu)
        # return a one dimensional tensor of the activation values 
        # (a neurons fires at most once per token)
        return torch.sum(neu_inverse, dim=1)
    #^
#^

def create_pmf(rlz: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
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
#^ 

def compute_r(joint_rlz: torch.Tensor, width: int) -> int:
    """
    Compute the degree of redundnacy of the joint pmf.
    The r value is computed by normalizing the sum of 
    the marginal entropies of individual neurons by the 
    joint pmf (sum_i H(X_i)/H(X_1,...,X_n)).
    Args:
        rlz (torch.Tensor): The realization tensor of shape (n_tokens, top_k).
        width (int): The number of neurons of the sae.
    Returns:
        r (int): The degree of redundnacy r.
    """
    # compute the joint pmf
    jnt_pmf = create_pmf(joint_rlz, dim=0)
    # compute the joint entropy 
    H_jnt = -(torch.log2(jnt_pmf)).mean()
    # compute the marginal entropy per neuron 
    s_H_mrg = 0 
    # compute a map of neuron idices in realization 
    k = joint_rlz.shape[1]//2
    _, m_rlz_inverse = torch.unique(joint_rlz[:, :k], sorted=True, return_inverse=True)
    # range shifted by 1 due to -1 padding
    for i in range(1, width+1):
        # compute the marginal rlz per neuron
        print(f"create marginal realization for neuron {i-1}...")
        m_rlz = create_marginal_rlz(joint_rlz, m_rlz_inverse, i, complement=False)
        # compute the marginal pmf per neuron
        m_pmf = create_pmf(m_rlz)
        # compute the marginal entropy per neuron
        s_H_mrg += -(torch.log2(m_pmf)).mean()
    #^ 
    # compute the degree of redundancy r 
    r = s_H_mrg / H_jnt
    return r
#^


def compute_v(joint_rlz: torch.Tensor, width: int) -> int:
    """
    Compute the degree of vulnerability of the joint pmf.
    The v value is computed by normalizing the sum of 
    the conditional marginal entropies of individual 
    neurons conditioned on the rest by the joint entropy 
    (sum_i H(X_i|X_1,...,X_n)/H(X_1,...,X_n)).
    Args:
        rlz (torch.Tensor): The realization tensor of shape (n_tokens, top_k).
        width (int): The number of neurons of the sae.
    Returns:
        r (int): The degree of redundnacy r.
    """
    # compute the joint pmf
    jnt_pmf = create_pmf(joint_rlz, dim=0)
    # compute the joint entropy 
    H_jnt = -(torch.log2(jnt_pmf)).mean()
    # compute the marginal entropy per neuron 
    s_H_c_mrg = 0 
    # compute a map of neuron idices in realization 
    k = joint_rlz.shape[1]//2
    _, m_rlz_inverse = torch.unique(joint_rlz[:, :k], sorted=True, return_inverse=True)
    # range shifted by 1 due to -1 padding
    for i in range(1, width+1):
        # compute the marginal rlz per neuron
        print(f"create marginal realization for neuron {i-1}...")
        c_m_rlz = create_marginal_rlz(joint_rlz, m_rlz_inverse, i, complement=True)
        # compute the marginal pmf per neuron
        c_m_pmf = create_pmf(c_m_rlz, dim=0)
        # compute the conditional marginal entropy per neuron 
        # H(X_1|X_2,X_3) = H(X_1,X_2,X_3) - H(X_2,X_3)
        s_H_c_mrg += H_jnt + (torch.log2(c_m_pmf)).mean()
    #^ 
    # compute the degree of redundancy v
    v = s_H_c_mrg / H_jnt
    return v
#^