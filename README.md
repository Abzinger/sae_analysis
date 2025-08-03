# sae_analysis 
This toolbox provides an information-theoretic analysis (IT-analysis) of sparse-autoencoders (SAEs). The current aim is to explore whether information-theortic signatures for the size of the dictionary size can be found. We plan to extend this toobox for more information-theretic analysis of SAEs. 

## Toolbox Backbone 
The toolbox modified the SAE training package [sparsify](https://github.com/EleutherAI/sparsify) by [EleutherAI](https://github.com/EleutherAI) to serve as a backend for training SAEs. Additionally, it uses [delphi](https://github.com/EleutherAI/delphi) by [EleutherAI](https://github.com/EleutherAI) as a backend for caching activations for IT-analysis and generating explanations of SAEs latents and scoring them. 

The toolbox implements IT-analysis measures such as degree of redundancy and degree of vulnerability introduced in this [preprint](https://arxiv.org/abs/2504.15779?):

- **Shannon invariants: A scalable approach to information decomposition** by Aaron J. Gutknecht, Fernando E. Rosas, David A. Ehrlich, <u>Abdullah Makkeh</u>, Pedro A. M. Mediano, Michael Wibral
