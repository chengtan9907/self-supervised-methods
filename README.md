# self-supervised-methods

The implementations about some recent self-supervised methods.

# Introduction

Algorithms:

- [x] [SimCLR]
- [ ] [MoCo]
- [x] [Byol] 
- [x] [SimSiam]

Datasets:

- [x] [CIFAR-10]

# Dependency

1. numpy
2. torch, torchvision
3. scipy
4. addict
5. json
6. tqdm

# Reference

[1] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A simple framework for contrastive learning of visual representations. arXiv preprint arXiv:2002.05709.

[2] Grill, J. B., Strub, F., Altché, F., Tallec, C., Richemond, P., Buchatskaya, E., ... & Piot, B. (2020). Bootstrap your own latent-a new approach to self-supervised learning. Advances in Neural Information Processing Systems, 33.

[3] Chen, X., & He, K. (2020). Exploring Simple Siamese Representation Learning. arXiv preprint arXiv:2011.10566.

[4] Github: https://github.com/leftthomas/SimCLR/tree/master

[5] Github: https://github.com/lucidrains/byol-pytorch/tree/master