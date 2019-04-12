## Introduction
This is the implementation of Lei's [beta-Dropout: A Unified Dropout] paper in PyTorch.
L. Liu, Y. Luo, X. Shen, M. Sun and B. Li, "$\beta$ -Dropout: A Unified Dropout," in IEEE Access, vol. 7, pp. 36140-36153, 2019.
doi: 10.1109/ACCESS.2019.2904881

In this paper, we attempt to change the selection problem to a parameter tuning problem by proposing a general form of dropout, β-dropout, to unify the discrete dropout with continuous dropout.
We show that by adjusting the shape parameter β, the β-dropout can yield the Bernoulli dropout, uniform dropout, and approximate Gaussian dropout. Furthermore, it can obtain continuous regularization strength, which paves the way for self-adaptive dropout regularization.

## Requirement
* python 2
* pytorch > 0.4.0
* torchvision = 0.2.1
* numpy
* argparse

## Folder 
* data               MINIST dataset
* dropout_methods    Source code for Bernoulli, Gaussian, uniform and beta dropout methods
* examples           Source code used to reproduce the experimental results in the paper with Hinton's network(784-800-800-10) on MINIST
* trained_models     Provide trained model and inference code for easy comparison with adaptive beta dropout

## Usage
```
cd examples
```
Then,
```
sh mlp_beta_dropout.sh
```
or

```
python mlp_beta_dropout.py

You can use mlp_beta_dropout_approach_bernoulli.sh to reproduce the result of the Bernoulli dropout.
You can use mlp_beta_dropout_approach_uniform.sh to reproduce the result of the uniform dropout.

## Contact

Please feel free to leave suggestions or comments to Lei Liu (liulei13@mail.ustc.edu.cn)

# Reference
If you find this code useful for your research, please cite
```
@ARTICLE{8666975,
author={L. {Liu} and Y. {Luo} and X. {Shen} and M. {Sun} and B. {Li}},
journal={IEEE Access},
title={ $\beta$ -Dropout: A Unified Dropout},
year={2019},
volume={7},
number={},
pages={36140-36153},
keywords={Shape;Task analysis;Neurons;Gaussian distribution;Tuning;Deep learning;Training;Regularization;dropout;deep learning;Gaussian dropout;Bernoulli dropout},
doi={10.1109/ACCESS.2019.2904881},
ISSN={2169-3536},
month={},}
```
