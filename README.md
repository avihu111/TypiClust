# Active Learning on a Budget - Opposite Strategies Suit High and Low Budgets



# Deep Active Learning Toolkit for Image Classification in PyTorch

This is a code base for deep active learning for image classification written in [PyTorch](https://pytorch.org/). It is build on top of FAIR's [pycls](https://github.com/facebookresearch/pycls/). I want to emphasize that this is a derivative of the toolkit originally shared with me via email by Prateek Munjal _et al._, the authors of the paper _"Towards Robust and Reproducible Active Learning using Neural Networks"_, paper available [here](https://arxiv.org/abs/2002.09564).  

## Introduction

The goal of this repository is to provide a simple and flexible codebase for deep active learning. It is designed to support rapid implementation and evaluation of research ideas. We also provide a results on CIFAR10 below.

The codebase currently only supports single-machine single-gpu training. We will soon scale it to single-machine multi-gpu training, powered by the PyTorch distributed package.

## Using the Toolkit

Please see [`GETTING_STARTED`](docs/GETTING_STARTED.md) for brief instructions on installation, adding new datasets, basic usage examples, etc.

## Active Learning Methods Supported
* Uncertainty Sampling
  * Least Confidence
  * Min-Margin
  * Max-Entropy
  * Deep Bayesian Active Learning (DBAL) [1]
  * Bayesian Active Learning by Disagreement (BALD) [1]
* Diversity Sampling 
  * Coreset (greedy) [2]
  * Variational Adversarial Active Learning (VAAL) [3]
* Query-by-Committee Sampling
  * Ensemble Variation Ratio (Ens-varR) [4]


## Datasets Supported
* [CIFAR10/100](https://www.cs.toronto.edu/~kriz/cifar.html)
* [MNIST](http://yann.lecun.com/exdb/mnist/)
* [SVHN](http://ufldl.stanford.edu/housenumbers/)
* [Tiny ImageNet](https://www.kaggle.com/c/tiny-imagenet) (Download the zip file [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip))
* Long-Tail CIFAR-10/100

Follow the instructions in [`GETTING_STARTED`](docs/GETTING_STARTED.md) to add a new dataset. 

## Results on CIFAR10 and CIFAR100 

The following are the results on CIFAR10 and CIFAR100, trained with hyperameters present in `configs/cifar10/al/RESNET18.yaml` and `configs/cifar100/al/RESNET18.yaml` respectively. All results were averaged over 3 runs. 

## Citing this Repository

If you find this repo helpful in your research, please consider citing us and the owners of the original toolkit:

```
@article{Chandra2021DeepAL,
    Author = {Akshay L Chandra and Vineeth N Balasubramanian},
    Title = {Deep Active Learning Toolkit for Image Classification in PyTorch},
    Journal = {https://github.com/acl21/deep-active-learning-pytorch},
    Year = {2021}
}

@article{Munjal2020TowardsRA,
  title={Towards Robust and Reproducible Active Learning Using Neural Networks},
  author={Prateek Munjal and N. Hayat and Munawar Hayat and J. Sourati and S. Khan},
  journal={ArXiv},
  year={2020},
  volume={abs/2002.09564}
}
```

## License

This toolkit is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
