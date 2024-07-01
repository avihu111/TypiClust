# DCoM - Getting Started

## Background
The files in this zip are taken from the AL repository cited in the paper.
In this document, we provide you the steps for running our new algorithm - DCoM.

## Setup

Clone the repository, create a virtual environment using Python 3.9, and install the packages from `deep-al/requirements.txt`.

## Representation Learning
DCoM relies on representation learning. 
To train CIFAR-10 on SimCLR, please run:

```
cd scan
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10.yml
cd ..
```
When this finishes, the file `./results/cifar-10/pretext/features_seed1.npy` should exist.
You can use other representations and change the path in the file `deep-al/pycls/datasets/utils/features.py`.

## DCoM

Now, you can run the active learning experiment by executing the following script:

Example DCoM script:

```
cd deep-al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al dcom --exp-name auto --initial_size 0 --budget 10 --initial_delta 0.75
cd ../../
```

You can add the `a_logistic` and `k_logistic` parameters to the run using `--a_logistic` and `--k_logistic`.
