# Getting Started

## Setup

Clone the repository:
```
git clone https://github.com/avihu111/TypiClust
cd TypiClust
```

Create an environment using this command:
```
conda create --name typiclust-env.txt typiclust
```

or by running the following commands:
```
conda create --name typiclust python=3.7
conda activate typiclust
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install matplotlib scipy scikit-learn pandas
conda install -c conda-forge faiss-gpu
pip install pyyaml easydict termcolor tqdm simplejson yacs
```

## Representation Learning
Both TypiClust variants rely on representation learning. 
To train CIFAR-10 on simclr please run:
```
cd scan
python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_cifar10.yml
cd ..
```
When this finishes, the file ```./results/cifar-10/pretext/features_seed1.npy``` should exist.

## TypiClust - K-Means variant
To select samples according to TypiClust (K-Means) where the `initial_size=0` and the `budget=50` please run: 
```
cd deep-al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al typiclust_rp --exp-name typiclust_budget_50 --initial_size 0 --budget 50
cd ../../
```


## TypiClust - SCAN variant
We first must run SCAN clustering algorithm, as TypiClust uses its features cluster assignments.
Please repeat this command  `for k in [50, 100, 150, 200, 250, 300]`

```
cd scan
python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_cifar10.yml --num_clusters k
cd .. 
```

Then, you can run the experiment by
```
cd deep-al/tools
python train_al.py --cfg ../configs/cifar10/al/RESNET18.yaml --al typiclust_dc --exp-name typiclust_budget_50 --initial_size 0 --budget 50
cd ../../
```
