# Getting Started

## Environment Setup

Clone the repository:

```
git clone https://github.com/acl21/deep-active-learning-pytorch
```

Install dependencies:

```
pip install -r requirements.txt
```

## Understanding the Config File
```
# Folder name where best model logs etc are saved. Setting EXP_NAME: "auto" creates a timestamp named folder
EXP_NAME: 'YOUR_EXPERIMENT_NAME'
# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries
RNG_SEED: 1
# GPU ID you want to execute the process on (this feature isn't working as of now, use the commands shown in this file below instead)
GPU_ID: '3'
DATASET:
  NAME: CIFAR10 # or CIFAR100, MNIST, SVHN, TINYIMAGENET, IMBALANCED_CIFAR10/100
  ROOT_DIR: 'data' # Relative path where data should be downloaded
  # Specifies the proportion of data in train set that should be considered as the validation data
  VAL_RATIO: 0.1
  # Data augmentation methods - 'simclr', 'randaug', 'hflip'
  AUG_METHOD: 'hflip' 
MODEL:
  # Model type. 
  # Choose from vgg style ['vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19_bn', 'vgg19',]
  # or from resnet style ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 
  # 'wide_resnet50_2', 'wide_resnet101_2']
  # or `alexnet`
  TYPE: resnet18
  NUM_CLASSES: 10
OPTIM:
  TYPE: 'sgd' # or 'adam'
  BASE_LR: 0.025
  # Learning rate policy select from {'cos', 'exp', 'steps' or 'none'}
  LR_POLICY: cos
  # Steps for 'steps' policy (in epochs)
  STEPS: [0] #[0, 30, 60, 90]
  # Training Epochs
  MAX_EPOCH: 1
  # Momentum
  MOMENTUM: 0.9
  # Nesterov Momentum
  NESTEROV: False
  # L2 regularization
  WEIGHT_DECAY: 0.0005
  # Exponential decay factor
  GAMMA: 0.1
TRAIN:
  SPLIT: train
  # Training mini-batch size
  BATCH_SIZE: 96
  # Image size
  IM_SIZE: 32
  IM_CHANNELS = 3
  # Evaluate model on test data every eval period epochs
  EVAL_PERIOD: 2
TEST:
  SPLIT: test
  # Testing mini-batch size
  BATCH_SIZE: 200
  # Image size
  IM_SIZE: 32
  # Saved model to use for testing (useful when running `tools/test_model.py`)
  MODEL_PATH: ''
DATA_LOADER:
  NUM_WORKERS: 4
CUDNN:
  BENCHMARK: True
ACTIVE_LEARNING:
  # Active sampling budget (at each episode)
  BUDGET_SIZE: 5000
  # Active sampling method
  SAMPLING_FN: 'dbal' # 'random', 'uncertainty', 'entropy', 'margin', 'bald', 'vaal', 'coreset', 'ensemble_var_R'
  # Initial labeled pool ratio (% of total train set that should be labeled before AL begins)
  INIT_L_RATIO: 0.1
  # Max AL episodes
  MAX_ITER: 5
  DROPOUT_ITERATIONS: 25 # Used by DBAL
# Useful when running `ensemble_al.py` or `ensemble_train.py`
ENSEMBLE: 
  NUM_MODELS: 3
  MODEL_TYPE: ['resnet18']
```

Please refer to `pycls/core/config.py` to configure your experiments at a deeper level. 


## Execution Commands
### Active Learning
Once the config file is configured appropriately, perform DBAL active learning with the following command inside the `tools` directory. 

```
CUDA_VISIBLE_DEVICES=0 python train_al.py \
    --cfg=../configs/cifar10/al/RESNET18.yaml --al=dbal --exp-name=YOUR_EXPERIMENT_NAME
```

### Ensemble Active Learning 

Watch out for the ensemble options in the config file. This setting by default using _Ensemble Variation-Ratio_ as the query method. 

```
CUDA_VISIBLE_DEVICES=0 python ensemble_al.py \
    --cfg=../configs/cifar10/al/RESNET18.yaml --exp-name=YOUR_EXPERIMENT_NAME
```

### Passive Learning

```
CUDA_VISIBLE_DEVICES=0 python train.py \
    --cfg=../configs/cifar10/train/RESNET18.yaml --exp-name=YOUR_EXPERIMENT_NAME
```

[comment]: <> (### Ensemble Passive Learning)

[comment]: <> (Watch out for the ensemble options in the config file.)

[comment]: <> (```)

[comment]: <> (CUDA_VISIBLE_DEVICES=0 python ensemble_train.py \)

[comment]: <> (    --cfg=../configs/cifar10/train/RESNET18_ENS.yaml --exp-name=YOUR_EXPERIMENT_NAME)

[comment]: <> (```)

### Specific Model Evaluation

This is useful if you want to evaluate a particular saved model. Pass the path to the model in the yaml file. Refer to the file inside the `config/evaluate` directory for clarity. 

```
CUDA_VISIBLE_DEVICES=0 python test_model.py \
    --cfg configs/cifar10/evaluate/RESNET18.yaml
```


[comment]: <> (## Add Your Own Dataset )

[comment]: <> (To add your own dataset, you need to do the following: )

[comment]: <> (1. Write the PyTorch Dataset code for your custom dataset &#40;or you could directly use the ones [PyTorch provides]&#40;https://pytorch.org/vision/stable/datasets.html&#41;&#41;. )

[comment]: <> (2. Create a sub class of the above Dataset with some desirable modifications and add it to the `pycls/datasets/custom_datasets.py`.)

[comment]: <> (    * We add two new variables to the dataset - a boolean flag `no_aug` and `test_transform`. )

[comment]: <> (    * We set the flag `no_aug = True` before iterating through unlabeled and the validations dataloaders so that data doesn't get augmented. )

[comment]: <> (    * See how we modify the `__get_item__` function to achieve that:)

[comment]: <> (```)

[comment]: <> (class CIFAR10&#40;torchvision.datasets.CIFAR10&#41;:)

[comment]: <> (      def __init__&#40;self, root, train, transform, test_transform, download=True&#41;:)

[comment]: <> (          super&#40;CIFAR10, self&#41;.__init__&#40;root, train, transform=transform, download=download&#41;)

[comment]: <> (          self.test_transform = test_transform)

[comment]: <> (          self.no_aug = False)
  
[comment]: <> (      def __getitem__&#40;self, index: int&#41;:)

[comment]: <> (          """)

[comment]: <> (          Args:)

[comment]: <> (              index &#40;int&#41;: Index)
  
[comment]: <> (          Returns:)

[comment]: <> (              tuple: &#40;image, target&#41; where target is index of the target class.)

[comment]: <> (          """)

[comment]: <> (          img, target = self.data[index], self.targets[index])
  
[comment]: <> (          # doing this so that it is consistent with all other datasets)

[comment]: <> (          # to return a PIL Image)

[comment]: <> (          img = Image.fromarray&#40;img&#41;)
          
[comment]: <> (          ##########################)

[comment]: <> (          # set True before iterating through unlabeled or validation set)

[comment]: <> (          if self.no_aug: )

[comment]: <> (              if self.test_transform is not None:)

[comment]: <> (                  img = self.test_transform&#40;img&#41;            )

[comment]: <> (          else:)

[comment]: <> (              if self.transform is not None:)

[comment]: <> (                  img = self.transform&#40;img&#41;)

[comment]: <> (          #########################)
          
[comment]: <> (          return img, target)

[comment]: <> (```)

[comment]: <> (3. Add your dataset in `pycls/dataset/data.py` )

[comment]: <> (    * Add appropriate preprocessing steps to `getPreprocessOps` )

[comment]: <> (    * Add the dataset call to `getDataset`)

[comment]: <> (4. Create appropriate config `yaml` files and use them for training AL.)


## Some Comments About Our Toolkit
* Our toolkit currently only supports 'SGD' (with learning rate scheduler)  and 'Adam' (no scheduler). 
* We log everything. Our toolkit saves the indices of the initial labeled pool, samples queried each episode, episode wise best model, visual plots for "Iteration vs Loss", "Epoch vs Val Accuracy", "Episode vs Test Accuracy" and more. Please check an experiment's logs at `output/CIFAR10/resnet18/ENT_1/` for clarity.
* We added dropout (p=0.5) to all our models just before the final fully connected layer. We do this to allow the DBAL and BALD query methods to work.
* We also provide an iPython notebook that aggregates results directly from the experiment folders. You can find it at `output/results_aggregator.ipynb`. 
* If you add your own dataset, please make sure you to create the custom version as explained in point 2 in the instructions. Failing to do that would mean that your unlabeled data (big red flag for AL) and validation data will have been augmentated. This is because we use a single dataset instance and subset and index based dataloaders.   
* We tested the toolkit only on a Linux machine with Python 3.8.
* Please create an issue with appropriate details:
  * if you are unable to get the toolkit to work or run into any problems
  * if we have not provided credits correctly to the rightful owner (please attach proof)
  * if you notice any flaws in the implementation
