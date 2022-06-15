import os
import numpy as np
from PIL import Image

import torch
import torchvision.datasets as datasets

from typing import Any


class TinyImageNet(datasets.ImageFolder):
    """`Tiny ImageNet Classification Dataset.

    Args:
        root (string): Root directory of the ImageNet Dataset.
        split (string, optional): The dataset split, supports ``train``, or ``val``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class name tuples.
        class_to_idx (dict): Dict with items (class_name, class_index).
        wnids (list): List of the WordNet IDs.
        wnid_to_idx (dict): Dict with items (wordnet_id, class_index).
        samples (list): List of (image path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    def __init__(self, root: str, split: str = 'train', transform=None, test_transform=None, only_features=False, **kwargs: Any) -> None:
        self.root = root
        self.test_transform = test_transform
        self.no_aug = False

        assert self.check_root(), "Something is wrong with the Tiny ImageNet dataset path. Download the official dataset zip from http://cs231n.stanford.edu/tiny-imagenet-200.zip and unzip it inside {}.".format(self.root)
        self.split = datasets.utils.verify_str_arg(split, "split", ("train", "val"))
        wnid_to_classes = self.load_wnid_to_classes()

        super(TinyImageNet, self).__init__(self.split_folder, **kwargs)
        self.transform = transform
        self.wnids = self.classes
        self.wnid_to_idx = self.class_to_idx
        self.classes = [wnid_to_classes[wnid] for wnid in self.wnids]
        self.class_to_idx = {cls: idx
                             for idx, clss in enumerate(self.classes)
                             for cls in clss}
        # Tiny ImageNet val directory structure is not similar to that of train's
        # So a custom loading function is necessary
        self.only_features = only_features
        if self.split == 'train':
            self.features = np.load('../../scan/results/tiny-imagenet/pretext/features_seed1.npy')
        elif self.split == 'val':
            self.features = np.load('../../scan/results/tiny-imagenet/pretext/test_features_seed1.npy')
            self.root = root
            self.imgs, self.targets = self.load_val_data()
            self.samples = [(self.imgs[idx], self.targets[idx]) for idx in range(len(self.imgs))]
            self.root = os.path.join(self.root, 'val')



    # Split folder is used for the 'super' call. Since val directory is not structured like the train, 
    # we simply use train's structure to get all classes and other stuff
    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, 'train')


    def load_val_data(self):
        imgs, targets = [], []
        with open(os.path.join(self.root, 'val', 'val_annotations.txt'), 'r') as file:
            for line in file:
                if line.split()[1] in self.wnids:
                    img_file, wnid = line.split('\t')[:2]
                    imgs.append(os.path.join(self.root, 'val', 'images', img_file))
                    targets.append(wnid)
        targets = np.array([self.wnid_to_idx[wnid] for wnid in targets])
        return imgs, targets


    def load_wnid_to_classes(self):
        wnid_to_classes = {}
        with open(os.path.join(self.root, 'words.txt'), 'r') as file:
            lines = file.readlines()
            lines = [x.split('\t') for x in lines]
            wnid_to_classes = {x[0]:x[1].strip() for x in lines}
        return wnid_to_classes

    def check_root(self):
        tinyim_set = ['words.txt', 'wnids.txt', 'train', 'val', 'test']
        for x in os.scandir(self.root):
            if x.name not in tinyim_set:
                return False
        return True

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.only_features:
            sample = self.features[index]
        else:
            if self.no_aug:
                if self.test_transform is not None:
                    sample = self.test_transform(sample)
            else:
                if self.transform is not None:
                    sample = self.transform(sample)

        return sample, target