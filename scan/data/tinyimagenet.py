"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import os
import pickle
from PIL import Image
from torchvision.datasets.utils import check_integrity
from torchvision import datasets
from typing import Any


def unpickle_object(path):
    with open(path, 'rb+') as file_pi:
        res = pickle.load(file_pi)
    return res


class TinyImageNet(datasets.VisionDataset):
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

    def __init__(self, root: str, split: str = 'train', transform=None, **kwargs: Any) -> None:
        self.root = root
        if split == 'train+unlabeled':
            split = 'train'
        self.split = datasets.utils.verify_str_arg(split, "split", ("train", "val"))

        if self.split == 'train':
            self.images, self.targets, self.cls_to_id = unpickle_object('../../../daphna/data/tiny_imagenet/tiny-imagenet-200/train.pkl')
        elif self.split == 'val':
            self.images, self.targets, self.cls_to_id = unpickle_object('../../../daphna/data/tiny_imagenet/tiny-imagenet-200/val.pkl')
        else:
            raise NotImplementedError('unknown split')
        self.targets = self.targets.astype(int)
        self.classes = list(self.cls_to_id.keys())
        super(TinyImageNet, self).__init__(root, **kwargs)
        self.transform = transform

    # Split folder is used for the 'super' call. Since val directory is not structured like the train,
    # we simply use train's structure to get all classes and other stuff
    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, 'train')

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        sample = Image.fromarray(self.images[index])
        target = int(self.targets[index])

        if self.transform is not None:
            sample = self.transform(sample)

        out = {'image': sample, 'target': target, 'meta': {'im_size': 64, 'index': index, 'class_name': target}}
        return out

    def __len__(self):
        return len(self.targets)
