import torch
import torchvision
from PIL import Image

class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train, transform, test_transform, download=True):
        super(CIFAR10, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)            
        else:
            if self.transform is not None:
                img = self.transform(img)


        return img, target


class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, train, transform, test_transform, download=True):
        super(CIFAR100, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)            
        else:
            if self.transform is not None:
                img = self.transform(img)


        return img, target


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, root, train, transform, test_transform, download=True):
        super(MNIST, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        
        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)            
        else:
            if self.transform is not None:
                img = self.transform(img)


        return img, target


class SVHN(torchvision.datasets.SVHN):
    def __init__(self, root, train, transform, test_transform, download=True):
        super(SVHN, self).__init__(root, train, transform=transform, download=download)
        self.test_transform = test_transform
        self.no_aug = False

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        if self.no_aug:
            if self.test_transform is not None:
                img = self.test_transform(img)            
        else:
            if self.transform is not None:
                img = self.transform(img)


        return img, target