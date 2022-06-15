# This file is directly taken from code implementation by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564

# ----------------------------------------------------------
#This file was modified from implementation of [Auto-Augment](https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py) to add Random Augmentations.

""" AutoAugment ==[Modified to]==> RandAugment"""


from PIL import Image, ImageEnhance, ImageOps
import PIL.ImageDraw as ImageDraw
import numpy as np
import random

class RandAugmentPolicy(object):
    """ Randomly choose one of the best 25 Sub-policies on CIFAR10.
        Example:
        >>> policy = RandAugmentPolicy()
        >>> transformed = policy(image)
        Example as a PyTorch Transform:
        >>> transform=transforms.Compose([
        >>>     transforms.Resize(256),
        >>>     RandAugmentPolicy(),
        >>>     transforms.ToTensor()])
    """
    #I change fill color from (128, 128, 128) to (0, 0, 0)
    def __init__(self, fillcolor=(0,0,0), N=1, M=5):
        self.policies = ["invert","autocontrast","equalize","rotate","solarize","color", \
            "posterize","contrast","brightness","sharpness","shearX","shearY","translateX", \
            "translateY","cutout"]
        self.N = N
        self.M = M

    def __call__(self, img):
        choosen_policies = np.random.choice(self.policies, self.N)
        for policy in choosen_policies:
            subpolicy_obj = SubPolicy(operation=policy, magnitude=self.M)
            img = subpolicy_obj(img)

        return img

    def __repr__(self):
        return "RandAugment CIFAR10 Policy with Cutout"

class SubPolicy(object):
    def __init__(self, operation, magnitude, fillcolor=(128, 128, 128), MAX_PARAM=10):
        ranges = {
            "shearX": np.linspace(0, 0.3, MAX_PARAM),
            "shearY": np.linspace(0, 0.3, MAX_PARAM),
            "translateX": np.linspace(0, 150 / 331, MAX_PARAM),
            "translateY": np.linspace(0, 150 / 331, MAX_PARAM),
            "rotate": np.linspace(0, 30, MAX_PARAM),
            "color": np.linspace(0.0, 0.9, MAX_PARAM),
            "posterize": np.round(np.linspace(8, 4, MAX_PARAM), 0).astype(np.int),
            "solarize": np.linspace(256, 0, MAX_PARAM),
            "contrast": np.linspace(0.0, 0.9, MAX_PARAM),
            "sharpness": np.linspace(0.0, 0.9, MAX_PARAM),
            "brightness": np.linspace(0.0, 0.9, MAX_PARAM),
            "autocontrast": [0] * MAX_PARAM,
            "equalize": [0] * MAX_PARAM,
            "invert": [0] * MAX_PARAM,
            "cutout":  np.linspace(0.0,0.8, MAX_PARAM),
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            #return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)
            return Image.composite(rot, Image.new("RGBA", rot.size, (0,) * 4), rot).convert(img.mode)
        
        def Cutout(img, v):  # [0, 60] => percentage: [0, 0.2]
            assert 0.0 <= v <= 0.8
            if v <= 0.:
                return img

            v = v * img.size[0]

            return CutoutAbs(img, v)


        def CutoutAbs(img, v):  # [0, 60] => percentage: [0, 0.2]
            # assert 0 <= v <= 20
            if v < 0:
                return img
            w, h = img.size
            x0 = np.random.uniform(w)
            y0 = np.random.uniform(h)

            x0 = int(max(0, x0 - v / 2.))
            y0 = int(max(0, y0 - v / 2.))
            x1 = min(w, x0 + v)
            y1 = min(h, y0 + v)

            xy = (x0, y0, x1, y1)
            #color = (125, 123, 114)
            color = (0, 0, 0)
            img = img.copy()
            ImageDraw.Draw(img).rectangle(xy, color)
            return img

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img),
            "cutout": lambda img, magnitude: Cutout(img, magnitude)
        }

        self.operation = func[operation]
        self.magnitude = ranges[operation][magnitude]


    def __call__(self, img):
        img = self.operation(img, self.magnitude)
        return img

