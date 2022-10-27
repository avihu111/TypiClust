import numpy as np
DATASET_FEATURES_DICT = {
    'train':
        {
            'CIFAR10':'../../scan/results/cifar-10/pretext/features_seed{seed}.npy',
            'CIFAR100':'../../scan/results/cifar-100/pretext/features_seed{seed}.npy',
            'TINYIMAGENET': '../../scan/results/tiny-imagenet/pretext/features_seed{seed}.npy',
            'IMAGENET50': '../../dino/runs/trainfeat.pth',
            'IMAGENET100': '../../dino/runs/trainfeat.pth',
            'IMAGENET200': '../../dino/runs/trainfeat.pth',
        },
    'test':
        {
            'CIFAR10': '../../scan/results/cifar-10/pretext/test_features_seed{seed}.npy',
            'CIFAR100': '../../scan/results/cifar-100/pretext/test_features_seed{seed}.npy',
            'TINYIMAGENET': '../../scan/results/tiny-imagenet/pretext/test_features_seed{seed}.npy',
            'IMAGENET50': '../../dino/runs/testfeat.pth',
            'IMAGENET100': '../../dino/runs/testfeat.pth',
            'IMAGENET200': '../../dino/runs/testfeat.pth',
        }
}

def load_features(ds_name, seed=1, train=True, normalized=True):
    " load pretrained features for a dataset "
    split = "train" if train else "test"
    fname = DATASET_FEATURES_DICT[split][ds_name].format(seed=seed)
    if fname.endswith('.npy'):
        features = np.load(fname)
    elif fname.endswith('.pth'):
        features = torch.load(fname)
    else:
        raise Exception("Unsupported filetype")
    if normalized:
        features = features / np.linalg.norm(features, axis=1, keepdims=True)
    return features