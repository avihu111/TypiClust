import numpy as np
import pandas as pd
import faiss


def calculate_typicality(features, num_neighbors):
    mean_distance = get_mean_nn_dist(features, num_neighbors)
    # low distance to NN is high density
    density = 1 / (mean_distance + 1e-5)
    return density


def get_mean_nn_dist(features, num_neighbors, return_indices=False):
    distances, indices = get_nn(features, num_neighbors)
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance


def get_nn(features, num_neighbors):
    # calculates nearest neighbors on GPU
    d = features.shape[1]
    features = features.astype(np.float32)
    cpu_index = faiss.IndexFlatL2(d)
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    gpu_index.add(features)  # add vectors to the index
    distances, indices = gpu_index.search(features, num_neighbors + 1)
    # 0 index is the same sample, dropping it
    return distances[:, 1:], indices[:, 1:]


def kmeans(features, num_clusters):
    from sklearn.cluster import MiniBatchKMeans, KMeans
    if num_clusters <= 50:
        print('using regular kmeans without minibatch')
        km = KMeans(n_clusters=num_clusters)
        km.fit_predict(features)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, batch_size=5000)
        km.fit_predict(features)
    return km.labels_


class TypiClust:
    MIN_CLUSTER_SIZE = 5
    MAX_NUM_CLUSTERS = 500
    K_NN = 20

    def __init__(self, cfg, lSet, uSet, budgetSize, is_scan=False):
        self.cfg = cfg
        self.ds_name = self.cfg['DATASET']['NAME']
        self.seed = self.cfg['RNG_SEED']
        self.features = None
        self.clusters = None
        self.lSet = lSet
        self.uSet = uSet
        self.budgetSize = budgetSize
        self.init_features_and_clusters(is_scan)

    def init_features_and_clusters(self, is_scan):
        if is_scan:
            num_clusters = min(len(self.lSet) + self.budgetSize, self.MAX_NUM_CLUSTERS)
            fname_dict = {'CIFAR10': f'../../Unsupervised-Classification/results/cifar-10/scan/features_seed{self.seed}_clusters{num_clusters}.npy',
                          'CIFAR100': f'../../Unsupervised-Classification/results/cifar-20/scan/features_seed{self.seed}_clusters{num_clusters}.npy',
                          'TINYIMAGENET': f'../../Unsupervised-Classification/results/tiny-imagenet/scan/features_seed{self.seed}_clusters{num_clusters}.npy',
                          }
            fname = fname_dict[self.ds_name]
            self.features = np.load(fname)
            self.clusters = np.load(fname.replace('features', 'probs')).argmax(axis=-1)
        else:
            fname_dict = {'CIFAR10': f'../../Unsupervised-Classification/results/cifar-10/pretext/features_seed{self.seed}.npy',
                          'CIFAR100': f'../../Unsupervised-Classification/results/cifar-20/pretext/features_seed{self.seed}.npy',
                          'TINYIMAGENET': f'../../Unsupervised-Classification/results/tiny-imagenet/pretext/features_seed{self.seed}.npy',
                          'IMAGENET50': '../../../dino/runs/trainfeat.pth',
                          'IMAGENET100': '../../../dino/runs/trainfeat.pth',
                          'IMAGENET200': '../../../dino/runs/trainfeat.pth',
                          }
            fname = fname_dict[self.ds_name]
            self.features = np.load(fname)
            self.clusters = kmeans(self.features, num_clusters=len(self.lSet) + self.budgetSize)

    def select_samples(self, ):
        # using only labeled+unlabeled indices, without validation set.
        relevant_indices = np.concatenate([self.lSet, self.uSet]).astype(int)
        features = self.features[relevant_indices]
        labels = np.copy(self.clusters[relevant_indices])
        existing_indices = np.arange(len(self.lSet))
        # counting cluster sizes and number of labeled samples per cluster
        cluster_ids, cluster_sizes = np.unique(labels, return_counts=True)
        cluster_labeled_counts = np.bincount(labels[existing_indices], minlength=len(cluster_ids))
        clusters_df = pd.DataFrame({'cluster_id': cluster_ids, 'cluster_size': cluster_sizes, 'existing_count': cluster_labeled_counts,
                                    'neg_cluster_size': -1 * cluster_sizes})
        # drop too small clusters
        clusters_df = clusters_df[clusters_df.cluster_size > self.MIN_CLUSTER_SIZE]
        # sort clusters by lowest number of existing samples, and then by cluster sizes (large to small)
        clusters_df = clusters_df.sort_values(['existing_count', 'neg_cluster_size'])
        labels[existing_indices] = -1

        selected = []

        for i in range(self.budgetSize):
            cluster = clusters_df.iloc[i % len(clusters_df)].cluster_id
            indices = (labels == cluster).nonzero()[0]
            rel_feats = features[indices]
            # in case we have too small cluster, calculate density among half of the cluster
            typicality = calculate_typicality(rel_feats, min(self.K_NN, len(indices) // 2))
            idx = indices[typicality.argmax()]
            selected.append(idx)
            labels[idx] = -1

        selected = np.array(selected)
        assert len(selected) == self.budgetSize, 'added a different number of samples'
        assert len(np.intersect1d(selected, existing_indices)) == 0, 'should be new samples'
        activeSet = relevant_indices[selected]
        remainSet = np.array(sorted(list(set(self.uSet) - set(activeSet))))
        return activeSet, remainSet
