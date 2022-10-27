import numpy as np
import pandas as pd
import faiss
from sklearn.cluster import MiniBatchKMeans, KMeans
import pycls.datasets.utils as ds_utils

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


def get_mean_nn_dist(features, num_neighbors, return_indices=False):
    distances, indices = get_nn(features, num_neighbors)
    mean_distance = distances.mean(axis=1)
    if return_indices:
        return mean_distance, indices
    return mean_distance


def calculate_typicality(features, num_neighbors):
    mean_distance = get_mean_nn_dist(features, num_neighbors)
    # low distance to NN is high density
    typicality = 1 / (mean_distance + 1e-5)
    return typicality


def kmeans(features, num_clusters):
    if num_clusters <= 50:
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
        num_clusters = min(len(self.lSet) + self.budgetSize, self.MAX_NUM_CLUSTERS)
        print(f'Clustering into {num_clusters} clustering. Scan clustering: {is_scan}')
        if is_scan:
            fname_dict = {'CIFAR10': f'../../scan/results/cifar-10/scan/features_seed{self.seed}_clusters{num_clusters}.npy',
                          'CIFAR100': f'../../scan/results/cifar-100/scan/features_seed{self.seed}_clusters{num_clusters}.npy',
                          'TINYIMAGENET': f'../../scan/results/tiny-imagenet/scan/features_seed{self.seed}_clusters{num_clusters}.npy',
                          }
            fname = fname_dict[self.ds_name]
            self.features = np.load(fname)
            self.clusters = np.load(fname.replace('features', 'probs')).argmax(axis=-1)
        else:
            self.features = ds_utils.load_features(self.ds_name, self.seed)
            self.clusters = kmeans(self.features, num_clusters=num_clusters)
        print(f'Finished clustering into {num_clusters} clusters.')

    def select_samples(self):
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

        print(f'Finished the selection of {len(activeSet)} samples.')
        print(f'Active set is {activeSet}')
        return activeSet, remainSet
