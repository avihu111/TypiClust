# This file is directly taken from a code implementation shared with me by Prateek Munjal et al., authors of the paper https://arxiv.org/abs/2002.09564
# GitHub: https://github.com/PrateekMunjal
# ----------------------------------------------------------

from torch.utils.data.sampler import Sampler


class IndexedSequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_idxes (Dataset indexes): dataset indexes to sample from
    """

    def __init__(self, data_idxes, isDebug=False):
        if isDebug: print("========= my custom squential sampler =========")
        self.data_idxes = data_idxes

    def __iter__(self):
        return (self.data_idxes[i] for i in range(len(self.data_idxes)))

    def __len__(self):
        return len(self.data_idxes)

# class IndexedDistributedSampler(Sampler):
#     """Sampler that restricts data loading to a particular index set of dataset.

#     It is especially useful in conjunction with
#     :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
#     process can pass a DistributedSampler instance as a DataLoader sampler,
#     and load a subset of the original dataset that is exclusive to it.

#     .. note::
#         Dataset is assumed to be of constant size.

#     Arguments:
#         dataset: Dataset used for sampling.
#         num_replicas (optional): Number of processes participating in
#             distributed training.
#         rank (optional): Rank of the current process within num_replicas.
#     """

#     def __init__(self, dataset, index_set, num_replicas=None, rank=None, allowRepeat=True):
#         if num_replicas is None:
#             if not dist.is_available():
#                 raise RuntimeError("Requires distributed package to be available")
#             num_replicas = dist.get_world_size()
#         if rank is None:
#             if not dist.is_available():
#                 raise RuntimeError("Requires distributed package to be available")
#             rank = dist.get_rank()
#         self.dataset = dataset
#         self.num_replicas = num_replicas
#         self.rank = rank
#         self.epoch = 0
#         self.index_set = index_set
#         self.allowRepeat = allowRepeat
#         if self.allowRepeat:
#             self.num_samples = int(math.ceil(len(self.index_set) * 1.0 / self.num_replicas))
#             self.total_size = self.num_samples * self.num_replicas
#         else:
#             self.num_samples = int(math.ceil((len(self.index_set)-self.rank) * 1.0 / self.num_replicas))
#             self.total_size = len(self.index_set)

#     def __iter__(self):
#         # deterministically shuffle based on epoch
#         g = torch.Generator()
#         g.manual_seed(self.epoch)
#         indices = torch.randperm(len(self.index_set), generator=g).tolist()
#         #To access valid indices
#         #indices = self.index_set[indices]
#         # add extra samples to make it evenly divisible
#         if self.allowRepeat:
#             indices += indices[:(self.total_size - len(indices))]
        
#         assert len(indices) == self.total_size

#         # subsample
#         indices = self.index_set[indices[self.rank:self.total_size:self.num_replicas]]
#         assert len(indices) == self.num_samples, "len(indices): {} and self.num_samples: {}"\
#             .format(len(indices), self.num_samples)

#         return iter(indices)

#     def __len__(self):
#         return self.num_samples

#     def set_epoch(self, epoch):
#         self.epoch = epoch
