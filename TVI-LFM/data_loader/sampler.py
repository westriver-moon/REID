import numpy as np
from torch.utils.data.sampler import Sampler


def _sample_identity_instances(positions, sample_count):
    replace = len(positions) < sample_count
    return np.random.choice(positions, sample_count, replace=replace)


def GenIdx(train_color_label, train_thermal_label):
    color_pos = []
    unique_label_color = np.unique(train_color_label)
    for i in range(len(unique_label_color)):
        tmp_pos = [k for k, v in enumerate(train_color_label) if v == unique_label_color[i]]
        color_pos.append(tmp_pos)

    thermal_pos = []
    unique_label_thermal = np.unique(train_thermal_label)
    for i in range(len(unique_label_thermal)):
        tmp_pos = [k for k, v in enumerate(train_thermal_label) if v == unique_label_thermal[i]]
        thermal_pos.append(tmp_pos)

    return color_pos, thermal_pos

class IdentitySampler(Sampler):
    """Sample person identities evenly in each batch.
        Args:
            train_color_label, train_thermal_label: labels of two modalities
            color_pos, thermal_pos: positions of each identity
            batchSize: batch size
    """

    def __init__(self, train_color_label, train_thermal_label, color_pos, thermal_pos, num_pos, batchSize):
        uni_label = np.unique(train_color_label)
        self.n_classes = len(uni_label)

        N = np.maximum(len(train_color_label), len(train_thermal_label))
        for j in range(int(N / (batchSize * num_pos)) + 1):
            batch_idx = np.random.choice(uni_label, batchSize, replace=False)
            for i in range(batchSize):
                sample_color = _sample_identity_instances(color_pos[batch_idx[i]], num_pos)
                sample_thermal = _sample_identity_instances(thermal_pos[batch_idx[i]], num_pos)

                if j == 0 and i == 0:
                    index1 = sample_color
                    index2 = sample_thermal
                else:
                    index1 = np.hstack((index1, sample_color))
                    index2 = np.hstack((index2, sample_thermal))

        self.index1 = index1
        self.index2 = index2
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N

class PairIdentitySampler(Sampler):
    """Sample paired visible/thermal instances by identity and within-ID order.

    The dataset receives the generated color and thermal indices at the same
    dataloader index, so cIndex[i] and tIndex[i] form the positive pair.
    """

    def __init__(self, train_color_label, train_thermal_label, num_pos, batchSize):
        self.num_pos = num_pos
        uni_label = sorted(set(train_color_label).intersection(set(train_thermal_label)))
        if len(uni_label) == 0:
            raise ValueError("No shared identities found for pair sampling")

        pairs_by_label = {}
        for label in uni_label:
            color_pos = np.where(np.asarray(train_color_label) == label)[0]
            thermal_pos = np.where(np.asarray(train_thermal_label) == label)[0]
            pair_count = min(len(color_pos), len(thermal_pos))
            if pair_count == 0:
                continue
            pairs_by_label[label] = list(zip(color_pos[:pair_count], thermal_pos[:pair_count]))

        self.uni_label = np.asarray(sorted(pairs_by_label.keys()))
        self.n_classes = len(self.uni_label)
        if self.n_classes == 0:
            raise ValueError("No valid visible/thermal pairs found")

        N = np.maximum(len(train_color_label), len(train_thermal_label))
        loops = int(N / (batchSize * num_pos)) + 1
        index1 = []
        index2 = []
        for _ in range(loops):
            batch_idx = np.random.choice(self.uni_label, batchSize, replace=False)
            for label in batch_idx:
                pairs = pairs_by_label[label]
                replace = len(pairs) < num_pos
                chosen = np.random.choice(len(pairs), num_pos, replace=replace)
                for pair_idx in chosen:
                    color_idx, thermal_idx = pairs[int(pair_idx)]
                    index1.append(color_idx)
                    index2.append(thermal_idx)

        self.index1 = np.asarray(index1)
        self.index2 = np.asarray(index2)
        self.N = N

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return self.N
