from __future__ import annotations

import numpy as np
import torch
from torch.utils.data.sampler import Sampler


def build_label_positions(labels) -> list[list[int]]:
    labels = np.asarray(labels)
    positions = []
    for label in np.unique(labels):
        positions.append(np.where(labels == label)[0].tolist())
    return positions


class PMTIdentitySampler(Sampler):
    """Official PMT PK sampler for two aligned modalities."""

    def __init__(self, color_labels, thermal_labels, color_pos, thermal_pos, batch_size: int, num_pos: int) -> None:
        self.color_labels = np.asarray(color_labels)
        self.thermal_labels = np.asarray(thermal_labels)
        self.batch_size = int(batch_size)
        self.num_pos = int(num_pos)
        assert self.batch_size % self.num_pos == 0, "batch_size_per_modality must be divisible by num_pos"
        self.ids_per_batch = self.batch_size // self.num_pos
        self.unique_labels = np.unique(self.color_labels)
        thermal_unique = np.unique(self.thermal_labels)
        assert np.array_equal(self.unique_labels, thermal_unique), "visible and IR label sets differ"
        self.color_pos = color_pos
        self.thermal_pos = thermal_pos
        self.index1, self.index2 = self._generate_indices()
        self.N = max(len(self.color_labels), len(self.thermal_labels))

    def _choice(self, pool, size: int):
        pool = np.asarray(pool)
        replace = len(pool) < size
        return np.random.choice(pool, size, replace=replace)

    def _generate_indices(self):
        color_chunks = []
        thermal_chunks = []
        n_batches = max(len(self.color_labels), len(self.thermal_labels)) // self.batch_size + 1
        for _ in range(n_batches):
            batch_labels = np.random.choice(self.unique_labels, self.ids_per_batch, replace=False)
            color_batch = np.empty(self.batch_size, dtype=np.int64)
            thermal_batch = np.empty(self.batch_size, dtype=np.int64)
            for slot, label in enumerate(batch_labels):
                start = slot * self.num_pos
                label_index = int(np.where(self.unique_labels == label)[0][0])
                color_batch[start : start + self.num_pos] = self._choice(self.color_pos[label_index], self.num_pos)
                thermal_batch[start : start + self.num_pos] = self._choice(self.thermal_pos[label_index], self.num_pos)
            color_chunks.append(color_batch)
            thermal_chunks.append(thermal_batch)
        return np.hstack(color_chunks), np.hstack(thermal_chunks)

    def __iter__(self):
        return iter(np.arange(len(self.index1)))

    def __len__(self):
        return len(self.index1)


def assert_pmt_batch_layout(label_visible, label_ir, num_pos: int, batch_size: int) -> None:
    assert label_visible.shape[0] == batch_size, f"visible batch must be {batch_size}"
    assert label_ir.shape[0] == batch_size, f"IR batch must be {batch_size}"
    assert torch.equal(label_visible, label_ir), "visible and IR labels must be aligned"
    for labels in (label_visible, label_ir):
        chunks = labels.view(batch_size // num_pos, num_pos)
        assert torch.all(chunks.eq(chunks[:, :1])), "each consecutive num_pos samples must share an identity"
        assert torch.unique(chunks[:, 0]).numel() == batch_size // num_pos, "wrong identities per batch"

