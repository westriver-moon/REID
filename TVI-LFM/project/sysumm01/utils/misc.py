import csv
import os
import random
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(param.numel() for param in model.parameters())


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0
        self.sum = 0.0
        self.avg = 0.0

    def update(self, value, n=1):
        self.sum += float(value) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def save_checkpoint(state, path):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def append_metrics_row(path, fieldnames, row):
    ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def strip_prefix_if_present(state_dict, prefix):
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            stripped[key[len(prefix):]] = value
        else:
            stripped[key] = value
    return stripped
