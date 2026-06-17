import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=0.3, memory_size=0):
        super().__init__()
        self.margin = margin
        self.memory_size = int(memory_size)
        self.register_buffer("memory_embeddings", None, persistent=False)
        self.register_buffer("memory_labels", None, persistent=False)
        self.memory_ptr = 0
        self.memory_count = 0

    def reset_memory(self):
        self.memory_embeddings = None
        self.memory_labels = None
        self.memory_ptr = 0
        self.memory_count = 0

    @torch.no_grad()
    def _enqueue(self, embeddings, labels):
        if self.memory_size <= 0:
            return
        embeddings = embeddings.detach().clone()
        labels = labels.detach().clone()
        if self.memory_embeddings is None:
            feat_dim = embeddings.shape[1]
            self.memory_embeddings = embeddings.new_zeros((self.memory_size, feat_dim))
            self.memory_labels = labels.new_full((self.memory_size,), -1)

        batch = embeddings.shape[0]
        for idx in range(batch):
            self.memory_embeddings[self.memory_ptr] = embeddings[idx]
            self.memory_labels[self.memory_ptr] = labels[idx]
            self.memory_ptr = (self.memory_ptr + 1) % self.memory_size
            self.memory_count = min(self.memory_count + 1, self.memory_size)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, dim=1)
        distances = torch.cdist(embeddings, embeddings, p=2)
        labels = labels.view(-1)
        label_mat = labels.unsqueeze(1)
        positive_mask = label_mat.eq(label_mat.t())
        negative_mask = ~positive_mask
        eye = torch.eye(positive_mask.shape[0], device=positive_mask.device, dtype=torch.bool)
        positive_mask = positive_mask & ~eye

        hardest_positive = (distances * positive_mask.float()).max(dim=1)[0]
        negative_distances = distances.masked_fill(~negative_mask, float("inf"))
        hardest_negative = negative_distances.min(dim=1)[0]

        if self.memory_size > 0 and self.memory_count > 0 and self.memory_embeddings is not None:
            memory_embeddings = self.memory_embeddings[: self.memory_count].detach().clone()
            memory_labels = self.memory_labels[: self.memory_count].detach().clone()
            dist_memory = torch.cdist(embeddings, memory_embeddings, p=2)

            pos_mask_mem = labels.unsqueeze(1).eq(memory_labels.unsqueeze(0))
            neg_mask_mem = ~pos_mask_mem

            mem_pos_dist = dist_memory.masked_fill(~pos_mask_mem, float("-inf"))
            mem_neg_dist = dist_memory.masked_fill(~neg_mask_mem, float("inf"))

            has_mem_pos = pos_mask_mem.any(dim=1)
            has_mem_neg = neg_mask_mem.any(dim=1)
            hardest_positive_mem = mem_pos_dist.max(dim=1)[0]
            hardest_negative_mem = mem_neg_dist.min(dim=1)[0]

            hardest_positive = torch.where(has_mem_pos, torch.maximum(hardest_positive, hardest_positive_mem), hardest_positive)
            hardest_negative = torch.where(has_mem_neg, torch.minimum(hardest_negative, hardest_negative_mem), hardest_negative)

        valid = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not valid.any():
            self._enqueue(embeddings, labels)
            return embeddings.new_tensor(0.0)

        loss = F.relu(hardest_positive[valid] - hardest_negative[valid] + self.margin)
        self._enqueue(embeddings, labels)
        return loss.mean()
