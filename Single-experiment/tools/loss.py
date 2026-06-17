import torch.nn as nn
import torch
import torch.nn.functional as F

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def pdist_torch(emb1, emb2):
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx

def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

class TripletLoss_WRT(nn.Module):

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        return loss


class TripletLoss_WRT_local(nn.Module):

    def __init__(self):
        super(TripletLoss_WRT_local, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs1, inputs2, targets1, targets2, normalize_feature=False):
        if normalize_feature:
            inputs1 = normalize(inputs1, axis=-1)
            inputs2 = normalize(inputs2, axis=-1)
        dist_mat = pdist_torch(inputs1, inputs2)

        N = dist_mat.size(0)
        M = dist_mat.size(1)
        targets1 = targets1.unsqueeze(1)
        targets2 = targets2.unsqueeze(1)
        is_pos = targets1.expand(N, M).eq(targets2.expand(M, N).t()).float()
        is_neg = targets1.expand(N, M).ne(targets2.expand(M, N).t()).float()

        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        return loss


# compute the cosine distance between each pair of features
def cosine_matrix_compute(feat1,feat2,logit_scale):
    feat1_norm = feat1 / feat1.norm(dim=-1, keepdim=True)
    feat2_norm = feat2 / feat2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits r2t # text
    similarity_per_feat1 = logit_scale * feat1_norm @ feat2_norm.t()

    # cosine_matrix = F.softmax(similarity_per_feat1, dim=-1)

    return similarity_per_feat1

def kl_align_matrix(aligned_matrix, true_matrix):
    kl_loss = F.kl_div(aligned_matrix, true_matrix, reduction='mean')
    return kl_loss


def kl_align_loss(ir_feat,fusion_feat,text_feat,logit_scale,mode='I2T'):
    ir2fusion_matrix = cosine_matrix_compute(ir_feat,fusion_feat,logit_scale)
    text2fusion_matrix = cosine_matrix_compute(text_feat,fusion_feat,logit_scale)
    if mode == 'I2T':
        return CoRefineLoss(ir2fusion_matrix,text2fusion_matrix.detach())
    elif mode == 'T2I':
        return CoRefineLoss(text2fusion_matrix,ir2fusion_matrix.detach())
    else:
        raise ValueError("mode should be 'I2T' or 'T2I'")

def CoRefineLoss(output1, output2):

    # Target is ignored at training time. Loss is defined as KL divergence
    # between the model output and the refined labels.
    if output2.requires_grad:
        raise ValueError("Refined labels should not require gradients.")

    output1_log_prob = F.log_softmax(output1, dim=1)
    output2_prob = F.softmax(output2, dim=1)

    _, pred_label = output2_prob.max(1)

    # Loss is normal cross entropy loss
    # base_loss = F.cross_entropy(output1, pred_label)

    # Loss is -dot(model_output_log_prob, refined_labels). Prepare tensors
    # for batch matrix multiplicatio

    model_output1_log_prob = output1_log_prob.unsqueeze(2)
    model_output2_prob = output2_prob.unsqueeze(1)

    # Compute the loss, and average/sum for the batch.
    kl_loss = -torch.bmm(model_output2_prob, model_output1_log_prob)

    return kl_loss.mean()



def L_i2t(img_feat,text_feat,logit_scale):
    assert img_feat.size(0) == text_feat.size(0) # batch size should be the same
    identity_mask = torch.eye(img_feat.size(0)).to(img_feat.device)
    return -(torch.log_softmax(cosine_matrix_compute(img_feat,text_feat,logit_scale),dim=-1) * identity_mask).mean()

def L_t2i(img_feat,text_feat,logit_scale,labels):
    assert img_feat.size(0) == text_feat.size(0) # batch size should be the same
    mask = torch.eq(labels.unsqueeze(1),labels.unsqueeze(0)).float()
    return -(torch.log_softmax(cosine_matrix_compute(text_feat,img_feat,logit_scale),dim=-1) * mask).mean()/sum(mask)


def supervised_clip_contrastive_loss(
    rgb_feat,
    ir_feat,
    rgb_labels,
    ir_labels,
    logit_scale,
    eps=1e-6,
):
    """CLIP-style symmetric contrastive loss with identity-aware positives."""
    rgb_norm = F.normalize(rgb_feat, dim=-1)
    ir_norm = F.normalize(ir_feat, dim=-1)

    logits_rgb_to_ir = logit_scale * rgb_norm @ ir_norm.t()
    logits_ir_to_rgb = logits_rgb_to_ir.t()

    pos_mask = torch.eq(rgb_labels.unsqueeze(1), ir_labels.unsqueeze(0)).float()
    pos_rgb = pos_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
    pos_ir = pos_mask.sum(dim=0, keepdim=True).t().clamp(min=1.0)

    loss_rgb_to_ir = -((F.log_softmax(logits_rgb_to_ir, dim=1) * pos_mask).sum(dim=1, keepdim=True) / (pos_rgb + eps)).mean()
    loss_ir_to_rgb = -((F.log_softmax(logits_ir_to_rgb, dim=1) * pos_mask.t()).sum(dim=1, keepdim=True) / (pos_ir + eps)).mean()

    return 0.5 * (loss_rgb_to_ir + loss_ir_to_rgb)


def pair_clip_contrastive_loss(
    rgb_feat,
    ir_feat,
    logit_scale,
):
    """CLIP-style symmetric loss for one-to-one RGB/IR pairs in a batch."""
    if rgb_feat.size(0) != ir_feat.size(0):
        raise ValueError(
            f"Pair CLIP requires equal batch sizes, got {rgb_feat.size(0)} and {ir_feat.size(0)}"
        )

    rgb_norm = F.normalize(rgb_feat, dim=-1)
    ir_norm = F.normalize(ir_feat, dim=-1)
    logits_rgb_to_ir = logit_scale * rgb_norm @ ir_norm.t()
    labels = torch.arange(rgb_feat.size(0), device=rgb_feat.device)
    loss_rgb_to_ir = F.cross_entropy(logits_rgb_to_ir, labels)
    loss_ir_to_rgb = F.cross_entropy(logits_rgb_to_ir.t(), labels)
    return 0.5 * (loss_rgb_to_ir + loss_ir_to_rgb)






