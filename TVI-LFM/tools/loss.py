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
    dist_mtx = dist_mtx.addmm_(emb1, emb2.t(), beta=1, alpha=-2)
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


def hard_example_mining(dist_mat, target):
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    num = dist_mat.size(0)
    is_pos = target.expand(num, num).eq(target.expand(num, num).t())
    is_neg = target.expand(num, num).ne(target.expand(num, num).t())
    dist_ap, _ = torch.max(dist_mat[is_pos].contiguous().view(num, -1), 1, keepdim=True)
    dist_an, _ = torch.min(dist_mat[is_neg].contiguous().view(num, -1), 1, keepdim=True)
    return dist_ap.squeeze(1), dist_an.squeeze(1)


class PMTTripletLoss(nn.Module):
    def __init__(self, margin=0.1, feat_norm="no"):
        super(PMTTripletLoss, self).__init__()
        self.margin = margin
        self.feat_norm = feat_norm
        if margin >= 0:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, global_feat1, global_feat2, target):
        if self.feat_norm == "yes":
            global_feat1 = F.normalize(global_feat1, p=2, dim=-1)
            global_feat2 = F.normalize(global_feat2, p=2, dim=-1)
        dist_mat = pdist_torch(global_feat1, global_feat2)
        dist_ap, dist_an = hard_example_mining(dist_mat, target)
        y = dist_an.new_ones(dist_an.size())
        if self.margin >= 0:
            return self.ranking_loss(dist_an, dist_ap, y)
        return self.ranking_loss(dist_an - dist_ap, y)


class PMTMSEL(nn.Module):
    def __init__(self, num_pos, feat_norm="no"):
        super(PMTMSEL, self).__init__()
        self.num_pos = int(num_pos)
        self.feat_norm = feat_norm

    def forward(self, inputs, targets):
        if self.feat_norm == "yes":
            inputs = F.normalize(inputs, p=2, dim=-1)

        assert inputs.size(0) % 2 == 0, "MSEL expects [visible, ir] concatenation"
        target, target_ir = targets.chunk(2, 0)
        assert torch.equal(target, target_ir), "MSEL requires aligned visible and IR labels"
        num = target.size(0)
        dist_mat = pdist_torch(inputs, inputs)

        dist_intra_rgb = dist_mat[0:num, 0:num]
        dist_cross_rgb = dist_mat[0:num, num : 2 * num]
        dist_intra_ir = dist_mat[num : 2 * num, num : 2 * num]
        dist_cross_ir = dist_mat[num : 2 * num, 0:num]

        is_pos = target.expand(num, num).eq(target.expand(num, num).t())

        intra_rgb, _ = (is_pos * dist_intra_rgb).topk(self.num_pos - 1, dim=1, largest=True, sorted=False)
        intra_mean_rgb = torch.mean(intra_rgb, dim=1)

        intra_ir, _ = (is_pos * dist_intra_ir).topk(self.num_pos - 1, dim=1, largest=True, sorted=False)
        intra_mean_ir = torch.mean(intra_ir, dim=1)

        dist_cross_rgb = dist_cross_rgb[is_pos].contiguous().view(num, -1)
        cross_mean_rgb = torch.mean(dist_cross_rgb, dim=1)

        dist_cross_ir = dist_cross_ir[is_pos].contiguous().view(num, -1)
        cross_mean_ir = torch.mean(dist_cross_ir, dim=1)

        loss = (
            torch.mean(torch.pow(cross_mean_rgb - intra_mean_rgb, 2))
            + torch.mean(torch.pow(cross_mean_ir - intra_mean_ir, 2))
        ) / 2
        return loss


class PMTDCL(nn.Module):
    def __init__(self, num_pos=4, feat_norm="no"):
        super(PMTDCL, self).__init__()
        self.num_pos = int(num_pos)
        self.feat_norm = feat_norm

    def forward(self, inputs, targets):
        if self.feat_norm == "yes":
            inputs = F.normalize(inputs, p=2, dim=-1)

        total = inputs.size(0)
        assert total % (2 * self.num_pos) == 0, "DCL expects [visible, ir] with fixed num_pos chunks"
        left, right = targets.chunk(2, 0)
        assert torch.equal(left, right), "DCL requires aligned visible and IR labels"
        id_num = total // 2 // self.num_pos

        is_neg = targets.expand(total, total).ne(targets.expand(total, total).t())
        is_neg_c2i = is_neg[:: self.num_pos, :].chunk(2, 0)[0]

        centers = []
        for i in range(id_num):
            centers.append(inputs[targets == targets[i * self.num_pos]].mean(0))
        centers = torch.stack(centers)

        dist_mat = pdist_torch(centers, inputs)
        an = dist_mat * is_neg_c2i
        an = an[an > 1e-6].view(id_num, -1)
        d_neg = torch.mean(an, dim=1, keepdim=True)
        mask_an = (an - d_neg).expand(id_num, total - 2 * self.num_pos).lt(0)
        an = an * mask_an

        list_an = []
        for i in range(id_num):
            values = an[i][an[i] > 1e-6]
            list_an.append(torch.mean(values) if values.numel() else d_neg[i, 0])
        an_mean = sum(list_an) / len(list_an)

        ap = dist_mat * ~is_neg_c2i
        ap_mean = torch.mean(ap[ap > 1e-6])
        return ap_mean / (an_mean + 1e-12)


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

    





