import torch
from torch import nn
from torch.nn import functional as F


class InfoNCE(nn.Module):
    def __init__(self, temperature):
        super(InfoNCE, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.temperature = temperature

    def forward(self, batch1, batch2):
        batch_size = batch1.size(0)
        batch1 = F.normalize(batch1)  # (bs, out_dim)
        batch2 = F.normalize(batch2)  # (bs, out_dim)

        similarity_matrix = torch.matmul(batch1, batch2.T)  # (bs, bs)

        mask = torch.eye(batch_size, dtype=torch.bool)  # (bs, bs)
        assert similarity_matrix.shape == mask.shape

        positives = similarity_matrix[mask].view(batch_size, -1)  # (bs,1)

        negatives = similarity_matrix[~mask].view(batch_size, -1)  # (bs,bs-1)

        logits = torch.cat([positives, negatives], dim=1)
        # (bs, bs)

        labels = torch.zeros(batch_size, dtype=torch.long).cuda()  # (bs)

        logits = logits / self.temperature
        loss = self.ce(logits, labels)
        # ipdb.set_trace()
        return loss, logits


class InfoNCE_v2(nn.Module):
    def __init__(self, temperature, reduction="mean"):
        super(InfoNCE_v2, self).__init__()
        self.ce = nn.CrossEntropyLoss(reduction=reduction)
        self.temperature = temperature

    def forward(self, anchor, positive):
        batch_size = anchor.size(0)
        anchor = F.normalize(anchor)
        positive = F.normalize(positive)

        similarity_matrix = torch.matmul(anchor, positive.T)  # (bs, bs)

        logits = similarity_matrix / self.temperature
        labels = torch.LongTensor([i for i in range(batch_size)]).cuda()
        loss = self.ce(logits, labels)
        return loss, logits
