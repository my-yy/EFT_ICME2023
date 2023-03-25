import torch


def calc_bce_acc_by_tensor(logits_tensor, labels_tensor):
    assert len(labels_tensor.shape) == 1
    assert len(logits_tensor.shape) == 1
    probs = torch.sigmoid(logits_tensor)
    zero_one = probs > 0.5
    acc_tensor = torch.sum(zero_one == labels_tensor) / len(probs)
    return acc_tensor.item()
