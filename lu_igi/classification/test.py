import pandas as pd
import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from .gcn import NUM_CLASSES, CLASSES


def _accuracy(pred, target):
    return (pred == target).sum().item() / target.numel()

def _true_positive(pred, target):
    out = []
    for i in range(NUM_CLASSES):
        out.append(((pred == i) & (target == i)).sum())

    return torch.tensor(out)

def _true_negative(pred, target):
    out = []
    for i in range(NUM_CLASSES):
        out.append(((pred != i) & (target != i)).sum())

    return torch.tensor(out)

def _false_positive(pred, target):
    out = []
    for i in range(NUM_CLASSES):
        out.append(((pred == i) & (target != i)).sum())

    return torch.tensor(out)

def _false_negative(pred, target):
    out = []
    for i in range(NUM_CLASSES):
        out.append(((pred != i) & (target == i)).sum())

    return torch.tensor(out)

def _precision(pred, target):
    tp = _true_positive(pred, target).to(torch.float)
    fp = _false_positive(pred, target).to(torch.float)

    out = tp / (tp + fp)
    out[torch.isnan(out)] = 0

    return out

def _recall(pred, target):
    tp = _true_positive(pred, target).to(torch.float)
    fn = _false_negative(pred, target).to(torch.float)

    out = tp / (tp + fn)
    out[torch.isnan(out)] = 0

    return out

def _f1_score(pred, target):
    prec = _precision(pred, target)
    rec = _recall(pred, target)

    score = 2 * (prec * rec) / (prec + rec)
    score[torch.isnan(score)] = 0

    return score



def _mean_iou(pred, target, batch=None):
    pred, target = F.one_hot(pred, NUM_CLASSES), F.one_hot(target, NUM_CLASSES)

    if batch is not None:
        i = scatter_add(pred & target, batch, dim=0).to(torch.float)
        u = scatter_add(pred | target, batch, dim=0).to(torch.float)
    else:
        i = (pred & target).sum(dim=0).to(torch.float)
        u = (pred | target).sum(dim=0).to(torch.float)

    iou = i / u
    iou[torch.isnan(iou)] = 1
    iou = iou.mean(dim=-1)
    return iou

def test_model(model, data, mask):
    model.eval()
    _, pred = model(data).max(dim=1)

    mask = (mask) & (data.y != -1)

    pred = pred[mask]
    target = data.y[mask]

    df = pd.DataFrame.from_dict(data={
        'precision': _precision(pred, target),
        'recall': _recall(pred, target),
        'f1 score': _f1_score(pred, target)
    }, orient='columns').astype(float)
    df.index = [CLASSES[i] for i in df.index]

    accuracy = _accuracy(pred, target)

    return accuracy, df