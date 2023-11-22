import numpy as np
import torch
import torch.nn.functional as F

def nll_loss(output, target):
    return F.nll_loss(output, target)


def MSE_loss(output, target):
    return F.mse_loss(output, target)


def CE_loss(output, target):
    return F.cross_entropy(output, target)


def normalize(x, norm=True, no_similarity_std=False):
    if x.dim() == 4:
        if not no_similarity_std and x.size(1) > 3 and x.size(2) > 1:
            z = F.max_pool2d(x, x.shape[2])
            x = z.reshape(z.shape[0], -1)
        else:
            x = x.reshape(x.size(0), -1)
    if norm:
        xc = x - x.mean(dim=1).unsqueeze(1)
        return xc / (1e-8 + torch.sqrt(torch.sum(xc ** 2, dim=1))).unsqueeze(1)
    else:
        return x


def correlation_loss(xn):
    x_corr = (xn.matmul(xn.transpose(1, 0))**2).clamp(-1, 1)
    x_corr[np.diag_indices(x_corr.shape[0])] = 0.0
    return torch.mean(x_corr)


def proto_vector_loss(x, y, no_similarity_std=False):
    if x.dim() == 4:
        if not no_similarity_std and x.size(1) > 3 and x.size(2) > 1:
            z = x.view(x.size(0), x.size(1), -1)
            x = z.std(dim=2)
        else:
            x = x.view(x.size(0), -1)
    xc = x - x.mean(dim=1).unsqueeze(1)
    xn = xc / (1e-8 + torch.sqrt(torch.sum(xc ** 2, dim=1))).unsqueeze(1)
    num_classes = y.shape[1]
    y = torch.where(y)[1].detach()
    x_proto = torch.cat([xn[y == i].mean(dim=0, keepdim=True) for i in range(num_classes)]).detach()
    loss_corr = torch.mean((xn.matmul(xn.transpose(1, 0)) ** 2).clamp(-1, 1))
    loss_sim = F.mse_loss(xn, x_proto[y], reduction='sum') / xn.shape[0] ** 2
    return loss_corr, loss_sim


def trainable_proto_vector_loss(x, y):
    xn = normalize(x)
    loss_corr = correlation_loss(xn)
    loss_sim = torch.sum(y * xn) / xn.shape[0] ** 2  # FIXME: add parameter
    return loss_corr, loss_sim


def get_proto_vectors(x, labels, num_classes=None):
    xn = normalize(x)
    num_classes = num_classes if num_classes is not None else labels.shape[1]
    if labels.dim() > 1:
        # One hot labels
        labels = torch.where(labels)[1].detach()
    else:
        # Int labels
        labels = labels.detach()
    x_proto = [None]*num_classes
    for i in range(num_classes):
        num_samples = torch.sum(labels == i)
        if num_samples > 0:
            x_proto[i] = xn[labels == i].mean(dim=0, keepdim=True)
        else:
            x_proto[i] = torch.zeros_like(xn[0]).unsqueeze(0)
    x_proto = torch.cat(x_proto).detach()
    return x_proto, labels


def nokland_sim_loss(fwd_logits, bwd_logits, bias=0.0, temperature=1.0, targets=None):
    return sum(proto_vector_loss(fwd_logits, targets))


