import math
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

def debug(x):
    print(x)
    print(x.shape)
    input()

class SyncFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size
        return grad_input[idx_from:idx_to]

class InfoNCELoss(nn.Module):
    def __init__(self, tau=0.05):
        super(InfoNCELoss, self).__init__()
        self.tau = tau

    def forward(self, feature_1, feature_2, eps=1e-6):
        """
        assume out_1 and out_2 are normalized
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        # out_1_dist: [batch_size * world_size, dim]
        # out_2_dist: [batch_size * world_size, dim]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(feature_1)
            out_2_dist = SyncFunction.apply(feature_2)
        else:
            out_1_dist = feature_1
            out_2_dist = feature_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([feature_1, feature_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / self.tau)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = Tensor(neg.shape).fill_(math.e ** (1 / self.tau)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(feature_1 * feature_2, dim=-1) / self.tau)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss