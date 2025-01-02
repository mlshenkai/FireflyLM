# -*- coding: utf-8 -*-
# @Author: watcher
# @Created Time: 2024/12/26 14:31
# @File: softmoe_model
# @Email: mlshenkai@163.com
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from .firefly_model import FireflyConfig


def softmax(x: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:

    dtype = x.dtype
    x = x.to(torch.float32)
    max_vals = torch.amax(x, dim=dim, keepdim=True)
    e_x = torch.exp(x - max_vals)
    sum_exp = e_x.sum(dim=dim, keepdim=True)
    return (e_x / sum_exp).to(dtype)


# copy from https://github.com/bwconrad/soft-moe
class EasySoftMOE(nn.Module):
    def __init__(self, config: FireflyConfig, layer: Callable) -> None:
        super().__init__()
        self.dim = config.hidden_size
        self.num_exports = config.n_experts
        self.slots_per_expert = config.slots_per_expert
        self.normalize = True

        self.phi = nn.Parameter(
            torch.zeros(self.dim, self.num_exports, self.slots_per_expert)
        )
        if self.normalize:
            self.scale = nn.Parameter(torch.ones(1))

        nn.init.normal_(self.phi, mean=0, std=1 / self.dim**0.5)

        self.experts = nn.ModuleList([layer(config) for _ in range(self.num_exports)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"
        assert (
            len(x.shape) == 3
        ), f"Input expected to have 3 dimensions but has {len(x.shape)}"

        phi = self.phi

        # Normalize input and phi
        if self.normalize:
            x = F.normalize(x, dim=2)  # [b, m, d]
            phi = self.scale * F.normalize(phi, dim=0)  # [d, n, p]

        # Compute dispatch and combine weights
        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        d = softmax(logits, dim=1)
        c = softmax(logits, dim=(2, 3))
        # tmp = c[0,:,:,0].reshape([c.shape[1],-1])
        # print("num:",tmp, "shape:",tmp.shape, "sum:",tmp.sum(dim=1))
        # Compute input slots as weighted average of input tokens using dispatch weights
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)

        # Apply expert to corresponding slots
        ys = torch.stack(
            [f_i(xs[:, i, :, :]) for i, f_i in enumerate(self.experts)], dim=1
        )

        # Compute output tokens as weighted average of output slots using combine weights
        y = torch.einsum("bnpd,bmnp->bmd", ys, c)

        return y
