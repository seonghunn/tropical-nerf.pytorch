# TropicalNeRF
# Copyright (c) 2024-present NAVER Cloud Corp.
# CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)

from typing import *
from torch import Tensor, LongTensor
from torch.nn import Module

import numpy as np
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def nonzero_last(t: Tensor) -> Tensor:
    assert 2 == len(t.shape)
    inds = t.new_zeros(t.nonzero()[:, 0].unique().shape[0], 2).long()
    idx = -1
    idx_v = -1
    for x in t.nonzero():
        if x[0] != idx_v:
            idx += 1
            idx_v = x[0]
        inds[idx, 0] = x[0]
        inds[idx, 1] = x[1]
    return inds


def nonzero_first(t: Tensor) -> Tensor:
    assert 2 == len(t.shape)
    inds = t.new_zeros(t.nonzero()[:, 0].unique().shape[0], 2).long()
    idx = -1
    idx_v = -1
    for x in t.nonzero():
        if x[0] != idx_v:
            idx += 1
            idx_v = x[0]
            inds[idx, 0] = x[0]
            inds[idx, 1] = x[1]
    return inds


# Batched index_select
def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b x e x f
    return out


# Batched unique_consecutive. warning: it's slow.
def batched_unique_consecutive(t, null_value=-1):
    # Apply torch.unique_consecutive to each row
    unique_consecutive_rows = [torch.unique_consecutive(row) for row in t]

    # Determine the maximum length of the unique consecutive rows
    max_length = max(len(row) for row in unique_consecutive_rows)

    # Pad the rows to the maximum length and stack them into a tensor
    padded_rows = [F.pad(row, (0, max_length - len(row)), value=null_value)
                   for row in unique_consecutive_rows]
    result_tensor = torch.stack(padded_rows)

    return result_tensor
