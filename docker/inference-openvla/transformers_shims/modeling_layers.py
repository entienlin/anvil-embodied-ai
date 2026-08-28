"""Compatibility shim, see masking_utils.py in this same directory for why."""

import torch.nn as nn


class GradientCheckpointingLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "GradientCheckpointingLayer is a compatibility import-shim stub — "
            "PI0/PI0Fast are not usable in this environment."
        )
