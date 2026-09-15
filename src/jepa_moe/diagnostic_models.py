"""Oracle ablations for diagnosing expert-router coupling."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .models import (
    INPUT_DIM,
    NUM_EXPERTS,
    STATE_DIM,
    PredictorOutput,
    _mlp,
    _model_input,
)
from .synthetic import ground_truth_expert_outputs, true_routing_weights


class OracleRouterMoEPredictor(nn.Module):
    """Trainable experts mixed by the parameter-free ground-truth router."""

    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [_mlp((INPUT_DIM, 128, 128, STATE_DIM)) for _ in range(NUM_EXPERTS)]
        )

    def forward(self, state: Tensor, action: Tensor) -> PredictorOutput:
        model_input = _model_input(state, action)
        expert_outputs = torch.stack(
            [expert(model_input) for expert in self.experts], dim=1
        )
        alpha = true_routing_weights(action)
        pred = torch.sum(alpha.unsqueeze(-1) * expert_outputs, dim=1)
        return PredictorOutput(pred, alpha, expert_outputs)


class OracleExpertMoEPredictor(nn.Module):
    """Parameter-free ground-truth experts mixed by the original learned router."""

    def __init__(self) -> None:
        super().__init__()
        self.router = _mlp((INPUT_DIM, 64, 64, NUM_EXPERTS))

    def forward(self, state: Tensor, action: Tensor) -> PredictorOutput:
        model_input = _model_input(state, action)
        expert_outputs = ground_truth_expert_outputs(state, action)
        alpha = torch.softmax(self.router(model_input), dim=-1)
        pred = torch.sum(alpha.unsqueeze(-1) * expert_outputs, dim=1)
        return PredictorOutput(pred, alpha, expert_outputs)
