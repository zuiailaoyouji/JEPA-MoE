"""Fixed-architecture predictors for the first JEPA-MoE experiment stage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

import torch
from torch import Tensor, nn


STATE_DIM = 4
ACTION_DIM = 2
INPUT_DIM = STATE_DIM + ACTION_DIM
NUM_EXPERTS = 3
HISTORY_LENGTH = 4
HISTORY_CONTEXT_DIM = 64
RouterInput = Literal["state", "state_action"]
Experiment1ModelName = Literal["dense", "moe_a", "moe_s", "ours"]


class PredictorOutput(NamedTuple):
    """Common forward result for dense and MoE predictors."""

    pred: Tensor
    alpha: Tensor | None
    expert_outputs: Tensor | None


@dataclass(frozen=True)
class TrainingLoss:
    """Raw Experiment 1 losses plus optional legacy diagnostics."""

    total: Tensor
    prediction: Tensor
    control_response: Tensor
    balance: Tensor
    specialization: Tensor | None = None
    jacobian_diversity: Tensor | None = None
    activity: Tensor | None = None
    pairwise_cosines: Tensor | None = None
    expert_jacobian_norms: Tensor | None = None
    routing_weights: Tensor | None = None

    def logging_metrics(self) -> dict[str, Tensor]:
        """Return detached scalar tensors suitable for a training logger."""

        metrics = {
            "total_loss": self.total.detach(),
            "prediction_loss": self.prediction.detach(),
            "control_response_loss": self.control_response.detach(),
            "load_balance_loss": self.balance.detach(),
        }
        if self.specialization is None:
            return metrics
        metrics["control_jacobian_specialization_loss"] = (
            self.specialization.detach()
        )
        if self.jacobian_diversity is None:
            return metrics

        assert self.activity is not None
        assert self.pairwise_cosines is not None
        assert self.expert_jacobian_norms is not None
        assert self.routing_weights is not None
        metrics["jacobian_diversity_loss"] = self.jacobian_diversity.detach()
        metrics["activity_loss"] = self.activity.detach()

        pair_index = 0
        for first in range(NUM_EXPERTS):
            for second in range(first + 1, NUM_EXPERTS):
                metrics[f"jacobian_cosine_{first}_{second}"] = (
                    self.pairwise_cosines[pair_index].detach()
                )
                pair_index += 1
        for expert_index in range(NUM_EXPERTS):
            metrics[f"jacobian_norm_{expert_index}"] = (
                self.expert_jacobian_norms[expert_index].detach()
            )
            metrics[f"routing_weight_{expert_index}"] = (
                self.routing_weights[expert_index].detach()
            )
        return metrics


def _validate_inputs(state: Tensor, action: Tensor) -> None:
    if state.ndim != 2 or state.shape[-1] != STATE_DIM:
        raise ValueError(
            f"state must have shape [batch_size, {STATE_DIM}], got {tuple(state.shape)}"
        )
    if action.ndim != 2 or action.shape[-1] != ACTION_DIM:
        raise ValueError(
            f"action must have shape [batch_size, {ACTION_DIM}], got {tuple(action.shape)}"
        )
    if state.shape[0] != action.shape[0]:
        raise ValueError("state and action must have the same batch size")
    if state.device != action.device:
        raise ValueError("state and action must be on the same device")
    if state.dtype != action.dtype:
        raise ValueError("state and action must have the same dtype")


def _model_input(state: Tensor, action: Tensor) -> Tensor:
    _validate_inputs(state, action)
    return torch.cat((state, action), dim=-1)


def next_state_mse(pred: Tensor, next_state: Tensor) -> Tensor:
    if next_state.shape != pred.shape:
        raise ValueError(
            f"next_state must have shape {tuple(pred.shape)}, got {tuple(next_state.shape)}"
        )
    return (pred - next_state).square().mean()


def _mlp(widths: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (in_features, out_features) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(in_features, out_features))
        if index < len(widths) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class DensePredictor(nn.Module):
    """A dense baseline with the same input and result interface as the MoE models."""

    def __init__(self) -> None:
        super().__init__()
        self.network = _mlp((INPUT_DIM, 128, 128, STATE_DIM))

    def forward(self, state: Tensor, action: Tensor) -> PredictorOutput:
        pred = self.network(_model_input(state, action))
        return PredictorOutput(pred=pred, alpha=None, expert_outputs=None)

    def training_loss(
        self, state: Tensor, action: Tensor, next_state: Tensor
    ) -> TrainingLoss:
        output = self(state, action)
        prediction = next_state_mse(output.pred, next_state)
        zero = prediction.new_zeros(())
        return TrainingLoss(
            total=prediction,
            prediction=prediction,
            control_response=zero,
            balance=zero,
        )


class MoEPredictor(nn.Module):
    """Experiment 1 MoE with an explicit state-only or state-action router."""

    def __init__(
        self,
        *,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        num_experts: int = NUM_EXPERTS,
        hidden_dim: int = 128,
        router_hidden_dim: int = 64,
        router_input: RouterInput = "state",
    ) -> None:
        super().__init__()
        if min(
            state_dim,
            action_dim,
            num_experts,
            hidden_dim,
            router_hidden_dim,
        ) <= 0:
            raise ValueError("all model dimensions must be positive")
        if router_input not in ("state", "state_action"):
            raise ValueError(
                "router_input must be either 'state' or 'state_action'"
            )

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_experts = num_experts
        self.router_input = router_input
        expert_input_dim = state_dim + action_dim

        # Experts are initialized before the router so their RNG path is identical
        # for state-only and state-action routing under the same construction seed.
        self.experts = nn.ModuleList(
            [
                _mlp((expert_input_dim, hidden_dim, hidden_dim, state_dim))
                for _ in range(num_experts)
            ]
        )
        router_input_dim = state_dim if router_input == "state" else expert_input_dim
        self.router = _mlp(
            (router_input_dim, router_hidden_dim, router_hidden_dim, num_experts)
        )

    def _validate_inputs(self, state: Tensor, action: Tensor) -> None:
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(
                f"state must have shape [batch_size, {self.state_dim}], "
                f"got {tuple(state.shape)}"
            )
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"action must have shape [batch_size, {self.action_dim}], "
                f"got {tuple(action.shape)}"
            )
        if state.shape[0] != action.shape[0]:
            raise ValueError("state and action must have the same batch size")
        if state.device != action.device:
            raise ValueError("state and action must be on the same device")
        if state.dtype != action.dtype:
            raise ValueError("state and action must have the same dtype")

    def forward(self, state: Tensor, action: Tensor) -> PredictorOutput:
        self._validate_inputs(state, action)
        expert_input = torch.cat((state, action), dim=-1)
        expert_outputs = torch.stack(
            [expert(expert_input) for expert in self.experts], dim=1
        )
        router_features = (
            state if self.router_input == "state" else expert_input
        )
        alpha = torch.softmax(self.router(router_features), dim=-1)
        pred = torch.sum(alpha.unsqueeze(-1) * expert_outputs, dim=1)
        return PredictorOutput(pred=pred, alpha=alpha, expert_outputs=expert_outputs)


def _validate_history_inputs(
    state_history: Tensor, past_actions: Tensor
) -> None:
    batch_size = state_history.shape[0] if state_history.ndim > 0 else None
    expected_state_shape = (batch_size, HISTORY_LENGTH + 1, STATE_DIM)
    expected_action_shape = (batch_size, HISTORY_LENGTH, ACTION_DIM)
    if state_history.ndim != 3 or state_history.shape[1:] != expected_state_shape[1:]:
        raise ValueError(
            "state_history must have shape "
            f"[batch_size, {HISTORY_LENGTH + 1}, {STATE_DIM}], "
            f"got {tuple(state_history.shape)}"
        )
    if past_actions.ndim != 3 or past_actions.shape[1:] != expected_action_shape[1:]:
        raise ValueError(
            "past_actions must have shape "
            f"[batch_size, {HISTORY_LENGTH}, {ACTION_DIM}], "
            f"got {tuple(past_actions.shape)}"
        )
    if state_history.shape[0] != past_actions.shape[0]:
        raise ValueError("state_history and past_actions must have the same batch size")
    if state_history.device != past_actions.device:
        raise ValueError("state_history and past_actions must be on the same device")
    if state_history.dtype != past_actions.dtype:
        raise ValueError("state_history and past_actions must have the same dtype")


def flatten_history(state_history: Tensor, past_actions: Tensor) -> Tensor:
    """Interleave four past state-action pairs and append the current state."""

    _validate_history_inputs(state_history, past_actions)
    past_pairs = torch.cat((state_history[:, :-1], past_actions), dim=-1)
    return torch.cat(
        (past_pairs.flatten(start_dim=1), state_history[:, -1]), dim=-1
    )


class HistoryContextEncoder(nn.Module):
    """Encode an H=4 state-action history without observing the current action."""

    input_dim = (HISTORY_LENGTH + 1) * STATE_DIM + HISTORY_LENGTH * ACTION_DIM
    context_dim = HISTORY_CONTEXT_DIM

    def __init__(self) -> None:
        super().__init__()
        self.network = _mlp((self.input_dim, 128, 128, self.context_dim))

    def forward(self, state_history: Tensor, past_actions: Tensor) -> Tensor:
        return self.network(flatten_history(state_history, past_actions))


class HistoryRouter(nn.Module):
    """Map a history context to normalized expert composition weights."""

    def __init__(self) -> None:
        super().__init__()
        self.network = _mlp((HISTORY_CONTEXT_DIM, 128, 128, NUM_EXPERTS))

    def forward(self, context: Tensor) -> Tensor:
        if context.ndim != 2 or context.shape[-1] != HISTORY_CONTEXT_DIM:
            raise ValueError(
                "context must have shape "
                f"[batch_size, {HISTORY_CONTEXT_DIM}], got {tuple(context.shape)}"
            )
        return torch.softmax(self.network(context), dim=-1)


class HistoryConditionedMoEPredictor(nn.Module):
    """Experiment 2 MoE shared by the baseline and control-response objective."""

    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [_mlp((INPUT_DIM, 128, 128, STATE_DIM)) for _ in range(NUM_EXPERTS)]
        )
        self.context_encoder = HistoryContextEncoder()
        self.router = HistoryRouter()

    def routing_weights(
        self, state_history: Tensor, past_actions: Tensor
    ) -> tuple[Tensor, Tensor]:
        context = self.context_encoder(state_history, past_actions)
        return context, self.router(context)

    def forward(
        self,
        state_history: Tensor,
        past_actions: Tensor,
        current_action: Tensor,
    ) -> PredictorOutput:
        _, alpha = self.routing_weights(state_history, past_actions)
        current_state = state_history[:, -1]
        expert_input = _model_input(current_state, current_action)
        expert_outputs = torch.stack(
            [expert(expert_input) for expert in self.experts], dim=1
        )
        pred = torch.sum(alpha.unsqueeze(-1) * expert_outputs, dim=1)
        return PredictorOutput(pred=pred, alpha=alpha, expert_outputs=expert_outputs)


def build_experiment2_model(
    *, initialization_seed: int | None = None
) -> HistoryConditionedMoEPredictor:
    """Construct the single shared Experiment 2 predictor architecture on CPU."""

    if initialization_seed is None:
        return HistoryConditionedMoEPredictor()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed)
        return HistoryConditionedMoEPredictor()


def build_experiment1_model(
    model_name: Experiment1ModelName,
    *,
    initialization_seed: int | None = None,
) -> nn.Module:
    """Construct one of the four Experiment 1 baselines on CPU."""

    def construct() -> nn.Module:
        if model_name == "dense":
            return DensePredictor()
        if model_name == "moe_a":
            return MoEPredictor(router_input="state_action")
        if model_name in ("moe_s", "ours"):
            return MoEPredictor(router_input="state")
        raise ValueError(f"unknown Experiment 1 model: {model_name}")

    if initialization_seed is None:
        return construct()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(initialization_seed)
        return construct()


class _MoEPredictor(nn.Module):
    """The single shared architecture definition used by both MoE conditions."""

    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList(
            [_mlp((INPUT_DIM, 128, 128, STATE_DIM)) for _ in range(NUM_EXPERTS)]
        )
        self.router = _mlp((INPUT_DIM, 64, 64, NUM_EXPERTS))

    def forward(self, state: Tensor, action: Tensor) -> PredictorOutput:
        model_input = _model_input(state, action)
        expert_outputs = torch.stack(
            [expert(model_input) for expert in self.experts], dim=1
        )
        alpha = torch.softmax(self.router(model_input), dim=-1)

        # This is exactly sum(alpha[:, k:k+1] * expert_output[k]) in batched form.
        pred = torch.sum(alpha.unsqueeze(-1) * expert_outputs, dim=1)
        return PredictorOutput(pred=pred, alpha=alpha, expert_outputs=expert_outputs)

    def training_loss(
        self, state: Tensor, action: Tensor, next_state: Tensor
    ) -> TrainingLoss:
        from .losses import control_jacobian_specialization_terms

        output = self(state, action)
        prediction = next_state_mse(output.pred, next_state)
        diagnostics = control_jacobian_specialization_terms(
            self, state, action, create_graph=False
        )
        return TrainingLoss(
            total=prediction,
            prediction=prediction,
            control_response=prediction.new_zeros(()),
            balance=prediction.new_zeros(()),
            specialization=diagnostics.total,
            jacobian_diversity=diagnostics.diversity,
            activity=diagnostics.activity,
            pairwise_cosines=diagnostics.pairwise_cosines,
            expert_jacobian_norms=diagnostics.expert_jacobian_norms,
            routing_weights=diagnostics.routing_weights,
        )


class VanillaMoEPredictor(_MoEPredictor):
    """MoE trained only with next-state prediction loss."""


class JacobianMoEPredictor(_MoEPredictor):
    """The identical MoE architecture with an added Jacobian training objective."""

    def training_loss(
        self,
        state: Tensor,
        action: Tensor,
        next_state: Tensor,
        lambda_jac: float = 0.1,
        margin: float = 0.3,
        min_jacobian_norm: float = 0.05,
        beta_activity: float = 0.1,
        eps: float = 1e-8,
    ) -> TrainingLoss:
        if lambda_jac < 0:
            raise ValueError("lambda_jac must be non-negative")

        from .losses import control_jacobian_specialization_terms

        output = self(state, action)
        prediction = next_state_mse(output.pred, next_state)
        specialization_terms = control_jacobian_specialization_terms(
            self,
            state,
            action,
            create_graph=True,
            margin=margin,
            min_jacobian_norm=min_jacobian_norm,
            beta_activity=beta_activity,
            eps=eps,
        )
        total = prediction + lambda_jac * specialization_terms.total
        return TrainingLoss(
            total=total,
            prediction=prediction,
            control_response=prediction.new_zeros(()),
            balance=prediction.new_zeros(()),
            specialization=specialization_terms.total,
            jacobian_diversity=specialization_terms.diversity,
            activity=specialization_terms.activity,
            pairwise_cosines=specialization_terms.pairwise_cosines,
            expert_jacobian_norms=specialization_terms.expert_jacobian_norms,
            routing_weights=specialization_terms.routing_weights,
        )
