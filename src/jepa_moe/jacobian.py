"""Autograd utilities for decomposing a soft MoE control Jacobian."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from .models import ACTION_DIM, NUM_EXPERTS, STATE_DIM, MoEPredictor, _MoEPredictor
from .synthetic import basis_composition_action_jacobian


class ControlJacobianDecomposition(NamedTuple):
    """Every term in d(prediction) / d(action)."""

    expert_jacobians: Tensor
    router_jacobian: Tensor
    j_within: Tensor
    j_switch: Tensor
    direct: Tensor
    reconstructed: Tensor


def _differentiable_action(action: Tensor) -> Tensor:
    if action.requires_grad:
        return action
    return action.detach().requires_grad_(True)


def _batched_jacobian(
    output: Tensor, inputs: Tensor, *, create_graph: bool
) -> Tensor:
    """Return per-sample Jacobians for a batch-separable computation."""

    gradients = []
    for output_index in range(output.shape[-1]):
        gradient = torch.autograd.grad(
            output[:, output_index].sum(),
            inputs,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        gradients.append(gradient)
    return torch.stack(gradients, dim=1)


def expert_action_jacobians(
    model: _MoEPredictor,
    state: Tensor,
    action: Tensor,
    *,
    create_graph: bool = False,
) -> Tensor:
    """Compute all B_k with shape [batch, num_experts, state_dim, action_dim]."""

    action_for_grad = _differentiable_action(action)
    output = model(state, action_for_grad)
    assert output.expert_outputs is not None
    jacobians = [
        _batched_jacobian(
            output.expert_outputs[:, expert_index, :],
            action_for_grad,
            create_graph=create_graph,
        )
        for expert_index in range(NUM_EXPERTS)
    ]
    result = torch.stack(jacobians, dim=1)
    expected_shape = (state.shape[0], NUM_EXPERTS, STATE_DIM, ACTION_DIM)
    if result.shape != expected_shape:
        raise RuntimeError(f"unexpected expert Jacobian shape: {tuple(result.shape)}")
    return result


def predictor_action_jacobian(
    model: nn.Module,
    state: Tensor,
    action: Tensor,
    *,
    create_graph: bool = False,
) -> Tensor:
    """Return the complete d prediction / d action for any predictor."""

    action_for_grad = _differentiable_action(action)
    prediction = model(state, action_for_grad).pred
    result = _batched_jacobian(
        prediction, action_for_grad, create_graph=create_graph
    )
    expected_shape = (state.shape[0], prediction.shape[-1], action.shape[-1])
    if result.shape != expected_shape:
        raise RuntimeError(f"unexpected predictor Jacobian shape: {tuple(result.shape)}")
    return result


def _validate_action_direction(action: Tensor, direction: Tensor) -> None:
    if direction.shape != action.shape:
        raise ValueError(
            f"direction must have shape {tuple(action.shape)}, "
            f"got {tuple(direction.shape)}"
        )
    if direction.device != action.device:
        raise ValueError("direction and action must be on the same device")
    if direction.dtype != action.dtype:
        raise ValueError("direction and action must have the same dtype")


def expert_action_jvp(
    model: MoEPredictor,
    state: Tensor,
    action: Tensor,
    direction: Tensor,
) -> Tensor:
    """Return B_k(x, a) v without materializing the expert Jacobians."""

    _validate_action_direction(action, direction)

    def expert_predictions(candidate_action: Tensor) -> Tensor:
        outputs = model(state, candidate_action).expert_outputs
        if outputs is None:
            raise TypeError("expert_action_jvp requires an MoE predictor")
        return outputs

    _, responses = torch.func.jvp(
        expert_predictions,
        (action,),
        (direction,),
    )
    expected_shape = (state.shape[0], model.num_experts, model.state_dim)
    if responses.shape != expected_shape:
        raise RuntimeError(f"unexpected expert JVP shape: {tuple(responses.shape)}")
    return responses


def mixture_action_response(
    model: MoEPredictor,
    state: Tensor,
    action: Tensor,
    direction: Tensor,
) -> Tensor:
    """Return sum_k alpha_k(x) B_k(x, a) v for a state-only MoE."""

    if model.router_input != "state":
        raise ValueError("mixture_action_response requires a state-only router")
    output = model(state, action)
    if output.alpha is None:
        raise TypeError("mixture_action_response requires an MoE predictor")
    expert_responses = expert_action_jvp(model, state, action, direction)
    return torch.sum(output.alpha.unsqueeze(-1) * expert_responses, dim=1)


def basis_composition_target_response(
    state: Tensor,
    action: Tensor,
    direction: Tensor,
) -> Tensor:
    """Return the detached analytic J_true(x, a) v target for Experiment 1."""

    _validate_action_direction(action, direction)
    with torch.no_grad():
        jacobian = basis_composition_action_jacobian(state, action)
        return torch.einsum("bsa,ba->bs", jacobian, direction)


def router_action_jacobian(
    model: _MoEPredictor,
    state: Tensor,
    action: Tensor,
    *,
    create_graph: bool = False,
) -> Tensor:
    """Compute d alpha / d action with shape [batch, num_experts, action_dim]."""

    action_for_grad = _differentiable_action(action)
    output = model(state, action_for_grad)
    assert output.alpha is not None
    result = _batched_jacobian(
        output.alpha, action_for_grad, create_graph=create_graph
    )
    expected_shape = (state.shape[0], NUM_EXPERTS, ACTION_DIM)
    if result.shape != expected_shape:
        raise RuntimeError(f"unexpected router Jacobian shape: {tuple(result.shape)}")
    return result


def control_jacobian_decomposition(
    model: _MoEPredictor,
    state: Tensor,
    action: Tensor,
    *,
    create_graph: bool = False,
) -> ControlJacobianDecomposition:
    """Compute and directly verify the within-expert plus switching decomposition."""

    action_for_grad = _differentiable_action(action)
    output = model(state, action_for_grad)
    assert output.alpha is not None
    assert output.expert_outputs is not None

    expert_jacobians = torch.stack(
        [
            _batched_jacobian(
                output.expert_outputs[:, expert_index, :],
                action_for_grad,
                create_graph=create_graph,
            )
            for expert_index in range(NUM_EXPERTS)
        ],
        dim=1,
    )
    router_jacobian = _batched_jacobian(
        output.alpha, action_for_grad, create_graph=create_graph
    )
    direct = _batched_jacobian(
        output.pred, action_for_grad, create_graph=create_graph
    )

    j_within = torch.einsum("bk,bksa->bsa", output.alpha, expert_jacobians)
    j_switch = torch.einsum(
        "bks,bka->bsa", output.expert_outputs, router_jacobian
    )
    reconstructed = j_within + j_switch
    return ControlJacobianDecomposition(
        expert_jacobians=expert_jacobians,
        router_jacobian=router_jacobian,
        j_within=j_within,
        j_switch=j_switch,
        direct=direct,
        reconstructed=reconstructed,
    )
