"""Minimal MoE dynamics predictors for controlled experiments."""

from .diagnostic_models import OracleExpertMoEPredictor, OracleRouterMoEPredictor
from .jacobian import (
    ControlJacobianDecomposition,
    control_jacobian_decomposition,
    expert_action_jacobians,
    router_action_jacobian,
)
from .losses import (
    ControlJacobianSpecialization,
    control_jacobian_specialization_loss,
    control_jacobian_specialization_terms,
)
from .models import (
    ACTION_DIM,
    INPUT_DIM,
    NUM_EXPERTS,
    STATE_DIM,
    DensePredictor,
    JacobianMoEPredictor,
    PredictorOutput,
    TrainingLoss,
    VanillaMoEPredictor,
    next_state_mse,
)
from .synthetic import (
    HELDOUT_ACTION_BOUND,
    RolloutBatch,
    TransitionBatch,
    ground_truth_expert_action_jacobians,
    ground_truth_expert_outputs,
    make_rollout_batch,
    make_transition_batch,
    synthetic_transition,
    true_routing_weights,
)

__all__ = [
    "ACTION_DIM",
    "INPUT_DIM",
    "NUM_EXPERTS",
    "STATE_DIM",
    "ControlJacobianDecomposition",
    "ControlJacobianSpecialization",
    "DensePredictor",
    "HELDOUT_ACTION_BOUND",
    "JacobianMoEPredictor",
    "OracleExpertMoEPredictor",
    "OracleRouterMoEPredictor",
    "PredictorOutput",
    "RolloutBatch",
    "TrainingLoss",
    "TransitionBatch",
    "VanillaMoEPredictor",
    "control_jacobian_decomposition",
    "control_jacobian_specialization_loss",
    "control_jacobian_specialization_terms",
    "expert_action_jacobians",
    "ground_truth_expert_action_jacobians",
    "ground_truth_expert_outputs",
    "make_rollout_batch",
    "make_transition_batch",
    "next_state_mse",
    "router_action_jacobian",
    "synthetic_transition",
    "true_routing_weights",
]
