"""Unified Compression Action Space

Defines abstract compression actions that apply uniformly across all ONNX operator
types. The same 4 action dimensions (precision_level, prune_ratio, fuse_flag,
approx_rank) are searched by Optuna, and a thin adapter layer translates them
into operator-type-specific implementations.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any

from peops.graph.onnx_analyzer import OperatorCategory, OperatorInfo


class PrecisionLevel(enum.IntEnum):
    FP32 = 0
    FP16 = 1
    INT8 = 2
    INT4 = 3


@dataclass(frozen=True)
class CompressionConfig:
    """Abstract compression configuration for a single operator."""
    precision_level: PrecisionLevel = PrecisionLevel.FP32
    prune_ratio: float = 0.0
    fuse_enabled: bool = False
    approx_rank: int | None = None

    def is_no_compression(self) -> bool:
        return (
            self.precision_level == PrecisionLevel.FP32
            and self.prune_ratio == 0.0
            and not self.fuse_enabled
            and self.approx_rank is None
        )


@dataclass
class ActionSpace:
    """Available compression actions for an operator, constrained by type and sensitivity."""
    operator_name: str
    category: OperatorCategory
    allowed_precisions: list[PrecisionLevel]
    prune_ratio_range: tuple[float, float]
    fuse_available: bool
    approx_rank_range: tuple[int, int] | None

    def default_config(self) -> CompressionConfig:
        return CompressionConfig()


# INT4 is intentionally absent: TensorProto.INT4 needs opset >= 21 (the
# supported model zoo is opset 13/14) and sub-byte packing is impossible below
# that, so INT4 could never deliver real bytes beyond INT8's 4x. The
# transformer degrades any residual INT4 action to 16-level int8 storage.
_NEURAL_COMPUTE_ACTIONS = ActionSpace(
    operator_name="",
    category=OperatorCategory.DENSE_COMPUTE,
    allowed_precisions=[PrecisionLevel.FP32, PrecisionLevel.FP16, PrecisionLevel.INT8],
    prune_ratio_range=(0.0, 0.9),
    fuse_available=True,
    approx_rank_range=(1, 512),
)

_NORMALIZATION_ACTIONS = ActionSpace(
    operator_name="",
    category=OperatorCategory.NORMALIZATION,
    allowed_precisions=[PrecisionLevel.FP32, PrecisionLevel.FP16],
    prune_ratio_range=(0.0, 0.0),
    fuse_available=True,
    approx_rank_range=None,
)

_ACTIVATION_ACTIONS = ActionSpace(
    operator_name="",
    category=OperatorCategory.ACTIVATION,
    allowed_precisions=[PrecisionLevel.FP32],
    prune_ratio_range=(0.0, 0.0),
    fuse_available=True,
    approx_rank_range=None,
)

_EMBEDDING_ACTIONS = ActionSpace(
    operator_name="",
    category=OperatorCategory.EMBEDDING,
    allowed_precisions=[PrecisionLevel.FP32, PrecisionLevel.FP16, PrecisionLevel.INT8],
    prune_ratio_range=(0.0, 0.5),
    fuse_available=False,
    approx_rank_range=(1, 256),
)

_TREE_ENSEMBLE_ACTIONS = ActionSpace(
    operator_name="",
    category=OperatorCategory.TREE_ENSEMBLE,
    allowed_precisions=[PrecisionLevel.FP32, PrecisionLevel.FP16],
    prune_ratio_range=(0.0, 0.9),
    fuse_available=True,      # leaf merging
    approx_rank_range=(1, 50),  # max_depth limiting
)

_LINEAR_MODEL_ACTIONS = ActionSpace(
    operator_name="",
    category=OperatorCategory.LINEAR_MODEL,
    allowed_precisions=[PrecisionLevel.FP32, PrecisionLevel.FP16, PrecisionLevel.INT8],
    prune_ratio_range=(0.0, 0.8),
    fuse_available=False,
    approx_rank_range=None,
)

_SVM_ACTIONS = ActionSpace(
    operator_name="",
    category=OperatorCategory.SVM,
    allowed_precisions=[PrecisionLevel.FP32, PrecisionLevel.FP16],
    prune_ratio_range=(0.0, 0.8),
    fuse_available=False,
    approx_rank_range=None,
)

_CATEGORY_ACTION_TEMPLATES: dict[OperatorCategory, ActionSpace] = {
    OperatorCategory.DENSE_COMPUTE: _NEURAL_COMPUTE_ACTIONS,
    OperatorCategory.NORMALIZATION: _NORMALIZATION_ACTIONS,
    OperatorCategory.ACTIVATION: _ACTIVATION_ACTIONS,
    OperatorCategory.EMBEDDING: _EMBEDDING_ACTIONS,
    OperatorCategory.TREE_ENSEMBLE: _TREE_ENSEMBLE_ACTIONS,
    OperatorCategory.LINEAR_MODEL: _LINEAR_MODEL_ACTIONS,
    OperatorCategory.SVM: _SVM_ACTIONS,
}


# Operators holding less than this share of total weight bytes get a singleton
# action space: compressing them can't move the size needle, so spending Optuna
# search dimensions on them only dilutes TPE over the layers that matter.
TINY_OP_BYTES_SHARE = 0.005


def get_action_space(
    op: OperatorInfo,
    sensitivity: float | None = None,
    sensitivity_threshold: float = 0.1,
    is_protected: bool | None = None,
    param_share: float | None = None,
) -> ActionSpace:
    """Get the available compression action space for an operator.

    Protection semantics (matches the validated UOSA-mixed configuration):
    pass ``is_protected`` from ``SensitivityProfile.get_protection_set`` —
    rank-based top-p membership. When ``is_protected`` is None, the legacy
    normalized-score threshold behavior applies for backward compatibility.

    Byte awareness: ``param_share`` (this op's parameter bytes / total model
    weight bytes) collapses negligible operators to a singleton space —
    full INT8 when unprotected, untouched FP32 when protected — removing
    wasted search dimensions without changing reachable outcomes materially.
    """
    template = _CATEGORY_ACTION_TEMPLATES.get(op.category)
    if template is None:
        return ActionSpace(
            operator_name=op.name,
            category=op.category,
            allowed_precisions=[PrecisionLevel.FP32],
            prune_ratio_range=(0.0, 0.0),
            fuse_available=False,
            approx_rank_range=None,
        )

    allowed_precisions = list(template.allowed_precisions)
    prune_range = template.prune_ratio_range
    fuse_available = template.fuse_available
    approx_range = template.approx_rank_range

    if is_protected is None:
        protected = sensitivity is not None and sensitivity > sensitivity_threshold
    else:
        protected = is_protected

    if protected:
        # Highly sensitive: restrict to lighter compression
        allowed_precisions = [p for p in allowed_precisions if p <= PrecisionLevel.FP16]
        if not allowed_precisions:
            allowed_precisions = [PrecisionLevel.FP32]
        max_prune = min(prune_range[1], 0.3)
        prune_range = (prune_range[0], max_prune)

    if (
        param_share is not None
        and param_share < TINY_OP_BYTES_SHARE
        and op.category in (OperatorCategory.DENSE_COMPUTE, OperatorCategory.EMBEDDING)
    ):
        if protected:
            allowed_precisions = [PrecisionLevel.FP32]
        else:
            best = max(allowed_precisions)
            allowed_precisions = [best]
        prune_range = (0.0, 0.0)
        fuse_available = False
        approx_range = None

    return ActionSpace(
        operator_name=op.name,
        category=op.category,
        allowed_precisions=allowed_precisions,
        prune_ratio_range=prune_range,
        fuse_available=fuse_available,
        approx_rank_range=approx_range,
    )


@dataclass
class ConcreteAction:
    """A concrete compression action translated from abstract config for a specific operator type."""
    operator_name: str
    op_type: str
    category: OperatorCategory
    action_type: str
    parameters: dict[str, Any]


class ActionTranslator:
    """Translates abstract CompressionConfig into concrete operator-specific actions."""

    def translate(self, op: OperatorInfo, config: CompressionConfig) -> list[ConcreteAction]:
        if config.is_no_compression():
            return []

        category = op.category
        translators = {
            OperatorCategory.DENSE_COMPUTE: self._translate_neural,
            OperatorCategory.NORMALIZATION: self._translate_normalization,
            OperatorCategory.ACTIVATION: self._translate_activation,
            OperatorCategory.EMBEDDING: self._translate_embedding,
            OperatorCategory.TREE_ENSEMBLE: self._translate_tree,
            OperatorCategory.LINEAR_MODEL: self._translate_linear,
            OperatorCategory.SVM: self._translate_svm,
        }
        translator = translators.get(category)
        if translator is None:
            return []
        return translator(op, config)

    def _translate_neural(self, op: OperatorInfo, config: CompressionConfig) -> list[ConcreteAction]:
        actions = []
        if config.precision_level > PrecisionLevel.FP32:
            actions.append(ConcreteAction(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                action_type="quantization",
                parameters={
                    "target_dtype": config.precision_level.name,
                    "method": "static" if config.precision_level >= PrecisionLevel.INT8 else "cast",
                },
            ))
        if config.prune_ratio > 0:
            actions.append(ConcreteAction(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                action_type="channel_pruning",
                parameters={"ratio": config.prune_ratio},
            ))
        if config.fuse_enabled:
            actions.append(ConcreteAction(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                action_type="operator_fusion",
                parameters={},
            ))
        if config.approx_rank is not None:
            actions.append(ConcreteAction(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                action_type="low_rank_svd",
                parameters={"rank": config.approx_rank},
            ))
        return actions

    def _translate_normalization(self, op: OperatorInfo, config: CompressionConfig) -> list[ConcreteAction]:
        actions = []
        if config.fuse_enabled:
            actions.append(ConcreteAction(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                action_type="bn_fusion",
                parameters={},
            ))
        if config.precision_level > PrecisionLevel.FP32:
            actions.append(ConcreteAction(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                action_type="param_quantization",
                parameters={"target_dtype": config.precision_level.name},
            ))
        return actions

    def _translate_activation(self, op: OperatorInfo, config: CompressionConfig) -> list[ConcreteAction]:
        actions = []
        if config.fuse_enabled:
            actions.append(ConcreteAction(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                action_type="activation_fusion",
                parameters={},
            ))
        return actions

    def _translate_embedding(self, op: OperatorInfo, config: CompressionConfig) -> list[ConcreteAction]:
        actions = []
        if config.precision_level > PrecisionLevel.FP32:
            actions.append(ConcreteAction(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                action_type="embedding_quantization",
                parameters={"target_dtype": config.precision_level.name},
            ))
        if config.prune_ratio > 0:
            actions.append(ConcreteAction(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                action_type="vocabulary_pruning",
                parameters={"ratio": config.prune_ratio},
            ))
        if config.approx_rank is not None:
            actions.append(ConcreteAction(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                action_type="embedding_dimension_reduction",
                parameters={"rank": config.approx_rank},
            ))
        return actions

    def _translate_tree(self, op: OperatorInfo, config: CompressionConfig) -> list[ConcreteAction]:
        actions = []
        if config.precision_level > PrecisionLevel.FP32:
            actions.append(ConcreteAction(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                action_type="leaf_quantization",
                parameters={"target_dtype": config.precision_level.name},
            ))
        if config.prune_ratio > 0:
            actions.append(ConcreteAction(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                action_type="tree_pruning",
                parameters={"ratio": config.prune_ratio},
            ))
        if config.fuse_enabled:
            actions.append(ConcreteAction(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                action_type="leaf_merging",
                parameters={},
            ))
        if config.approx_rank is not None:
            actions.append(ConcreteAction(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                action_type="depth_limiting",
                parameters={"max_depth": config.approx_rank},
            ))
        return actions

    def _translate_linear(self, op: OperatorInfo, config: CompressionConfig) -> list[ConcreteAction]:
        actions = []
        if config.precision_level > PrecisionLevel.FP32:
            actions.append(ConcreteAction(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                action_type="coefficient_quantization",
                parameters={"target_dtype": config.precision_level.name},
            ))
        if config.prune_ratio > 0:
            actions.append(ConcreteAction(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                action_type="feature_pruning",
                parameters={"ratio": config.prune_ratio},
            ))
        return actions

    def _translate_svm(self, op: OperatorInfo, config: CompressionConfig) -> list[ConcreteAction]:
        actions = []
        if config.precision_level > PrecisionLevel.FP32:
            actions.append(ConcreteAction(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                action_type="sv_coefficient_quantization",
                parameters={"target_dtype": config.precision_level.name},
            ))
        if config.prune_ratio > 0:
            actions.append(ConcreteAction(
                operator_name=op.name,
                op_type=op.op_type,
                category=op.category,
                action_type="support_vector_pruning",
                parameters={"ratio": config.prune_ratio},
            ))
        return actions
