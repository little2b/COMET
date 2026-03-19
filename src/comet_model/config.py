from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


def _default_input_dims() -> Dict[str, int]:
    return {
        "gene": 200,
        "type": 768,
        "aa": 480,
        "cdna": 1552,
        "exon": 16,
        "vaf": 16,
    }


def _default_projection_dims() -> Dict[str, int]:
    return {
        "gene": 48,
        "type": 24,
        "aa": 48,
        "cdna": 32,
        "exon": 16,
        "vaf": 16,
    }


def _default_feature_toggles() -> Dict[str, bool]:
    return {
        "gene": True,
        "type": True,
        "aa": True,
        "cdna": True,
        "exon": True,
        "vaf": True,
    }


def _default_categorical_dims() -> Dict[str, int]:
    return {
        "sex": 2,
        "smoking": 2,
        "drinking": 2,
        "family_history": 2,
        "pathology": 5,
        "nodule_type": 5,
    }


@dataclass
class CometConfig:
    """
    Public configuration for the training-ready COMET architecture.

    The defaults mirror the current hybrid model implementation used in the
    main project, while keeping the package generic enough for custom datasets.
    """

    input_dims: Dict[str, int] = field(default_factory=_default_input_dims)
    projection_dims: Dict[str, int] = field(default_factory=_default_projection_dims)
    feature_toggles: Dict[str, bool] = field(default_factory=_default_feature_toggles)
    categorical_dims: Dict[str, int] = field(default_factory=_default_categorical_dims)

    num_numerical: int = 3
    n_variant_stats: int = 4
    use_variant_stats: bool = True

    variant_hidden_dim: int = 128
    variant_output_dim: int = 128
    clinical_hidden_dim: int = 64
    clinical_output_dim: int = 64
    fusion_output_dim: int = 64
    prediction_hidden_dim: int = 32

    num_attention_heads: int = 4
    num_aggregator_layers: int = 2
    dropout: float = 0.5
    projection_dropout: float = 0.3
    attention_dropout: float = 0.15

    def validate(self) -> None:
        expected_keys = set(_default_input_dims())
        if set(self.input_dims) != expected_keys:
            raise ValueError(f"input_dims keys must be {sorted(expected_keys)}")
        if set(self.projection_dims) != expected_keys:
            raise ValueError(f"projection_dims keys must be {sorted(expected_keys)}")
        if set(self.feature_toggles) != expected_keys:
            raise ValueError(f"feature_toggles keys must be {sorted(expected_keys)}")
        if self.variant_output_dim <= 0 or self.clinical_output_dim <= 0:
            raise ValueError("output dimensions must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.variant_output_dim % self.num_attention_heads != 0:
            raise ValueError(
                "variant_output_dim must be divisible by num_attention_heads "
                "for the variant aggregation transformer"
            )
        if self.fusion_output_dim % self.num_attention_heads != 0:
            raise ValueError(
                "fusion_output_dim must be divisible by num_attention_heads "
                "for bidirectional cross-attention"
            )

