from __future__ import annotations

from typing import Mapping, MutableMapping, Optional

import torch

from .config import CometConfig


REQUIRED_EMBED_KEYS = ("gene", "type", "aa", "cdna", "exon", "vaf")


def validate_batch(
    config: CometConfig,
    *,
    high_dim_embeds: Mapping[str, torch.Tensor],
    variant_mask: torch.Tensor,
    numerical_features: torch.Tensor,
    categorical_features: Mapping[str, torch.Tensor],
    variant_stats: Optional[torch.Tensor] = None,
) -> None:
    """Validate that a batch matches the public COMET input contract."""

    for key in REQUIRED_EMBED_KEYS:
        if key not in high_dim_embeds:
            raise KeyError(f"Missing high-dimensional embedding: {key}")

    batch_size = None
    max_variants = None
    for key in REQUIRED_EMBED_KEYS:
        tensor = high_dim_embeds[key]
        if tensor.ndim != 3:
            raise ValueError(f'{key} must have shape [B, M, D], got {tuple(tensor.shape)}')
        expected_dim = config.input_dims[key]
        if tensor.shape[-1] != expected_dim:
            raise ValueError(f'{key} last dimension must be {expected_dim}, got {tensor.shape[-1]}')
        if batch_size is None:
            batch_size, max_variants = tensor.shape[:2]
        elif tensor.shape[:2] != (batch_size, max_variants):
            raise ValueError(f"{key} batch/max_variants dimensions do not match the other inputs")

    if variant_mask.ndim != 2 or tuple(variant_mask.shape) != (batch_size, max_variants):
        raise ValueError(
            f"variant_mask must have shape {(batch_size, max_variants)}, got {tuple(variant_mask.shape)}"
        )

    if numerical_features.ndim != 2 or numerical_features.shape != (batch_size, config.num_numerical):
        raise ValueError(
            f"numerical_features must have shape {(batch_size, config.num_numerical)}, "
            f"got {tuple(numerical_features.shape)}"
        )

    expected_categorical = set(config.categorical_dims)
    provided_categorical = set(categorical_features)
    if expected_categorical != provided_categorical:
        raise KeyError(
            f"categorical_features keys must be {sorted(expected_categorical)}, "
            f"got {sorted(provided_categorical)}"
        )

    for name, num_classes in config.categorical_dims.items():
        tensor = categorical_features[name]
        if tensor.ndim not in (1, 2):
            raise ValueError(f"{name} must have shape [B] or [B, 1], got {tuple(tensor.shape)}")
        if tensor.ndim == 2 and tensor.shape[1] != 1:
            raise ValueError(f"{name} second dimension must be 1 when using rank-2 tensors")
        if tensor.shape[0] != batch_size:
            raise ValueError(f"{name} batch dimension must be {batch_size}, got {tensor.shape[0]}")
        if torch.any(tensor < 0) or torch.any(tensor >= num_classes):
            raise ValueError(f"{name} indices must be in [0, {num_classes - 1}]")

    if config.use_variant_stats:
        if variant_stats is None:
            raise ValueError("variant_stats is required when use_variant_stats=True")
        if variant_stats.ndim != 2 or variant_stats.shape != (batch_size, config.n_variant_stats):
            raise ValueError(
                f"variant_stats must have shape {(batch_size, config.n_variant_stats)}, "
                f"got {tuple(variant_stats.shape)}"
            )


def move_batch_to_device(batch: MutableMapping[str, object], device: torch.device | str) -> MutableMapping[str, object]:
    """Move a nested COMET batch dictionary to a target device."""

    moved: MutableMapping[str, object] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        elif isinstance(value, dict):
            moved[key] = {
                inner_key: inner_value.to(device) if isinstance(inner_value, torch.Tensor) else inner_value
                for inner_key, inner_value in value.items()
            }
        else:
            moved[key] = value
    return moved
