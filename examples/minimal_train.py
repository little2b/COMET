from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comet_model import COMETModel, CometConfig, count_parameters  # noqa: E402


def make_dummy_batch(config: CometConfig, batch_size: int = 8, max_variants: int = 50) -> dict:
    return {
        "high_dim_embeds": {
            "gene": torch.randn(batch_size, max_variants, config.input_dims["gene"]),
            "type": torch.randn(batch_size, max_variants, config.input_dims["type"]),
            "aa": torch.randn(batch_size, max_variants, config.input_dims["aa"]),
            "cdna": torch.randn(batch_size, max_variants, config.input_dims["cdna"]),
            "exon": torch.randn(batch_size, max_variants, config.input_dims["exon"]),
            "vaf": torch.randn(batch_size, max_variants, config.input_dims["vaf"]),
        },
        "variant_mask": torch.ones(batch_size, max_variants),
        "variant_stats": torch.randn(batch_size, config.n_variant_stats),
        "numerical_features": torch.randn(batch_size, config.num_numerical),
        "categorical_features": {
            name: torch.randint(0, num_classes, (batch_size,))
            for name, num_classes in config.categorical_dims.items()
        },
        "lnm_targets": torch.randint(0, 2, (batch_size,), dtype=torch.float32),
        "dm_targets": torch.randint(0, 2, (batch_size,), dtype=torch.float32),
    }


def main() -> None:
    config = CometConfig()
    model = COMETModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    batch = make_dummy_batch(config)
    outputs = model(
        high_dim_embeds=batch["high_dim_embeds"],
        variant_mask=batch["variant_mask"],
        numerical_features=batch["numerical_features"],
        categorical_features=batch["categorical_features"],
        variant_stats=batch["variant_stats"],
    )

    lnm_loss = F.binary_cross_entropy_with_logits(outputs.lnm_logits, batch["lnm_targets"])
    dm_loss = F.binary_cross_entropy_with_logits(outputs.dm_logits, batch["dm_targets"])
    loss = lnm_loss + dm_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("COMETModel smoke-train step completed.")
    print(f"Trainable parameters: {count_parameters(model):,}")
    print(f"Loss: {loss.item():.4f}")
    print(f"LNM logits shape: {tuple(outputs.lnm_logits.shape)}")
    print(f"DM logits shape: {tuple(outputs.dm_logits.shape)}")


if __name__ == "__main__":
    main()
