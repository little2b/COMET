from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comet_model import CometConfig, validate_batch  # noqa: E402


class CustomCometDataset(Dataset):
    """
    Template dataset for training COMET on custom data.

    Replace the synthetic tensors below with your own preprocessed features.
    """

    def __init__(self, num_samples: int = 32, max_variants: int = 50, config: CometConfig | None = None) -> None:
        self.num_samples = num_samples
        self.max_variants = max_variants
        self.config = config or CometConfig()

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> Dict[str, object]:
        cfg = self.config
        return {
            "high_dim_embeds": {
                "gene": torch.randn(self.max_variants, cfg.input_dims["gene"]),
                "type": torch.randn(self.max_variants, cfg.input_dims["type"]),
                "aa": torch.randn(self.max_variants, cfg.input_dims["aa"]),
                "cdna": torch.randn(self.max_variants, cfg.input_dims["cdna"]),
                "exon": torch.randn(self.max_variants, cfg.input_dims["exon"]),
                "vaf": torch.randn(self.max_variants, cfg.input_dims["vaf"]),
            },
            "variant_mask": torch.ones(self.max_variants),
            "variant_stats": torch.randn(cfg.n_variant_stats),
            "numerical_features": torch.randn(cfg.num_numerical),
            "categorical_features": {
                name: torch.randint(0, num_classes, (1,))
                for name, num_classes in cfg.categorical_dims.items()
            },
            "lnm_target": torch.randint(0, 2, (1,), dtype=torch.float32),
            "dm_target": torch.randint(0, 2, (1,), dtype=torch.float32),
        }


def collate_comet_batch(samples: list[Dict[str, object]]) -> Dict[str, object]:
    return {
        "high_dim_embeds": {
            key: torch.stack([sample["high_dim_embeds"][key] for sample in samples])
            for key in samples[0]["high_dim_embeds"]
        },
        "variant_mask": torch.stack([sample["variant_mask"] for sample in samples]),
        "variant_stats": torch.stack([sample["variant_stats"] for sample in samples]),
        "numerical_features": torch.stack([sample["numerical_features"] for sample in samples]),
        "categorical_features": {
            key: torch.cat([sample["categorical_features"][key] for sample in samples], dim=0)
            for key in samples[0]["categorical_features"]
        },
        "lnm_target": torch.cat([sample["lnm_target"] for sample in samples], dim=0),
        "dm_target": torch.cat([sample["dm_target"] for sample in samples], dim=0),
    }


def main() -> None:
    config = CometConfig()
    dataset = CustomCometDataset(config=config)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_comet_batch)
    batch = next(iter(loader))
    validate_batch(
        config,
        high_dim_embeds=batch["high_dim_embeds"],
        variant_mask=batch["variant_mask"],
        numerical_features=batch["numerical_features"],
        categorical_features=batch["categorical_features"],
        variant_stats=batch["variant_stats"],
    )
    print("Custom dataset template batch is valid.")


if __name__ == "__main__":
    main()
