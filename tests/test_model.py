from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from comet_model import COMETModel, CometConfig, validate_batch  # noqa: E402


def make_batch(config: CometConfig, batch_size: int = 2, max_variants: int = 7) -> dict:
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
    }


class CometModelTest(unittest.TestCase):
    def test_forward_shapes(self) -> None:
        config = CometConfig()
        batch = make_batch(config)
        validate_batch(
            config,
            high_dim_embeds=batch["high_dim_embeds"],
            variant_mask=batch["variant_mask"],
            numerical_features=batch["numerical_features"],
            categorical_features=batch["categorical_features"],
            variant_stats=batch["variant_stats"],
        )

        model = COMETModel(config)
        outputs = model(**batch)

        self.assertEqual(tuple(outputs.lnm_logits.shape), (2,))
        self.assertEqual(tuple(outputs.dm_logits.shape), (2,))
        self.assertEqual(tuple(outputs.lnm_probs.shape), (2,))
        self.assertEqual(tuple(outputs.dm_probs.shape), (2,))
        self.assertEqual(tuple(outputs.modality_gate.shape), (2, config.fusion_output_dim))
        self.assertEqual(tuple(outputs.cascade_gate.shape), (2,))


if __name__ == "__main__":
    unittest.main()
