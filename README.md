# COMET Model

Training-ready PyTorch implementation of the COMET architecture for joint prediction of lymph node metastasis and distant metastasis from genomic and clinical inputs.

This repository is intended for researchers who want to train the architecture on their own data. It contains the model definition, configuration, input validation helpers, and runnable examples. It does not contain internal data, trained weights, deployment code, or project-specific pipelines.

## Web inference demo

A publicly accessible web deployment of COMET is available at:

`https://comet.nlhcqmu.tech/`

This online interface can be used for direct inference and demonstration purposes, whereas this GitHub repository is intended for researchers who want to train or adapt the model on their own datasets.

## Included modules

- Variant-level projection encoder
- Enhanced variant aggregation module
- Clinical encoder
- Bidirectional cross-modal fusion
- Cascade multitask prediction head
- Input validation helpers
- Minimal synthetic-data training example
- Custom dataset template
- Unit test and GitHub Actions smoke test

## Repository layout

```text
comet-model/
├── .github/workflows/ci.yml
├── CONTRIBUTING.md
├── CITATION.cff
├── pyproject.toml
├── requirements.txt
├── README.md
├── src/
│   └── comet_model/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       └── model.py
├── examples/
│   ├── custom_dataset_template.py
│   └── minimal_train.py
└── tests/
    └── test_model.py
```

## Installation

```bash
pip install -e .
```

Or:

```bash
pip install -r requirements.txt
```

## What you need to prepare

This public package assumes that you already have per-variant feature tensors ready for training. The repository focuses on the COMET architecture itself, not on rebuilding every upstream embedding pipeline.

Each batch must provide:

| Input                                      | Type             | Shape                 | Notes                                            |
| ------------------------------------------ | ---------------- | --------------------- | ------------------------------------------------ |
| `high_dim_embeds["gene"]`                | `torch.Tensor` | `[B, M, 200]`       | Per-variant gene embedding                       |
| `high_dim_embeds["type"]`                | `torch.Tensor` | `[B, M, 768]`       | Per-variant mutation-type embedding              |
| `high_dim_embeds["aa"]`                  | `torch.Tensor` | `[B, M, 480]`       | Per-variant amino-acid embedding                 |
| `high_dim_embeds["cdna"]`                | `torch.Tensor` | `[B, M, 1552]`      | Per-variant cDNA embedding                       |
| `high_dim_embeds["exon"]`                | `torch.Tensor` | `[B, M, 16]`        | Per-variant exon feature                         |
| `high_dim_embeds["vaf"]`                 | `torch.Tensor` | `[B, M, 16]`        | Per-variant VAF feature                          |
| `variant_mask`                           | `torch.Tensor` | `[B, M]`            | `1` for valid variants and `0` for padding   |
| `variant_stats`                          | `torch.Tensor` | `[B, 4]`            | `[TMB, mutated_gene_count, max_vaf, mean_vaf]` |
| `numerical_features`                     | `torch.Tensor` | `[B, 3]`            | Numerical clinical variables                     |
| `categorical_features["sex"]`            | `torch.Tensor` | `[B]` or `[B, 1]` | Integer indices                                  |
| `categorical_features["smoking"]`        | `torch.Tensor` | `[B]` or `[B, 1]` | Integer indices                                  |
| `categorical_features["drinking"]`       | `torch.Tensor` | `[B]` or `[B, 1]` | Integer indices                                  |
| `categorical_features["family_history"]` | `torch.Tensor` | `[B]` or `[B, 1]` | Integer indices                                  |
| `categorical_features["pathology"]`      | `torch.Tensor` | `[B]` or `[B, 1]` | Integer indices                                  |
| `categorical_features["nodule_type"]`    | `torch.Tensor` | `[B]` or `[B, 1]` | Integer indices                                  |

Default categorical cardinalities:

```python
{
    "sex": 2,
    "smoking": 2,
    "drinking": 2,
    "family_history": 2,
    "pathology": 3,
    "nodule_type": 4,
}
```

You can override these values in `CometConfig`.

## Quick start

```python
import torch
from comet_model import COMETModel, CometConfig

config = CometConfig()
model = COMETModel(config)

batch_size = 4
max_variants = 50

high_dim_embeds = {
    "gene": torch.randn(batch_size, max_variants, 200),
    "type": torch.randn(batch_size, max_variants, 768),
    "aa": torch.randn(batch_size, max_variants, 480),
    "cdna": torch.randn(batch_size, max_variants, 1552),
    "exon": torch.randn(batch_size, max_variants, 16),
    "vaf": torch.randn(batch_size, max_variants, 16),
}

variant_mask = torch.ones(batch_size, max_variants)
variant_stats = torch.randn(batch_size, 4)
numerical_features = torch.randn(batch_size, 3)
categorical_features = {
    "sex": torch.randint(0, 2, (batch_size,)),
    "smoking": torch.randint(0, 2, (batch_size,)),
    "drinking": torch.randint(0, 2, (batch_size,)),
    "family_history": torch.randint(0, 2, (batch_size,)),
    "pathology": torch.randint(0, 5, (batch_size,)),
    "nodule_type": torch.randint(0, 5, (batch_size,)),
}

outputs = model(
    high_dim_embeds=high_dim_embeds,
    variant_mask=variant_mask,
    numerical_features=numerical_features,
    categorical_features=categorical_features,
    variant_stats=variant_stats,
)

print(outputs.lnm_logits.shape, outputs.dm_logits.shape)
```

## Training on your own data

The package does not enforce a specific file format. You only need to convert your dataset into batches that match the input contract above.

Recommended training setup:

- Use `BCEWithLogitsLoss` for both tasks
- Optimize `lnm_logits` and `dm_logits`
- Mask padded variants with `variant_mask`
- Keep `variant_stats` in the exact order `[TMB, mutated_gene_count, max_vaf, mean_vaf]`
- Validate the first batch before launching a long run

Examples:

- [`examples/minimal_train.py`](examples/minimal_train.py): one synthetic optimization step
- [`examples/custom_dataset_template.py`](examples/custom_dataset_template.py): dataset and collate template for your own data

## Public API

- `CometConfig`
- `CometOutput`
- `COMETModel`
- `validate_batch`
- `move_batch_to_device`
- `count_parameters`

## Reproducibility and publication notes

- `CITATION.cff` is included so GitHub can display citation metadata
- A CI workflow is included to run a smoke test on every push
- The repository is released under the MIT License

## Citation

If you use this repository in academic work, please cite the associated COMET manuscript. GitHub will also expose the citation metadata from [`CITATION.cff`](CITATION.cff).
