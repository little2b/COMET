from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import CometConfig


@dataclass
class CometOutput:
    lnm_logits: torch.Tensor
    dm_logits: torch.Tensor
    lnm_probs: torch.Tensor
    dm_probs: torch.Tensor
    modality_gate: torch.Tensor
    cascade_gate: torch.Tensor


class EnhancedVariantAggregator(nn.Module):
    """Aggregate variant embeddings into a patient-level genomic representation."""

    def __init__(self, config: CometConfig) -> None:
        super().__init__()
        embed_dim = config.variant_output_dim
        self.use_variant_stats = config.use_variant_stats
        self.n_variant_stats = config.n_variant_stats

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=config.num_attention_heads,
                    dim_feedforward=embed_dim * 4,
                    dropout=config.attention_dropout,
                    batch_first=True,
                )
                for _ in range(config.num_aggregator_layers)
            ]
        )

        self.residual_proj = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

        if self.use_variant_stats:
            stats_per_feature_dim = 16
            total_stats_dim = self.n_variant_stats * stats_per_feature_dim
            self.tmb_encoder = self._make_stats_encoder(stats_per_feature_dim)
            self.n_genes_encoder = self._make_stats_encoder(stats_per_feature_dim)
            self.max_vaf_encoder = self._make_stats_encoder(stats_per_feature_dim)
            self.mean_vaf_encoder = self._make_stats_encoder(stats_per_feature_dim)
            self.stats_fusion = nn.Sequential(
                nn.Linear(embed_dim + total_stats_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU(),
                nn.Dropout(config.attention_dropout),
            )

    @staticmethod
    def _make_stats_encoder(output_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(
        self,
        variant_embeddings: torch.Tensor,
        variant_mask: Optional[torch.Tensor] = None,
        variant_stats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = self.residual_proj(variant_embeddings)
        encoded = variant_embeddings
        key_padding_mask = None
        if variant_mask is not None:
            key_padding_mask = variant_mask == 0

        for layer in self.layers:
            encoded = layer(encoded, src_key_padding_mask=key_padding_mask)

        encoded = self.layer_norm(encoded + residual)

        attn_scores = self.attention_pool(encoded).squeeze(-1)
        if variant_mask is not None:
            attn_scores = attn_scores.masked_fill(variant_mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)
        pooled = (encoded * attn_weights).sum(dim=1)

        if self.use_variant_stats and variant_stats is not None:
            tmb = variant_stats[:, 0:1]
            n_genes = variant_stats[:, 1:2]
            max_vaf = variant_stats[:, 2:3]
            mean_vaf = variant_stats[:, 3:4]
            stats_encoded = torch.cat(
                [
                    self.tmb_encoder(tmb),
                    self.n_genes_encoder(n_genes),
                    self.max_vaf_encoder(max_vaf),
                    self.mean_vaf_encoder(mean_vaf),
                ],
                dim=-1,
            )
            pooled = self.stats_fusion(torch.cat([pooled, stats_encoded], dim=-1))

        return pooled


class ClinicalEncoder(nn.Module):
    """Encode numerical and categorical clinical features into a compact vector."""

    def __init__(self, config: CometConfig) -> None:
        super().__init__()
        hidden_dim = config.clinical_hidden_dim
        embed_dim = config.clinical_output_dim

        self.numerical_encoder = nn.Sequential(
            nn.Linear(config.num_numerical, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

        self.categorical_embeddings = nn.ModuleDict()
        total_cat_dim = 0
        for name, num_classes in config.categorical_dims.items():
            emb_dim = min(num_classes * 2, 16)
            self.categorical_embeddings[name] = nn.Embedding(num_classes + 1, emb_dim)
            total_cat_dim += emb_dim

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + total_cat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        numerical_features: torch.Tensor,
        categorical_features: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        num_encoded = self.numerical_encoder(numerical_features)

        cat_embeds = []
        for name, embedding in self.categorical_embeddings.items():
            if name not in categorical_features:
                raise KeyError(f"Missing categorical feature: {name}")
            indices = categorical_features[name].long().view(-1)
            cat_embeds.append(embedding(indices))

        combined = torch.cat([num_encoded, *cat_embeds], dim=-1)
        return self.fusion(combined)


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional genomic-clinical cross-attention followed by gated fusion."""

    def __init__(self, config: CometConfig) -> None:
        super().__init__()
        output_dim = config.fusion_output_dim

        self.gene_proj = nn.Linear(config.variant_output_dim, output_dim)
        self.clinical_proj = nn.Linear(config.clinical_output_dim, output_dim)

        self.gene_to_clinical = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )
        self.clinical_to_gene = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            batch_first=True,
        )

        self.layer_norm_gene = nn.LayerNorm(output_dim)
        self.layer_norm_clinical = nn.LayerNorm(output_dim)
        self.gate = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(config.attention_dropout),
        )

    def forward(
        self,
        genomic_representation: torch.Tensor,
        clinical_representation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gene_proj = self.gene_proj(genomic_representation).unsqueeze(1)
        clinical_proj = self.clinical_proj(clinical_representation).unsqueeze(1)

        gene_attended, _ = self.gene_to_clinical(gene_proj, clinical_proj, clinical_proj)
        gene_enhanced = self.layer_norm_gene(gene_proj + gene_attended)

        clinical_attended, _ = self.clinical_to_gene(clinical_proj, gene_proj, gene_proj)
        clinical_enhanced = self.layer_norm_clinical(clinical_proj + clinical_attended)

        gene_enhanced = gene_enhanced.squeeze(1)
        clinical_enhanced = clinical_enhanced.squeeze(1)

        combined = torch.cat([gene_enhanced, clinical_enhanced], dim=-1)
        gate = self.gate(combined)
        fused = gate * gene_enhanced + (1.0 - gate) * clinical_enhanced
        final = self.fusion(torch.cat([fused, gene_enhanced], dim=-1))
        return final, gate


class CascadeMultiTaskHead(nn.Module):
    """Shared prediction head with cascade-gated transfer from LNM to DM."""

    def __init__(self, config: CometConfig) -> None:
        super().__init__()
        hidden_dim = config.prediction_hidden_dim
        input_dim = config.fusion_output_dim

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.lnm_feature = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.lnm_classifier = nn.Linear(hidden_dim // 2, 1)

        self.cascade_gate = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.dm_feature = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.dm_classifier = nn.Linear(hidden_dim // 2, 1)

    def forward(self, fused_representation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.shared(fused_representation)
        lnm_feature = self.lnm_feature(shared)
        lnm_logits = self.lnm_classifier(lnm_feature).squeeze(-1)

        cascade_gate = self.cascade_gate(lnm_feature)
        gated_lnm_feature = cascade_gate * lnm_feature

        dm_input = torch.cat([shared, gated_lnm_feature], dim=-1)
        dm_feature = self.dm_feature(dm_input)
        dm_logits = self.dm_classifier(dm_feature).squeeze(-1)
        return lnm_logits, dm_logits, cascade_gate.squeeze(-1)


class HybridProjectionEncoder(nn.Module):
    """Project heterogeneous variant feature embeddings into a shared variant space."""

    def __init__(self, config: CometConfig) -> None:
        super().__init__()
        self.input_dims = config.input_dims
        self.output_dims = config.projection_dims
        self.feature_toggles = config.feature_toggles
        dropout = config.projection_dropout

        self.gene_proj = nn.Linear(self.input_dims["gene"], self.output_dims["gene"])
        self.type_proj = nn.Linear(self.input_dims["type"], self.output_dims["type"])
        self.aa_proj = nn.Linear(self.input_dims["aa"], self.output_dims["aa"])
        self.cdna_proj = nn.Linear(self.input_dims["cdna"], self.output_dims["cdna"])
        self.exon_proj = nn.Linear(self.input_dims["exon"], self.output_dims["exon"])
        self.vaf_proj = nn.Linear(self.input_dims["vaf"], self.output_dims["vaf"])

        self.projection_dropout = nn.Dropout(dropout)
        total_dim = sum(self.output_dims.values())
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, config.variant_hidden_dim),
            nn.LayerNorm(config.variant_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.variant_hidden_dim, config.variant_output_dim),
        )

    def _project_or_zero(
        self,
        tensor: torch.Tensor,
        projection: nn.Linear,
        output_dim: int,
        enabled: bool,
    ) -> torch.Tensor:
        if enabled:
            return self.projection_dropout(projection(tensor))
        shape = tensor.shape[:-1] + (output_dim,)
        return torch.zeros(shape, device=tensor.device, dtype=tensor.dtype)

    def forward(self, high_dim_embeds: Mapping[str, torch.Tensor]) -> torch.Tensor:
        required = {"gene", "type", "aa", "cdna", "exon", "vaf"}
        missing = required.difference(high_dim_embeds)
        if missing:
            raise KeyError(f"Missing high-dimensional inputs: {sorted(missing)}")

        projections = [
            self._project_or_zero(
                high_dim_embeds["gene"],
                self.gene_proj,
                self.output_dims["gene"],
                self.feature_toggles["gene"],
            ),
            self._project_or_zero(
                high_dim_embeds["type"],
                self.type_proj,
                self.output_dims["type"],
                self.feature_toggles["type"],
            ),
            self._project_or_zero(
                high_dim_embeds["aa"],
                self.aa_proj,
                self.output_dims["aa"],
                self.feature_toggles["aa"],
            ),
            self._project_or_zero(
                high_dim_embeds["cdna"],
                self.cdna_proj,
                self.output_dims["cdna"],
                self.feature_toggles["cdna"],
            ),
            self._project_or_zero(
                high_dim_embeds["exon"],
                self.exon_proj,
                self.output_dims["exon"],
                self.feature_toggles["exon"],
            ),
            self._project_or_zero(
                high_dim_embeds["vaf"],
                self.vaf_proj,
                self.output_dims["vaf"],
                self.feature_toggles["vaf"],
            ),
        ]

        concat = torch.cat(projections, dim=-1)
        batch_size, max_variants, _ = concat.shape
        fused = self.fusion(concat.reshape(batch_size * max_variants, -1))
        return fused.reshape(batch_size, max_variants, -1)


class COMETModel(nn.Module):
    """
    Training-ready public COMET architecture.

    The model expects precomputed per-variant feature embeddings plus clinical
    features and genomic summary statistics. This matches the current hybrid
    formulation of the internal project while removing project-specific I/O.
    """

    def __init__(self, config: Optional[CometConfig] = None) -> None:
        super().__init__()
        self.config = config or CometConfig()
        self.config.validate()

        self.projection_encoder = HybridProjectionEncoder(self.config)
        self.variant_aggregator = EnhancedVariantAggregator(self.config)
        self.clinical_encoder = ClinicalEncoder(self.config)
        self.cross_modal_fusion = BidirectionalCrossAttention(self.config)
        self.prediction_head = CascadeMultiTaskHead(self.config)

    def forward(
        self,
        high_dim_embeds: Mapping[str, torch.Tensor],
        variant_mask: torch.Tensor,
        numerical_features: torch.Tensor,
        categorical_features: Mapping[str, torch.Tensor],
        variant_stats: Optional[torch.Tensor] = None,
    ) -> CometOutput:
        variant_embeddings = self.projection_encoder(high_dim_embeds)
        genomic_representation = self.variant_aggregator(
            variant_embeddings=variant_embeddings,
            variant_mask=variant_mask,
            variant_stats=variant_stats,
        )
        clinical_representation = self.clinical_encoder(
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )
        fused_representation, modality_gate = self.cross_modal_fusion(
            genomic_representation=genomic_representation,
            clinical_representation=clinical_representation,
        )
        lnm_logits, dm_logits, cascade_gate = self.prediction_head(fused_representation)
        return CometOutput(
            lnm_logits=lnm_logits,
            dm_logits=dm_logits,
            lnm_probs=torch.sigmoid(lnm_logits),
            dm_probs=torch.sigmoid(dm_logits),
            modality_gate=modality_gate,
            cascade_gate=cascade_gate,
        )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

