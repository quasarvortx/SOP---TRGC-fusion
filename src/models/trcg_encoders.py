
"""Reusable encoders for TRCG Stage 2 and later fusion."""

import math
import torch
from torch import nn
from torchvision import models


class TimeAwareEHR(nn.Module):
    def __init__(
        self,
        categorical_vocab,
        d_model=128,
        nhead=4,
        layers=2,
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model

        # Each clinical variable has its own learned representation.
        self.value_basis = nn.Parameter(
            torch.randn(17, d_model) * 0.02
        )
        self.mask_basis = nn.Parameter(
            torch.randn(17, d_model) * 0.02
        )
        self.delta_basis = nn.Parameter(
            torch.randn(17, d_model) * 0.02
        )

        self.cat_indices = {
            "Capillary refill rate": 0,
            "Glascow coma scale eye opening": 3,
            "Glascow coma scale motor response": 4,
            "Glascow coma scale verbal response": 6,
        }

        self.cat_embeddings = nn.ModuleDict({
            name: nn.Embedding(
                len(categorical_vocab.get(name, {})) + 1,
                d_model,
                padding_idx=0,
            )
            for name in self.cat_indices
        })

        self.in_norm = nn.LayerNorm(d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=layers,
        )

        self.out_norm = nn.LayerNorm(d_model)

    def time_embedding(self, hours):
        half = self.d_model // 2

        exponents = torch.arange(
            half,
            device=hours.device,
            dtype=hours.dtype,
        )

        freqs = torch.exp(
            -math.log(10000.0)
            * exponents
            / max(half - 1, 1)
        )

        angle = hours.unsqueeze(-1) * freqs

        return torch.cat(
            [angle.sin(), angle.cos()],
            dim=-1,
        )

    def forward(self, batch, return_sequence=False):
        x = torch.einsum(
            "btv,vd->btd",
            batch["values"],
            self.value_basis,
        )

        x = x + torch.einsum(
            "btv,vd->btd",
            batch["mask"],
            self.mask_basis,
        )

        x = x + torch.einsum(
            "btv,vd->btd",
            batch["delta"],
            self.delta_basis,
        )

        for name, idx in self.cat_indices.items():
            x = x + self.cat_embeddings[name](
                batch["categories"][..., idx]
            )

        x = self.in_norm(
            x + self.time_embedding(batch["times"])
        )

        seq_len = x.shape[1]

        # Critical: an earlier EHR state must not see
        # observations that happened later.
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        lengths = batch["lengths"]

        padding_mask = (
            torch.arange(
                seq_len,
                device=x.device,
            )[None, :]
            >= lengths[:, None]
        )

        h = self.out_norm(
            self.encoder(
                x,
                mask=causal_mask,
                src_key_padding_mask=padding_mask,
            )
        )

        final = h[
            torch.arange(
                len(lengths),
                device=x.device,
            ),
            lengths - 1,
        ]

        if return_sequence:
            return final, h

        return final


class EHRMortality(nn.Module):
    def __init__(self, categorical_vocab):
        super().__init__()

        self.encoder = TimeAwareEHR(
            categorical_vocab
        )

        self.head = nn.Linear(128, 1)

    def forward(self, batch):
        features = self.encoder(batch)
        return self.head(features).squeeze(-1)


class CXRMortality(nn.Module):
    def __init__(self, imagenet_checkpoint):
        super().__init__()

        net = models.resnet34(weights=None)

        if imagenet_checkpoint is not None:
            weights = torch.load(
                imagenet_checkpoint,
                map_location="cpu",
                weights_only=True,
            )
            net.load_state_dict(weights)

        net.fc = nn.Identity()

        self.encoder = net
        self.head = nn.Linear(512, 1)

    def forward(self, image):
        features = self.encoder(image)
        return self.head(features).squeeze(-1)
