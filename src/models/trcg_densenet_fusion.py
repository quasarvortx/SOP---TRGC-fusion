"""Stage 8 DenseNet121 frozen-feature TRCG with protected EHR mortality anchor.

DenseNet CXR features: 1024 dimensions (original train_densenet checkpoint).
EHR vectors: 128 dimensions (TRCG Stage 2).
Stage 3 ResNet34 CXR alignment weights are intentionally NOT reused.
"""
import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ('dense_learned', 'dense_fixed', 'dense_age')

class DenseSafeTRCG(nn.Module):
    def __init__(self, ehr_head_state, cxr_head_state, variant, constant_gamma=0.3689):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.constant_gamma = float(constant_gamma)
        self.ehr_head = nn.Linear(128, 1)
        self.ehr_head.load_state_dict(ehr_head_state, strict=True)
        self.ehr_head.requires_grad_(False)
        self.cxr_head = nn.Linear(1024, 1)
        self.cxr_head.load_state_dict(cxr_head_state, strict=True)
        self.cxr_head.requires_grad_(False)

        self.ehr_proj = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 128))
        self.cxr_proj = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, 128), nn.GELU(), nn.Linear(128, 128))
        self.image_quality = nn.Sequential(nn.Linear(1024, 32), nn.GELU(), nn.Linear(32, 16))
        self.ehr_reliability = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 1))
        self.cxr_reliability = nn.Sequential(nn.Linear(18, 32), nn.ReLU(), nn.Linear(32, 1))
        # z_CXR + z_EHR_at_CXR + zero-clinical-change + concordance
        self.residual = nn.Sequential(nn.Linear(1024 + 128 + 128 + 512, 256),
                                      nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 128), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(512 + 128 + 4, 128), nn.ReLU(), nn.Linear(128, 1))
        nn.init.constant_(self.gate[-1].bias, -2.0)
        # Exact initial EHR anchor: no random CXR disruption.
        nn.init.zeros_(self.residual[-2].weight)
        nn.init.zeros_(self.residual[-2].bias)
        if variant == 'dense_fixed':
            self.gate.requires_grad_(False)

    def forward(self, b):
        zt, za, zc = b['ehr_at_48'], b['ehr_at_cxr'], b['cxr']
        age = (b['cxr_age'].reshape(-1, 1) / 48.).clamp(0, 1)
        valid = b['anchor_valid'].reshape(-1, 1).float()
        entropy = b['cxr_entropy'].reshape(-1, 1)
        qe = F.normalize(self.ehr_proj(za), dim=-1)
        qc = F.normalize(self.cxr_proj(zc), dim=-1)
        agreement = torch.cat((qc, qe, (qc - qe).abs(), qc * qe), dim=1)
        d = torch.zeros_like(zt)  # Stage 7 found learned change module weak.
        re = torch.sigmoid(self.ehr_reliability(b['ehr_quality']))
        rc = torch.sigmoid(self.cxr_reliability(
            torch.cat((age, entropy, self.image_quality(zc)), dim=1)))
        delta = self.residual(torch.cat((zc, za, d, agreement), dim=1))
        if self.variant == 'dense_fixed':
            gamma = torch.full_like(age, self.constant_gamma)
        else:
            gamma = torch.sigmoid(self.gate(torch.cat((agreement, d, age, re, rc, valid), dim=1)))
        effective = gamma * rc * valid
        fused = zt + effective * delta
        logits = self.ehr_head(fused).flatten()
        return {
            'logits': logits,
            'ehr_logits': self.ehr_head(zt).flatten(),
            'cxr_logits': self.cxr_head(zc).flatten(),
            'gamma': gamma.flatten(),
            'r_ehr': re.flatten(),
            'r_cxr': rc.flatten(),
            'effective': effective.flatten(),
            'delta': delta,
        }
