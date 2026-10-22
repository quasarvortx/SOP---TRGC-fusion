"""TRCG Stage-4A frozen-feature gated residual fusion head.

EHR/CXR encoders remain frozen; Stage-2 pooled features are read from Stage-3 caches.
This module implements reliability, temporal concordance, clinical change, and
complementarity-gated residual fusion. Encoder fine-tuning is Stage-4B (later).
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class AlignmentProjectors(nn.Module):
    """Matches Stage-3 AlignmentProjectors state_dict keys exactly."""
    def __init__(self):
        super().__init__()
        self.ehr = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 128),
                                 nn.GELU(), nn.Linear(128, 128))
        self.cxr = nn.Sequential(nn.LayerNorm(512), nn.Linear(512, 128),
                                 nn.GELU(), nn.Linear(128, 128))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, ehr, cxr):
        return (F.normalize(self.ehr(ehr), dim=-1),
                F.normalize(self.cxr(cxr), dim=-1))


class TRCGFusion(nn.Module):
    def __init__(self, ehr_head_state, cxr_head_state, align_state=None):
        super().__init__()
        self.projections = AlignmentProjectors()
        if align_state is not None:
            self.projections.load_state_dict(align_state, strict=True)

        self.ehr_head = nn.Linear(128, 1)
        self.cxr_head = nn.Linear(512, 1)
        self.ehr_head.load_state_dict(ehr_head_state, strict=True)
        self.cxr_head.load_state_dict(cxr_head_state, strict=True)

        self.change = nn.Sequential(nn.Linear(128 * 3, 128), nn.LayerNorm(128), nn.GELU())
        self.image_quality = nn.Sequential(nn.Linear(512, 32), nn.GELU(), nn.Linear(32, 16))
        self.ehr_reliability = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 1))
        self.cxr_reliability = nn.Sequential(nn.Linear(16 + 2, 32), nn.ReLU(), nn.Linear(32, 1))
        concordance_dim = 128 * 4
        self.residual = nn.Sequential(nn.Linear(512 + 128 + 128 + concordance_dim, 256),
                                      nn.LayerNorm(256), nn.GELU(), nn.Linear(256, 128), nn.Tanh())
        self.gate = nn.Sequential(nn.Linear(concordance_dim + 128 + 4, 128),
                                  nn.ReLU(), nn.Linear(128, 1))
        # Start close to existing EHR model, not a random large CXR correction.
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, b):
        zt = b['ehr_at_48']             # [B,128]
        za = b['ehr_at_cxr']            # [B,128] causal anchor
        zc = b['cxr']                   # [B,512]
        age = (b['cxr_age'].view(-1, 1) / 48.0).clamp(0, 1)
        valid = b['anchor_valid'].float().view(-1, 1)
        entropy = b['cxr_entropy'].view(-1, 1)

        qe, qc = self.projections(za, zc)
        cp = torch.cat((qc, qe, (qc - qe).abs(), qc * qe), dim=-1)
        d = self.change(torch.cat((zt, za, zt - za), dim=-1))
        re = torch.sigmoid(self.ehr_reliability(b['ehr_quality']))
        rc = torch.sigmoid(self.cxr_reliability(
            torch.cat((age, entropy, self.image_quality(zc)), dim=-1)))
        delta = self.residual(torch.cat((zc, za, d, cp), dim=-1))
        gate_inputs = torch.cat((cp, d, age, re, rc, valid), dim=-1)
        gamma = torch.sigmoid(self.gate(gate_inputs))
        fused = zt + gamma * rc * delta

        return {
            'logits': self.ehr_head(fused).squeeze(-1),
            'ehr_logits': self.ehr_head(zt).squeeze(-1),
            'cxr_logits': self.cxr_head(zc).squeeze(-1),
            'gamma': gamma.squeeze(-1),
            'r_ehr': re.squeeze(-1),
            'r_cxr': rc.squeeze(-1),
            'q_ehr': qe, 'q_cxr': qc,
            'change': d,
        }
