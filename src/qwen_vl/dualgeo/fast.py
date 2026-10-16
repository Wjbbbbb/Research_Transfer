"""The M1 VGGT-free spatial probe and calibration heads.

This module intentionally accepts only token features emitted by the fast Qwen
backbone.  It never imports, stores, or accepts a VGGT feature tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .config import FastAdapterConfig


@dataclass(frozen=True)
class FastRouteState:
    """Per-example summaries used by M1 probes and later M3 routing.

    ``risk_logit`` is deliberately uncalibrated; temperature fitting belongs on
    a held-out calibration split and is not learned inside this module.
    """

    question_state: torch.Tensor
    spatial_state: torch.Tensor
    risk_logit: torch.Tensor
    predicate_logits: torch.Tensor | None


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor | None, name: str) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, tokens, hidden]")
    if tokens.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one token")
    if mask is None:
        return tokens.mean(dim=1)
    if mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
        raise ValueError(f"{name}_mask must be bool with shape [batch, tokens]")
    weights = mask.unsqueeze(-1).to(tokens.dtype)
    count = weights.sum(dim=1)
    if torch.any(count == 0):
        raise ValueError(f"each {name} row must retain at least one token")
    return (tokens * weights).sum(dim=1) / count


class FastSpatialAdapter(nn.Module):
    """Question-conditioned visual pooling plus a small spatial residual MLP.

    It is a trainable probe over frozen Qwen features during M1.  Answer
    generation remains the SELF Qwen backbone; this module learns spatial and
    risk summaries without changing that backbone or instantiating VGGT.
    """

    def __init__(self, config: FastAdapterConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        bottleneck = config.bottleneck_size
        self.visual_norm = nn.LayerNorm(hidden)
        self.question_norm = nn.LayerNorm(hidden)
        self.residual_mlp = nn.Sequential(
            nn.Linear(hidden * 2, bottleneck),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(bottleneck, hidden),
        )
        self.residual_gate = nn.Linear(hidden * 2, hidden)
        self.risk_head = nn.Linear(hidden, 1)
        self.predicate_head = nn.Linear(hidden, config.num_predicates) if config.num_predicates else None

        # M1 begins at the frozen Qwen visual summary; the trainable spatial
        # residual is introduced smoothly rather than perturbing the probe at
        # initialization.
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)
        nn.init.zeros_(self.risk_head.weight)
        nn.init.zeros_(self.risk_head.bias)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        question_tokens: torch.Tensor,
        *,
        visual_mask: torch.Tensor | None = None,
        question_mask: torch.Tensor | None = None,
    ) -> FastRouteState:
        """Return one VGGT-free state per batch item.

        Args:
            visual_tokens: Frozen SELF-Qwen visual states ``[B, V, D]``.
            question_tokens: Frozen SELF-Qwen question states ``[B, Q, D]``.
            visual_mask/question_mask: ``True`` for valid tokens.
        """

        if visual_tokens.shape[0] != question_tokens.shape[0]:
            raise ValueError("visual_tokens and question_tokens must have the same batch size")
        if visual_tokens.shape[-1] != self.config.hidden_size or question_tokens.shape[-1] != self.config.hidden_size:
            raise ValueError("token hidden size does not match FastAdapterConfig.hidden_size")
        question_state = _masked_mean(self.question_norm(question_tokens), question_mask, "question_tokens")
        visual_states = self.visual_norm(visual_tokens)
        scores = torch.einsum("bvd,bd->bv", visual_states, question_state) / math.sqrt(self.config.hidden_size)
        if visual_mask is not None:
            if visual_mask.shape != visual_tokens.shape[:2] or visual_mask.dtype != torch.bool:
                raise ValueError("visual_mask must be bool with shape [batch, visual_tokens]")
            if torch.any(~visual_mask.any(dim=1)):
                raise ValueError("each visual row must retain at least one token")
            scores = scores.masked_fill(~visual_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=1)
        visual_state = torch.einsum("bv,bvd->bd", weights, visual_states)
        joined = torch.cat((visual_state, question_state), dim=-1)
        residual = self.residual_mlp(joined)
        spatial_state = visual_state + torch.sigmoid(self.residual_gate(joined)) * residual
        predicate_logits = self.predicate_head(spatial_state) if self.predicate_head is not None else None
        return FastRouteState(
            question_state=question_state,
            spatial_state=spatial_state,
            risk_logit=self.risk_head(spatial_state).squeeze(-1),
            predicate_logits=predicate_logits,
        )
