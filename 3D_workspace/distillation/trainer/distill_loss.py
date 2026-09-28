"""Knowledge-distillation loss: temperature-scaled KL between student and teacher logits.

Caching raw logits (not softmax probabilities) lets the temperature be chosen at train time.
Masked to valid voxels (border padding is -1). Scaled by T^2 so gradient magnitude stays roughly
independent of T (standard KD scaling).

Installed flat into the venv `variants/` dir (see scripts/02_train/01_install_trainer.sh).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class KDLoss(nn.Module):
    def __init__(self, temperature: float = 2.0):
        super().__init__()
        self.T = float(temperature)

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                seg_target: torch.Tensor) -> torch.Tensor:
        log_s = F.log_softmax(student_logits / self.T, dim=1)
        soft_t = F.softmax(teacher_logits / self.T, dim=1)
        kl = F.kl_div(log_s, soft_t, reduction='none').sum(1)  # per-voxel KL, (B, *spatial)

        tgt = seg_target[:, 0] if seg_target.ndim == student_logits.ndim else seg_target
        valid = (tgt >= 0).to(kl.dtype)  # exclude border padding
        return (kl * valid).sum() / valid.sum().clamp(min=1.0) * (self.T ** 2)


class DeepSupervisionDistillation(nn.Module):
    """Apply a KD loss across deep-supervision scales, weighted the same way as the
    supervised loss so the two terms are comparable scale-for-scale."""
    def __init__(self, kd_loss: nn.Module, weight_factors=None):
        super().__init__()
        self.kd = kd_loss
        self.weight_factors = weight_factors

    def forward(self, student_outputs, teacher_targets, seg_targets) -> torch.Tensor:
        if not isinstance(student_outputs, (list, tuple)):
            return self.kd(student_outputs, teacher_targets, seg_targets)

        weights = self.weight_factors or [1.0] * len(student_outputs)
        total = torch.zeros((), device=student_outputs[0].device)
        for weight, student, teacher, target in zip(weights, student_outputs, teacher_targets, seg_targets):
            if weight != 0:
                total = total + weight * self.kd(student, teacher, target)
        return total
