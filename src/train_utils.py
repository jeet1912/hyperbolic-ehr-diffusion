# train_utils.py
import torch
import torch.nn.functional as F
import math


def get_cosine_temperature_schedule(total_steps: int, final_temp: float = 0.05):
    """Cosine annealing from 1.0 → final_temp over total_steps."""
    def temperature(step):
        if step >= total_steps:
            return final_temp
        progress = step / total_steps
        return final_temp + (1.0 - final_temp) * 0.5 * (1 + math.cos(math.pi * progress))
    return temperature


def apply_frequency_bias(logits: torch.Tensor, code_freq: torch.Tensor, bias_strength: float = 0.1):
    """
    logits:     [..., V]
    code_freq:  [V]  (raw counts or smoothed frequencies)
    """
    bias = bias_strength * torch.log(code_freq + 1.0)   # log-frequency bias
    return logits + bias.to(logits.device)