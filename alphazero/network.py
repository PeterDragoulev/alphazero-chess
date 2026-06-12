"""
network.py — AlphaZero-style policy+value ResNet, plus a batched evaluator
for MCTS inference (fp16 on GPU).
"""

import numpy as np
import torch
import torch.nn as nn

from config import CFG
from encoding import planes_to_float


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.net(x))


class PolicyValueNet(nn.Module):
    """
    ~3.5M params at default 128ch x 10 blocks.

    forward(x) -> (policy_logits (B, 4672), value (B,) in (-1, 1))
    Value is from the perspective of the side to move.
    """

    def __init__(self, channels: int = CFG.channels, blocks: int = CFG.blocks):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(CFG.input_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 73, 1),
            nn.Flatten(),                       # (B, 73*64) = plane*64 + square
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        x = self.tower(self.stem(x))
        return self.policy_head(x), self.value_head(x).squeeze(-1)


class Evaluator:
    """Batched inference wrapper: uint8 planes in, numpy (logits, values) out."""

    def __init__(self, model: PolicyValueNet, device: torch.device):
        self.model = model
        self.device = device
        self.use_amp = device.type == "cuda"

    @torch.inference_mode()
    def __call__(self, planes_u8: np.ndarray):
        """planes_u8: (B, 19, 8, 8) uint8 → (logits (B,4672), values (B,)) f32."""
        x = torch.from_numpy(planes_to_float(planes_u8)).to(self.device)
        was_training = self.model.training
        self.model.eval()
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
            logits, values = self.model(x)
        if was_training:
            self.model.train()
        return logits.float().cpu().numpy(), values.float().cpu().numpy()


def load_checkpoint(path: str, device: torch.device) -> tuple[PolicyValueNet, dict]:
    """Build the net from a checkpoint; returns (model, full checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net_cfg = ckpt.get("net", {})
    model = PolicyValueNet(
        channels=net_cfg.get("channels", CFG.channels),
        blocks=net_cfg.get("blocks", CFG.blocks),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, ckpt
