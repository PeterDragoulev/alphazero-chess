"""
network.py — AlphaZero-style policy+value ResNet, plus a batched evaluator
for MCTS inference (fp16 on GPU).
"""

import numpy as np
import torch
import torch.nn as nn

from config import CFG
from encoding import HALFMOVE_PLANE, planes_to_float


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
    """
    Batched inference wrapper: uint8 planes in, numpy (logits, values) out.

    On CUDA every call replays a captured CUDA graph. At MCTS batch sizes the
    GPU work is tiny and per-kernel launch overhead dominates (~5.6 ms per
    call at batch 1 on the 3070 Ti under WSL2, ~0.8 ms as a graph replay).
    Batches are padded up to a power-of-two bucket with one graph per bucket,
    captured lazily. The graph reads the model's parameter and BatchNorm
    buffers in place, so it keeps up with training (optimizer updates are
    in-place) and always runs in eval mode, whatever model.training says.
    """

    MAX_GRAPH_BATCH = 1024

    def __init__(self, model: PolicyValueNet, device: torch.device,
                 use_graphs: bool = True):
        self.model = model
        self.device = device
        self.use_amp = device.type == "cuda"
        self.use_graphs = use_graphs and device.type == "cuda"
        self._graphs = {}            # bucket -> (graph, in_u8, logits, values)
        self._pool = None

    @torch.inference_mode()
    def __call__(self, planes_u8: np.ndarray):
        """planes_u8: (B, 19, 8, 8) uint8 → (logits (B,4672), values (B,)) f32."""
        b = len(planes_u8)
        if self.use_graphs and b <= self.MAX_GRAPH_BATCH:
            graph, x_in, logits, values = self._graph(1 << (b - 1).bit_length())
            x_in[:b].copy_(torch.from_numpy(planes_u8))
            graph.replay()
            return (logits[:b].float().cpu().numpy(),
                    values[:b].float().cpu().numpy())

        x = torch.from_numpy(planes_to_float(planes_u8)).to(self.device)
        was_training = self.model.training
        self.model.eval()
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
            logits, values = self.model(x)
        if was_training:
            self.model.train()
        return logits.float().cpu().numpy(), values.float().cpu().numpy()

    def _forward(self, x_u8: torch.Tensor):
        x = x_u8.float()
        x[:, HALFMOVE_PLANE] /= 100.0
        # cache_enabled=False: re-cast the fp32 weights on every replay, so
        # the graph sees weight updates instead of a stale fp16 copy.
        with torch.autocast("cuda", dtype=torch.float16, cache_enabled=False):
            return self.model(x)

    def _graph(self, bucket: int):
        if bucket in self._graphs:
            return self._graphs[bucket]
        was_training = self.model.training
        self.model.eval()
        x_in = torch.zeros((bucket, CFG.input_planes, 8, 8),
                           dtype=torch.uint8, device=self.device)
        side = torch.cuda.Stream()               # warm up off the main stream
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._forward(x_in)
        torch.cuda.current_stream().wait_stream(side)
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._pool):
            logits, values = self._forward(x_in)
        if was_training:
            self.model.train()
        self._graphs[bucket] = (graph, x_in, logits, values)
        return self._graphs[bucket]


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
