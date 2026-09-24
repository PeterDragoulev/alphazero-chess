"""
network.py — AlphaZero-style policy+value ResNet, plus a batched evaluator
for MCTS inference (fp16 on GPU).
"""

import numpy as np
import torch
import torch.nn as nn

from config import CFG
from encoding import HALFMOVE_PLANE, planes_to_float


class SqueezeExcite(nn.Module):
    """
    Squeeze-and-excitation (Lc0 style, scale + bias): average each channel
    over the 64 squares, pass the channel summary through a small MLP, and
    use its output to rescale and shift every channel on every square — so a
    block can turn whole feature maps up or down from global context (king
    safety, game phase) that 3x3 convolutions only see slowly.

    The gate is 2*sigmoid(z) and the last layer starts at zero, so a fresh
    SE is an exact identity (gate 1.0, bias 0.0): adding it to a trained net
    changes nothing until fine-tuning teaches it something.
    """

    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // ratio)
        self.fc2 = nn.Linear(channels // ratio, 2 * channels)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        z = self.fc2(torch.relu(self.fc1(x.mean(dim=(2, 3)))))
        gate, bias = z.unsqueeze(-1).unsqueeze(-1).chunk(2, dim=1)
        return x * (2 * torch.sigmoid(gate)) + bias


class ResBlock(nn.Module):
    def __init__(self, channels: int, se: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.se = SqueezeExcite(channels) if se else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.net(x)
        if self.se is not None:
            y = self.se(y)
        return self.relu(x + y)


class PolicyValueNet(nn.Module):
    """
    ~3.5M params at default 128ch x 10 blocks.

    forward(x) -> (policy_logits (B, 4672), value (B,) in (-1, 1))
    Value is from the perspective of the side to move.
    """

    def __init__(self, channels: int = CFG.channels, blocks: int = CFG.blocks,
                 se: bool = CFG.se):
        super().__init__()
        self.net_config = {"channels": channels, "blocks": blocks, "se": se}
        self.stem = nn.Sequential(
            nn.Conv2d(CFG.input_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(*[ResBlock(channels, se) for _ in range(blocks)])

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
                 use_graphs: bool = True, frozen: bool = False):
        # frozen=True: the weights will never change (playing, not training),
        # so evaluate a private copy with BatchNorm folded into the convs and
        # everything cast to fp16 once, instead of re-casting the fp32
        # weights on every call. Same outputs up to fp16 rounding.
        self.frozen = frozen and device.type == "cuda"
        if self.frozen:
            # channels_last lets cuDNN use tensor-core kernels: batch-32 net
            # time ~1.44 -> ~0.99 ms on the 3070 Ti
            model = fold_batchnorm(model).half().to(memory_format=torch.channels_last)
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
            return self._outputs(logits, values, b)

        x = torch.from_numpy(planes_to_float(planes_u8)).to(self.device)
        if self.frozen:
            logits, values = self.model(x.half().contiguous(memory_format=torch.channels_last))
            return logits.float().cpu().numpy(), values.float().cpu().numpy()
        was_training = self.model.training
        self.model.eval()
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.use_amp):
            logits, values = self.model(x)
        if was_training:
            self.model.train()
        return logits.float().cpu().numpy(), values.float().cpu().numpy()

    # -- asynchronous use (pipelined search) ---------------------------------
    # submit() queues the evaluation and returns at once; fetch() waits for
    # it. One Evaluator holds ONE result per batch-size bucket, so a caller
    # keeping two batches in flight alternates between two Evaluators and
    # fetches each result before submitting to that Evaluator again.

    @torch.inference_mode()
    def submit(self, planes_u8: np.ndarray):
        b = len(planes_u8)
        if not (self.use_graphs and b <= self.MAX_GRAPH_BATCH):
            return ("done", self(planes_u8))
        graph, x_in, logits, values = self._graph(1 << (b - 1).bit_length())
        if getattr(self, "_pinned", None) is None:
            self._pinned = torch.empty((self.MAX_GRAPH_BATCH, CFG.input_planes, 8, 8),
                                       dtype=torch.uint8).pin_memory()
        self._pinned[:b].copy_(torch.from_numpy(planes_u8))
        x_in[:b].copy_(self._pinned[:b], non_blocking=True)   # no host wait
        graph.replay()
        return ("graph", (logits, values, b))

    @torch.inference_mode()
    def fetch(self, handle):
        kind, payload = handle
        if kind == "done":
            return payload
        logits, values, b = payload
        return self._outputs(logits, values, b)

    def _outputs(self, logits, values, b):
        if self.frozen:              # copy fp16 (half the bytes), widen on the CPU
            return (logits[:b].cpu().numpy().astype(np.float32),
                    values[:b].cpu().numpy().astype(np.float32))
        return logits[:b].float().cpu().numpy(), values[:b].float().cpu().numpy()

    def _forward(self, x_u8: torch.Tensor):
        x = x_u8.float()
        x[:, HALFMOVE_PLANE] /= 100.0
        if self.frozen:                          # fp16 weights already
            return self.model(x.half().contiguous(memory_format=torch.channels_last))
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


def fold_batchnorm(model: nn.Module) -> nn.Module:
    """Eval-mode copy of `model` with every Conv2d -> BatchNorm2d pair in an
    nn.Sequential merged into one conv (w' = w*g/s, b' = beta - mean*g/s,
    s = sqrt(var + eps)) and the BatchNorm replaced by Identity."""
    import copy
    m = copy.deepcopy(model).eval()
    for seq in [mod for mod in m.modules() if isinstance(mod, nn.Sequential)]:
        for i in range(len(seq) - 1):
            conv, bn = seq[i], seq[i + 1]
            if not (isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d)):
                continue
            scale = bn.weight.data / torch.sqrt(bn.running_var + bn.eps)
            fused = nn.Conv2d(conv.in_channels, conv.out_channels, conv.kernel_size,
                              conv.stride, conv.padding, bias=True).to(conv.weight.device)
            fused.weight.data = conv.weight.data * scale.view(-1, 1, 1, 1)
            bias = conv.bias.data if conv.bias is not None else torch.zeros_like(bn.running_mean)
            fused.bias.data = (bias - bn.running_mean) * scale + bn.bias.data
            seq[i], seq[i + 1] = fused, nn.Identity()
    return m


def load_checkpoint(path: str, device: torch.device) -> tuple[PolicyValueNet, dict]:
    """Build the net from a checkpoint; returns (model, full checkpoint dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net_cfg = ckpt.get("net", {})
    model = PolicyValueNet(
        channels=net_cfg.get("channels", CFG.channels),
        blocks=net_cfg.get("blocks", CFG.blocks),
        se=net_cfg.get("se", False),         # checkpoints before SE have none
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, ckpt
