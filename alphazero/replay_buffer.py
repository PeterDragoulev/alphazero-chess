"""
replay_buffer.py — disk-persisted ring buffer of training positions.

Positions live in preallocated RAM arrays (uint8 planes ≈ 1.2 KB each, so the
default 500k capacity is ~600 MB). Every save() flushes newly added positions
to a compressed shard file under data/buffer/, so training can be killed and
resumed without losing data; shards that have been fully overwritten in the
ring are deleted.

Policy targets are stored sparse (legal-move indices + probabilities) and
densified per batch at sample time.
"""

import json
import os

import numpy as np

from config import CFG
from encoding import PLANES, POLICY_SIZE, planes_to_float


class ReplayBuffer:
    def __init__(self, directory: str = CFG.buffer_dir,
                 capacity: int = CFG.buffer_capacity):
        self.dir = directory
        self.capacity = capacity
        self.total = 0                       # positions ever added
        self.planes = np.zeros((capacity, PLANES, 8, 8), dtype=np.uint8)
        self.z = np.zeros(capacity, dtype=np.int8)
        self.policies = [None] * capacity    # (idx u16 array, prob f16 array)
        self._unsaved = []                   # adds since last save()
        os.makedirs(self.dir, exist_ok=True)
        self._load()

    @property
    def size(self) -> int:
        return min(self.total, self.capacity)

    # -- adding -------------------------------------------------------------

    def add(self, planes_u8, pol_idx, pol_prob, z: int) -> None:
        row = self.total % self.capacity
        self.planes[row] = planes_u8
        self.z[row] = z
        self.policies[row] = (pol_idx, pol_prob)
        self._unsaved.append((planes_u8, pol_idx, pol_prob, z))
        self.total += 1

    def add_game(self, records, z_white: int) -> None:
        """records: list of (planes, pol_idx, pol_prob, turn_is_white)."""
        for planes_u8, pol_idx, pol_prob, turn in records:
            self.add(planes_u8, pol_idx, pol_prob, z_white if turn else -z_white)

    # -- sampling -------------------------------------------------------------

    def sample(self, batch_size: int):
        """Returns (planes f32 (B,19,8,8), pi f32 (B,4672), z f32 (B,))."""
        rows = np.random.randint(0, self.size, batch_size)
        planes = planes_to_float(self.planes[rows])
        pi = np.zeros((batch_size, POLICY_SIZE), dtype=np.float32)
        for i, row in enumerate(rows):
            idx, prob = self.policies[row]
            pi[i, idx] = prob.astype(np.float32)
        return planes, pi, self.z[rows].astype(np.float32)

    # -- persistence -----------------------------------------------------------

    def save(self) -> None:
        if self._unsaved:
            start = self.total - len(self._unsaved)
            path = os.path.join(self.dir, f"shard_{start:012d}.npz")
            pol_lens = np.array([len(p[1]) for p in self._unsaved], dtype=np.int32)
            np.savez_compressed(
                path,
                planes=np.stack([p[0] for p in self._unsaved]),
                z=np.array([p[3] for p in self._unsaved], dtype=np.int8),
                pol_idx=np.concatenate([p[1] for p in self._unsaved]),
                pol_prob=np.concatenate([p[2] for p in self._unsaved]),
                pol_lens=pol_lens,
            )
            self._unsaved = []
        self._write_meta()
        self._prune()

    def _write_meta(self) -> None:
        tmp = os.path.join(self.dir, "meta.json.tmp")
        with open(tmp, "w") as f:
            json.dump({"total": self.total}, f)
        os.replace(tmp, os.path.join(self.dir, "meta.json"))

    def _shards(self):
        """Sorted list of (start_index, path)."""
        out = []
        for name in os.listdir(self.dir):
            if name.startswith("shard_") and name.endswith(".npz"):
                out.append((int(name[6:-4]), os.path.join(self.dir, name)))
        return sorted(out)

    def _prune(self) -> None:
        shards = self._shards()
        for i, (start, path) in enumerate(shards):
            end = shards[i + 1][0] if i + 1 < len(shards) else self.total
            if end <= self.total - self.capacity:
                os.remove(path)

    def _load(self) -> None:
        meta_path = os.path.join(self.dir, "meta.json")
        if not os.path.exists(meta_path):
            return
        with open(meta_path) as f:
            self.total = json.load(f)["total"]
        for start, path in self._shards():
            data = np.load(path)
            planes, z = data["planes"], data["z"]
            pol_idx, pol_prob = data["pol_idx"], data["pol_prob"]
            offsets = np.concatenate([[0], np.cumsum(data["pol_lens"])])
            for j in range(len(planes)):
                abs_i = start + j
                if abs_i >= self.total or abs_i < self.total - self.capacity:
                    continue
                row = abs_i % self.capacity
                self.planes[row] = planes[j]
                self.z[row] = z[j]
                self.policies[row] = (pol_idx[offsets[j]:offsets[j + 1]],
                                      pol_prob[offsets[j]:offsets[j + 1]])
        loaded = sum(p is not None for p in self.policies)
        print(f"Replay buffer: resumed {loaded:,} positions "
              f"({self.total:,} generated lifetime)")
