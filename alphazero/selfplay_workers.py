"""
selfplay_workers.py — self-play in several processes feeding one trainer.

Self-play is bound by single-threaded Python (tree search, board ops), not by
the GPU, so one process leaves most CPU cores idle. Each worker here runs its
own SelfPlayPool (CFG.parallel_games games in lockstep, batched on the GPU)
with its own copy of the net, and sends finished games to the trainer over a
queue. Workers reload the weights whenever the trainer saves a new checkpoint
(load_state_dict copies in place, so the Evaluator's CUDA graphs stay valid).

Workers are started with "spawn" (a clean interpreter each), so it doesn't
matter whether the trainer has already initialized CUDA.
"""

import multiprocessing as mp
import os
import queue
import signal

import numpy as np

_CTX = mp.get_context("spawn")


class _QueueSink:
    """SelfPlayPool sink that ships finished games to the trainer."""

    def __init__(self, out, wid):
        self.out, self.wid = out, wid
        self.reloads = 0                             # weight updates picked up

    def add_game(self, records, z_white):
        self.out.put((self.wid, self.reloads, records, z_white))


class _EventStop:
    def __init__(self, event):
        self.event = event

    @property
    def requested(self):
        return self.event.is_set()


def _worker(wid, out, stop_event, ckpt_path, seed):
    signal.signal(signal.SIGINT, signal.SIG_IGN)     # the trainer handles Ctrl+C
    import torch
    torch.set_num_threads(1)                         # one core per worker
    from network import Evaluator, load_checkpoint
    from selfplay import SelfPlayPool

    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(ckpt_path, device)
    model.eval()
    pool = SelfPlayPool(Evaluator(model, device))
    mtime = os.path.getmtime(ckpt_path)
    sink, stop = _QueueSink(out, wid), _EventStop(stop_event)
    while not stop_event.is_set():
        pool.play_block(sink, target_games=1, stop=stop)
        try:
            new_mtime = os.path.getmtime(ckpt_path)
        except OSError:                              # mid os.replace: try later
            continue
        if new_mtime != mtime:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])     # in place: graphs stay valid
            mtime = new_mtime
            sink.reloads += 1


class SelfPlayWorkers:
    """N self-play processes; the trainer pulls finished games with get_games()."""

    def __init__(self, n, ckpt_path, seed):
        self.out = _CTX.Queue(maxsize=4 * n + 16)
        self.reloads = {}                            # worker id -> reloads seen
        self.stop_event = _CTX.Event()
        self.procs = [_CTX.Process(target=_worker, daemon=True,
                                   args=(i, self.out, self.stop_event, ckpt_path,
                                         seed + 1000 * i))
                      for i in range(n)]
        for p in self.procs:
            p.start()

    def get_games(self, n, stop):
        """Block until n finished games arrive (or stop). Yields (records, z)."""
        got = 0
        while got < n and not stop.requested:
            try:
                wid, reloads, records, z = self.out.get(timeout=1.0)
                self.reloads[wid] = reloads
            except queue.Empty:
                if not any(p.is_alive() for p in self.procs):
                    raise RuntimeError("all self-play workers died")
                continue
            got += 1
            yield records, z

    def close(self, timeout=30.0):
        """Stop workers. Keep draining the queue meanwhile: a worker blocked on
        a full queue (or flushing its feeder thread) can't exit otherwise."""
        import time
        self.stop_event.set()
        deadline = time.time() + timeout
        while any(p.is_alive() for p in self.procs) and time.time() < deadline:
            try:
                self.out.get(timeout=0.2)
            except queue.Empty:
                pass
        for p in self.procs:
            if p.is_alive():
                p.terminate()
            p.join(timeout=5)
