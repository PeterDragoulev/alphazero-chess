"""
live_viewer.py — watch the engine's games live while a match runs.

uci.py writes a small status file per engine process to /tmp/azchess_live/
on every move (position, last move, its color, eval, depth). This window
shows one board per active game, side by side, refreshed a few times a
second. Games disappear when their engine quits or goes quiet.

    .venv/bin/python live_viewer.py      # start before or during a match
"""

import glob
import json
import os
import time
import tkinter as tk

import chess

LIVE_DIR = "/tmp/azchess_live"
SQ = 52                           # square size in pixels
STALE = 120                       # hide games with no update for this long (s)
LIGHT, DARK = "#f0d9b5", "#b58863"
HL_LIGHT, HL_DARK = "#cdd26a", "#aaa23a"
GLYPH = {"P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
         "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚"}


def ensure_running():
    """Start the viewer in the background unless one is already open."""
    import subprocess
    import sys
    mine = os.getpid()
    for pid in os.listdir("/proc"):
        if pid.isdigit() and int(pid) != mine:
            try:
                cmd = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
            except OSError:
                continue
            if any(a.endswith(b"live_viewer.py") for a in cmd) and b"python" in cmd[0]:
                return
    subprocess.Popen([sys.executable, os.path.abspath(__file__)],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)


def read_games():
    games = []
    for path in sorted(glob.glob(os.path.join(LIVE_DIR, "*.json"))):
        try:
            with open(path) as f:
                g = json.load(f)
        except (OSError, ValueError):
            continue
        if time.time() - g.get("time", 0) < STALE:
            games.append(g)
    return games


class Viewer:
    def __init__(self, root):
        self.root = root
        root.title("AlphaZeroChess — live games")
        root.configure(bg="#262421")
        self.frame = tk.Frame(root, bg="#262421")
        self.frame.pack(padx=10, pady=10)
        self.empty = tk.Label(root, text="Waiting for games... (start a match)",
                              fg="#bababa", bg="#262421", font=("Sans", 12))
        self.panels = []
        self.refresh()

    def refresh(self):
        games = read_games()
        while len(self.panels) < len(games):
            self.panels.append(self._panel(len(self.panels)))
        for i, panel in enumerate(self.panels):
            if i < len(games):
                self._draw(panel, games[i])
                panel["frame"].grid()
            else:
                panel["frame"].grid_remove()
        if games:
            self.empty.pack_forget()
        else:
            self.empty.pack(pady=(0, 10))
        self.root.after(300, self.refresh)

    def _panel(self, i):
        frame = tk.Frame(self.frame, bg="#262421")
        frame.grid(row=0, column=i, padx=10)
        title = tk.Label(frame, fg="#ffffff", bg="#262421", font=("Sans", 11, "bold"))
        title.pack(anchor="w")
        canvas = tk.Canvas(frame, width=8 * SQ, height=8 * SQ, highlightthickness=0)
        canvas.pack()
        status = tk.Label(frame, fg="#bababa", bg="#262421", font=("Sans", 10),
                          justify="left")
        status.pack(anchor="w", pady=(4, 0))
        return {"frame": frame, "title": title, "canvas": canvas, "status": status}

    def _draw(self, panel, g):
        board = chess.Board(g["fen"])
        ours = g["color"]
        flip = ours == "black"                   # our engine at the bottom
        last = chess.Move.from_uci(g["last"]) if g.get("last") else None
        hl = {last.from_square, last.to_square} if last else set()
        c = panel["canvas"]
        c.delete("all")
        for sq in chess.SQUARES:
            f, r = chess.square_file(sq), chess.square_rank(sq)
            col, row = (7 - f, r) if flip else (f, 7 - r)
            light = (f + r) % 2 == 1
            fill = (HL_LIGHT if light else HL_DARK) if sq in hl else (LIGHT if light else DARK)
            x, y = col * SQ, row * SQ
            c.create_rectangle(x, y, x + SQ, y + SQ, fill=fill, outline=fill)
            piece = board.piece_at(sq)
            if piece:
                c.create_text(x + SQ / 2, y + SQ / 2, text=GLYPH[piece.symbol()],
                              font=("Segoe UI Symbol", int(SQ * 0.62)),
                              fill="#ffffff" if piece.color else "#000000")
        panel["title"].configure(text=f"AlphaZeroChess plays {ours.capitalize()}")
        if board.is_game_over():
            state = f"game over: {board.result()}"
        elif g.get("thinking"):
            state = "thinking..."
        else:
            state = f"eval {g.get('eval_cp', 0) / 100:+.2f} (for us)  |  depth {g.get('depth', '?')}  |  {g.get('sims', 0):,} sims"
        mv = f"move {(g['ply'] + 1) // 2}" + (f", last {last.uci()}" if last else "")
        panel["status"].configure(text=f"{mv}\n{state}")


if __name__ == "__main__":
    root = tk.Tk()
    Viewer(root)
    root.mainloop()
