"""
ui.py — tkinter chess UI.  Structure identical to Act 2.3; engine swapped.
"""

import tkinter as tk
import chess

from engine import choose_move, MODEL_PATH, _model
import os

SQUARE_SIZE = 64
BOARD_SIZE = 8
LIGHT_SQUARE = "#f0d9b5"
DARK_SQUARE = "#b58863"
PIECE_ICONS = {
    chess.WHITE: {
        chess.PAWN: "\u2659", chess.KNIGHT: "\u2658",
        chess.BISHOP: "\u2657", chess.ROOK: "\u2656",
        chess.QUEEN: "\u2655", chess.KING: "\u2654",
    },
    chess.BLACK: {
        chess.PAWN: "\u265F", chess.KNIGHT: "\u265E",
        chess.BISHOP: "\u265D", chess.ROOK: "\u265C",
        chess.QUEEN: "\u265B", chess.KING: "\u265A",
    },
}


class ChessUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.board = chess.Board()
        self.last_move = None
        self.selected_square = None
        self.valid_moves: set[int] = set()
        self.player_vs_bot = True
        self.player_color = chess.BLACK

        eval_tag = "Neural Net" if _model is not None else "Material (no model)"
        root.title(f"Act 3 — Neural Evaluator [{eval_tag}]")

        self.canvas = tk.Canvas(
            root,
            width=SQUARE_SIZE * BOARD_SIZE,
            height=SQUARE_SIZE * BOARD_SIZE,
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, columnspan=4, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.on_square_click)

        self.move_entry = tk.Entry(root, width=20)
        self.move_entry.grid(row=1, column=0, padx=10, sticky="w")

        self.apply_button = tk.Button(root, text="Apply Move", command=self.apply_move)
        self.apply_button.grid(row=1, column=1, padx=5)

        self.mode_var = tk.BooleanVar(value=True)
        self.mode_check = tk.Checkbutton(
            root,
            text="Player vs Bot (AI is White)",
            variable=self.mode_var,
            command=self.on_mode_change,
        )
        self.mode_check.grid(row=1, column=2, columnspan=2, padx=5, sticky="e")

        self.reset_button = tk.Button(root, text="New Game", command=self.reset_game)
        self.reset_button.grid(row=2, column=0, padx=10, sticky="w")

        model_status = f"Model: {MODEL_PATH}" if os.path.exists(MODEL_PATH) else "Model: NOT FOUND — run train.py first"
        self.model_label = tk.Label(root, text=model_status, anchor="w",
                                    fg="green" if os.path.exists(MODEL_PATH) else "red")
        self.model_label.grid(row=2, column=1, columnspan=3, padx=5, sticky="w")

        self.turn_label = tk.Label(root, text="", anchor="w")
        self.turn_label.grid(row=3, column=0, columnspan=4, padx=10, sticky="w")

        self.status_label = tk.Label(root, text="", anchor="w")
        self.status_label.grid(row=4, column=0, columnspan=4, padx=10, sticky="w")

        self.last_move_label = tk.Label(root, text="Last move: -", anchor="w")
        self.last_move_label.grid(row=5, column=0, columnspan=4, padx=10, pady=(0, 10), sticky="w")

        self.update_ui()
        self.on_mode_change()

    # ------------------------------------------------------------------

    def apply_move(self) -> None:
        if self.player_vs_bot and self.board.turn != self.player_color:
            self.set_status("It is the AI's turn. You are Black.")
            return
        raw = self.move_entry.get().strip()
        if not raw:
            self.set_status("Enter a move in UCI format like e2e4 or g1f3.")
            return
        try:
            move = chess.Move.from_uci(raw)
        except ValueError:
            self.set_status("Invalid UCI. Try something like e2e4 or g1f3.")
            return
        if move not in self.board.legal_moves:
            self.set_status("Illegal move for the side to move.")
            return
        self.push_move(move, source="You")
        self.move_entry.delete(0, tk.END)

    def on_square_click(self, event) -> None:
        if self.player_vs_bot and self.board.turn != self.player_color:
            self.set_status("It is the AI's turn. You are Black.")
            return
        file = event.x // SQUARE_SIZE
        rank = BOARD_SIZE - 1 - (event.y // SQUARE_SIZE)
        square = chess.square(file, rank)

        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.selected_square = square
                self.valid_moves = {m.to_square for m in self.board.legal_moves
                                    if m.from_square == square}
                self.set_status(f"Selected {chess.square_name(square)}.")
                self.update_ui()
            else:
                self.set_status("No piece to move there.")
        else:
            if square == self.selected_square:
                self.selected_square = None
                self.valid_moves = set()
                self.set_status("Deselected.")
                self.update_ui()
            elif square in self.valid_moves:
                move = chess.Move(self.selected_square, square)
                self.push_move(move, source="You")
                self.selected_square = None
                self.valid_moves = set()
            else:
                self.set_status("Invalid destination.")

    def engine_move(self) -> None:
        if self.board.is_game_over():
            self.set_status("Game over.")
            return
        move = choose_move(self.board)
        if move is None:
            self.set_status("No legal moves.")
            return
        self.push_move(move, source="Engine")

    def push_move(self, move: chess.Move, source: str) -> None:
        self.board.push(move)
        self.last_move = move
        self.selected_square = None
        self.valid_moves = set()
        self.set_status(f"{source} played {move.uci()}.")
        self.update_ui()
        self.maybe_handle_ai_move()

    def maybe_handle_ai_move(self) -> None:
        if not self.player_vs_bot:
            return
        if self.board.is_game_over():
            return
        if self.board.turn == chess.WHITE:
            self.root.after(250, self.engine_move)

    def update_ui(self) -> None:
        self.canvas.delete("all")
        for rank in range(BOARD_SIZE):
            for file in range(BOARD_SIZE):
                square = chess.square(file, BOARD_SIZE - 1 - rank)
                color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE
                x1, y1 = file * SQUARE_SIZE, rank * SQUARE_SIZE
                x2, y2 = x1 + SQUARE_SIZE, y1 + SQUARE_SIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline=color)

                if square == self.selected_square:
                    self.canvas.create_rectangle(x1, y1, x2, y2,
                                                 fill="#baca44", outline="#baca44")
                elif square in self.valid_moves:
                    self.canvas.create_oval(
                        x1 + SQUARE_SIZE // 3, y1 + SQUARE_SIZE // 3,
                        x2 - SQUARE_SIZE // 3, y2 - SQUARE_SIZE // 3,
                        fill="#baca44", outline="#baca44",
                    )

                piece = self.board.piece_at(square)
                if piece:
                    sym = PIECE_ICONS[piece.color][piece.piece_type]
                    self.canvas.create_text(
                        (x1 + x2) / 2, (y1 + y2) / 2,
                        text=sym,
                        fill="#000000" if piece.color == chess.BLACK else "#ffffff",
                        font=("Segoe UI Symbol", 32, "bold"),
                    )

        turn = "White" if self.board.turn == chess.WHITE else "Black"
        result = self.board.result() if self.board.is_game_over() else "*"
        self.turn_label.configure(text=f"Turn: {turn} | Result: {result}")
        self.last_move_label.configure(
            text=f"Last move: {self.last_move.uci()}" if self.last_move else "Last move: -"
        )

    def set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def on_mode_change(self) -> None:
        self.player_vs_bot = bool(self.mode_var.get())
        if self.player_vs_bot:
            self.set_status("Player vs Bot: you are Black, AI is White.")
        else:
            self.set_status("Manual mode: you can move either side.")
        self.maybe_handle_ai_move()

    def reset_game(self) -> None:
        self.board.reset()
        self.last_move = None
        self.selected_square = None
        self.valid_moves = set()
        self.set_status("New game. AI plays White.")
        self.update_ui()
        self.maybe_handle_ai_move()


def main() -> None:
    root = tk.Tk()
    ChessUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
