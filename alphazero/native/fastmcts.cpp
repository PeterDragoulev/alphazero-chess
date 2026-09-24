// fastmcts — the MCTS tree of mcts.py in C++ (pybind11 extension).
//
// Same algorithm and rules as mcts.MCTS (PUCT with FPU reduction, virtual
// loss in integer counters, twofold repetition = draw in search, 50-move /
// insufficient material / mate / stalemate terminals, tree reuse on advance),
// but move generation (chess.hpp), board push/pop, selection, backup, move
// indexing and board encoding all run natively. Python keeps only the network
// call: select_leaves() returns the leaves' encoded planes, the caller runs
// the net and hands the outputs back with expand_leaves().
//
// Arithmetic mirrors the numpy code (float32 storage for P and W, float64
// selection scores), so results match mcts.py up to float rounding. Legal
// moves are sorted canonically (from, to, promotion), unlike python-chess's
// generation order; mcts.py sorts the same way when compared in tests.
//
// Build: native/build.sh  (produces fastmcts*.so next to mcts.py)

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "chess.hpp"

namespace py = pybind11;
using namespace chess;

namespace {

constexpr int PLANES = 19;
constexpr int POLICY_SIZE = 73 * 64;
constexpr int HALFMOVE_PLANE = 17;
constexpr uint64_t DARK_SQUARES = 0xAA55AA55AA55AA55ULL;

// ---------------------------------------------------------------------------
// moves: our own compact form, standard (non-960) coordinates
// ---------------------------------------------------------------------------

struct Mv {
    uint8_t from, to, promo;   // promo: 0 none, else chess piece type 2..5 (N,B,R,Q)
    Move native;               // chess.hpp move (castling = king takes rook)
};

inline int std_to(const Move& m) {
    int from = m.from().index(), to = m.to().index();
    if (m.typeOf() == Move::CASTLING) {   // king-takes-rook -> king destination
        int rank = from / 8;
        to = rank * 8 + (to % 8 > from % 8 ? 6 : 2);
    }
    return to;
}

inline Mv make_mv(const Move& m) {
    Mv v;
    v.from = (uint8_t)m.from().index();
    v.to = (uint8_t)std_to(m);
    v.promo = 0;
    if (m.typeOf() == Move::PROMOTION) {
        // chess.hpp PieceType: PAWN 0, KNIGHT 1, BISHOP 2, ROOK 3, QUEEN 4
        v.promo = (uint8_t)(static_cast<int>(m.promotionType().internal()) + 1);
    }
    v.native = m;
    return v;
}

inline bool mv_less(const Mv& a, const Mv& b) {
    if (a.from != b.from) return a.from < b.from;
    if (a.to != b.to) return a.to < b.to;
    return a.promo < b.promo;
}

std::string mv_uci(const Mv& m) {
    std::string s;
    s += char('a' + m.from % 8);
    s += char('1' + m.from / 8);
    s += char('a' + m.to % 8);
    s += char('1' + m.to / 8);
    if (m.promo) s += " nbrq"[m.promo - 1];
    return s;
}

// AlphaZero 73x64 index, identical to encoding.move_to_index.
int move_to_index(const Mv& m, bool white_to_move) {
    int frm = m.from, to = m.to;
    if (!white_to_move) {
        frm ^= 56;
        to ^= 56;
    }
    int df = (to & 7) - (frm & 7);
    int dr = (to >> 3) - (frm >> 3);
    int plane;
    if (m.promo && m.promo != 5) {                 // under-promotion
        int kind = m.promo - 2;                    // N 0, B 1, R 2
        plane = 64 + kind * 3 + (df + 1);
    } else {
        static const int kn[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2},
                                     {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
        int k = -1;
        for (int i = 0; i < 8; i++)
            if (kn[i][0] == df && kn[i][1] == dr) k = i;
        if (k >= 0) {
            plane = 56 + k;
        } else {
            int dist = std::max(std::abs(df), std::abs(dr));
            int sx = (df > 0) - (df < 0), sy = (dr > 0) - (dr < 0);
            static const int dirs[8][2] = {{0, 1}, {1, 1}, {1, 0}, {1, -1},
                                           {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}};
            int d = 0;
            for (int i = 0; i < 8; i++)
                if (dirs[i][0] == sx && dirs[i][1] == sy) d = i;
            plane = d * 7 + (dist - 1);
        }
    }
    return plane * 64 + frm;
}

// ---------------------------------------------------------------------------
// board helpers
// ---------------------------------------------------------------------------

inline uint64_t bb(const Board& b, PieceType pt, Color c) { return b.pieces(pt, c).getBits(); }
inline uint64_t bbt(const Board& b, PieceType pt) { return b.pieces(pt).getBits(); }

inline uint64_t flip_vertical(uint64_t x) { return __builtin_bswap64(x); }

// python-chess Board.has_insufficient_material(color)
bool insufficient(const Board& b, Color c) {
    uint64_t own = b.us(c).getBits(), opp = b.us(~c).getBits();
    uint64_t pawns = bbt(b, PieceType::PAWN), knights = bbt(b, PieceType::KNIGHT),
             bishops = bbt(b, PieceType::BISHOP), rooks = bbt(b, PieceType::ROOK),
             queens = bbt(b, PieceType::QUEEN), kings = bbt(b, PieceType::KING);
    if (own & (pawns | rooks | queens)) return false;
    if (own & knights)
        return __builtin_popcountll(own) <= 2 && !(opp & ~kings & ~queens);
    if (own & bishops) {
        bool same = !(bishops & DARK_SQUARES) || !(bishops & ~DARK_SQUARES);
        return same && !pawns && !knights;
    }
    return true;
}

// python-chess: legal moves known -> value for the side to move or NaN
double terminal_value(const Board& b, int n_moves) {
    if (n_moves == 0) return b.inCheck() ? -1.0 : 0.0;
    if ((insufficient(b, Color::WHITE) && insufficient(b, Color::BLACK)) ||
        b.halfMoveClock() >= 100)
        return 0.0;
    return std::numeric_limits<double>::quiet_NaN();
}

// encoding.encode_board. ep = python-chess ep_square (set after ANY double
// push, even if no capture is possible) or -1.
void encode(const Board& b, int ep, uint8_t* out) {
    std::memset(out, 0, PLANES * 64);
    bool white = b.sideToMove() == Color::WHITE;
    Color us = b.sideToMove(), them = ~us;
    static const PieceType pts[6] = {PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
                                     PieceType::ROOK, PieceType::QUEEN, PieceType::KING};
    for (int ci = 0; ci < 2; ci++) {
        Color c = ci == 0 ? us : them;
        for (int p = 0; p < 6; p++) {
            uint64_t m = bb(b, pts[p], c);
            if (!white) m = flip_vertical(m);
            uint8_t* plane = out + (ci * 6 + p) * 64;
            while (m) {
                int sq = __builtin_ctzll(m);
                plane[sq] = 1;
                m &= m - 1;
            }
        }
    }
    auto cr = b.castlingRights();
    using Side = Board::CastlingRights::Side;
    // standard chess: king on its start square is implied by the rights
    if (cr.has(us, Side::KING_SIDE)) std::memset(out + 12 * 64, 1, 64);
    if (cr.has(us, Side::QUEEN_SIDE)) std::memset(out + 13 * 64, 1, 64);
    if (cr.has(them, Side::KING_SIDE)) std::memset(out + 14 * 64, 1, 64);
    if (cr.has(them, Side::QUEEN_SIDE)) std::memset(out + 15 * 64, 1, 64);
    if (ep >= 0) out[16 * 64 + (white ? ep : (ep ^ 56))] = 1;
    std::memset(out + HALFMOVE_PLANE * 64, (uint8_t)std::min<uint32_t>(b.halfMoveClock(), 100), 64);
    std::memset(out + 18 * 64, 1, 64);
}

// python-chess ep_square after playing m on b (b = position BEFORE m)
inline int ep_after(const Board& b, const Mv& m) {
    if (b.at<PieceType>(Square(m.from)) == PieceType::PAWN && std::abs(m.to - m.from) == 16)
        return (m.to + m.from) / 2;
    return -1;
}

// ---------------------------------------------------------------------------
// tree
// ---------------------------------------------------------------------------

struct Node {
    std::vector<Mv> moves;          // empty until expanded
    std::vector<float> P, W;
    std::vector<int32_t> N, VL;
    std::vector<int32_t> children;  // node index or -1
    bool expanded = false;
    bool in_flight = false;
    double terminal = std::numeric_limits<double>::quiet_NaN();   // NaN = not terminal / unknown
    bool terminal_known = false;
};

struct Leaf {
    int32_t node;
    std::vector<std::pair<int32_t, int32_t>> path;   // (node, edge)
    std::vector<Mv> moves;
    bool white;
};

class Tree {
  public:
    Tree(const std::string& start_fen, const std::vector<std::string>& moves_uci, int root_ep,
         double c_puct, py::object fpu_reduction, bool repetition_draws)
        : c_puct_(c_puct), repetition_draws_(repetition_draws) {
        use_fpu_ = !fpu_reduction.is_none();
        fpu_ = use_fpu_ ? fpu_reduction.cast<double>() : 0.0;
        board_.setFen(start_fen);
        history_.insert(board_.hash());
        for (const auto& u : moves_uci) {
            Move m = uci::uciToMove(board_, u);
            board_.makeMove(m);
            history_.insert(board_.hash());
        }
        board_ = Board(board_.getFen());   // fresh board: no undo stack needed
        root_ep_ = root_ep;
        root_ = new_node();
    }

    // ---- select / expand ---------------------------------------------------

    // One descent without virtual loss (select_leaf). Returns planes or None.
    py::object select_leaf() {
        if (!batch_.empty()) throw std::runtime_error("expand_backup() not called after select");
        int r = descend(false);
        if (r <= 0) return py::none();
        py::array_t<uint8_t> out({1, PLANES, 8, 8});
        std::memcpy(out.mutable_data(), planes_.data(), PLANES * 64);
        planes_.clear();
        return out[py::int_(0)];
    }

    void expand_backup(py::array_t<float, py::array::c_style | py::array::forcecast> logits,
                       double value) {
        if (batch_.size() != 1) throw std::runtime_error("no pending leaf");
        const float* lg = logits.data();
        expand(batch_[0], lg, value, false);
        batch_.clear();
    }

    // Up to k descents with virtual loss. Returns (planes (n,19,8,8) u8, sims).
    py::tuple select_leaves(int k) {
        if (!batch_.empty()) throw std::runtime_error("expand_leaves() not called after select");
        int sims = 0;
        for (int i = 0; i < k; i++) {
            int r = descend(true);
            if (r < 0) break;           // collision
            sims++;
        }
        size_t n = batch_.size();
        py::array_t<uint8_t> out({(py::ssize_t)n, (py::ssize_t)PLANES, (py::ssize_t)8, (py::ssize_t)8});
        if (n) std::memcpy(out.mutable_data(), planes_.data(), n * PLANES * 64);
        planes_.clear();
        return py::make_tuple(out, sims);
    }

    void expand_leaves(py::array_t<float, py::array::c_style | py::array::forcecast> logits,
                       py::array_t<float, py::array::c_style | py::array::forcecast> values) {
        size_t n = batch_.size();
        if ((size_t)logits.shape(0) < n || (size_t)values.shape(0) < n)
            throw std::runtime_error("expand_leaves: too few outputs");
        const float* lg = logits.data();
        const float* vs = values.data();
        std::vector<Leaf> batch;
        batch.swap(batch_);
        for (size_t i = 0; i < n; i++) expand(batch[i], lg + i * POLICY_SIZE, (double)vs[i], true);
    }

    // ---- root access -------------------------------------------------------

    bool root_expanded() const { return nodes_[root_].expanded; }
    std::vector<std::string> root_moves() const {
        std::vector<std::string> out;
        for (auto& m : nodes_[root_].moves) out.push_back(mv_uci(m));
        return out;
    }
    py::array_t<int32_t> root_N() const { return vec_arr(nodes_[root_].N); }
    py::array_t<float> root_W() const { return vec_arr(nodes_[root_].W); }
    py::array_t<float> root_P() const { return vec_arr(nodes_[root_].P); }
    void set_root_P(py::array_t<float, py::array::c_style | py::array::forcecast> p) {
        auto& node = nodes_[root_];
        if ((size_t)p.shape(0) != node.P.size()) throw std::runtime_error("set_root_P: size");
        std::memcpy(node.P.data(), p.data(), node.P.size() * sizeof(float));
    }

    // principal variation (most visited child at each step) as uci strings
    std::vector<std::string> pv(int max_len) const {
        std::vector<std::string> out;
        int32_t n = root_;
        while ((int)out.size() < max_len) {
            const Node& node = nodes_[n];
            if (!node.expanded || sum_n(node) == 0) break;
            int i = argmax_n(node);
            out.push_back(mv_uci(node.moves[i]));
            n = node.children[i];
            if (n < 0) break;
        }
        return out;
    }

    py::dict depth_stats() const {
        int depth = 0;
        int32_t n = root_;
        while (nodes_[n].expanded && sum_n(nodes_[n]) > 0) {
            int i = argmax_n(nodes_[n]);
            int32_t c = nodes_[n].children[i];
            if (c < 0 || !nodes_[c].expanded) break;
            depth++;
            n = c;
        }
        int seldepth = 0;
        int64_t total = 0, weighted = 0;
        std::vector<std::pair<int32_t, int>> stack{{root_, 0}};
        while (!stack.empty()) {
            auto [id, d] = stack.back();
            stack.pop_back();
            const Node& node = nodes_[id];
            if (!node.expanded) continue;
            seldepth = std::max(seldepth, d);
            int64_t s = sum_n(node);
            total += s;
            weighted += s * d;
            for (int32_t c : node.children)
                if (c >= 0) stack.push_back({c, d + 1});
        }
        py::dict out;
        out["depth"] = depth;
        out["seldepth"] = seldepth;
        out["mean_depth"] = total ? (double)weighted / (double)total : 0.0;
        return out;
    }

    // ---- advance -------------------------------------------------------------

    // Play a move (uci) at the root, keeping its subtree if present.
    void advance(const std::string& u) {
        if (!batch_.empty()) throw std::runtime_error("advance() with leaves pending");
        Move m = uci::uciToMove(board_, u);
        Mv mv = make_mv(m);
        int32_t new_root = -1;
        const Node& r = nodes_[root_];
        for (size_t i = 0; i < r.moves.size(); i++)
            if (r.moves[i].from == mv.from && r.moves[i].to == mv.to && r.moves[i].promo == mv.promo)
                new_root = r.children[i];
        root_ep_ = ep_after(board_, mv);
        board_.makeMove(m);
        board_ = Board(board_.getFen());       // drop the undo stack
        history_.insert(board_.hash());
        if (new_root < 0) {
            nodes_.clear();
            root_ = new_node();
        } else {
            compact(new_root);
        }
        // A leaf scored as a repetition draw can become the real position; at
        // the root we still need a move, so forget the cached verdict.
        Node& nr = nodes_[root_];
        if (!nr.expanded) {
            nr.terminal_known = false;
            nr.terminal = std::numeric_limits<double>::quiet_NaN();
        }
    }

    size_t node_count() const { return nodes_.size(); }

  private:
    // ---- internals -----------------------------------------------------------

    template <class T>
    static py::array_t<T> vec_arr(const std::vector<T>& v) {
        py::array_t<T> a((py::ssize_t)v.size());
        if (!v.empty()) std::memcpy(a.mutable_data(), v.data(), v.size() * sizeof(T));
        return a;
    }
    static int64_t sum_n(const Node& n) {
        int64_t s = 0;
        for (int32_t x : n.N) s += x;
        return s;
    }
    static int argmax_n(const Node& n) {
        int best = 0;
        for (size_t i = 1; i < n.N.size(); i++)
            if (n.N[i] > n.N[best]) best = (int)i;
        return best;
    }

    int32_t new_node() {
        nodes_.emplace_back();
        return (int32_t)nodes_.size() - 1;
    }

    // Keep only the subtree under `keep`, renumbered, as the new root.
    void compact(int32_t keep) {
        std::vector<Node> fresh;
        fresh.reserve(nodes_.size());
        std::vector<std::pair<int32_t, int32_t>> stack;   // (old id, new id)
        fresh.push_back(std::move(nodes_[keep]));
        stack.push_back({keep, 0});
        while (!stack.empty()) {
            auto [old_id, new_id] = stack.back();
            stack.pop_back();
            (void)old_id;
            for (auto& c : fresh[new_id].children) {
                if (c < 0) continue;
                int32_t nid = (int32_t)fresh.size();
                fresh.push_back(std::move(nodes_[c]));
                stack.push_back({c, nid});
                c = nid;
            }
        }
        nodes_.swap(fresh);
        root_ = 0;
    }

    int puct_select(const Node& node) const {
        const size_t k = node.moves.size();
        int64_t n_total = 0;
        double w_total = 0.0, p_visited = 0.0;
        const bool vl = !node.VL.empty();
        for (size_t i = 0; i < k; i++) {
            int32_t n = node.N[i] + (vl ? node.VL[i] : 0);
            n_total += n;
            w_total += (double)node.W[i] - (vl ? node.VL[i] : 0);
            if (n > 0) p_visited += node.P[i];
        }
        double fpu_q = 0.0;
        const bool fpu = n_total && use_fpu_;
        if (fpu) fpu_q = w_total / (double)n_total - fpu_ * std::sqrt(p_visited);
        const double sq = std::sqrt((double)(n_total + 1));
        const float cp32 = (float)c_puct_;
        int best = 0;
        double best_s = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < k; i++) {
            int32_t n = node.N[i] + (vl ? node.VL[i] : 0);
            double w = (double)node.W[i] - (vl ? node.VL[i] : 0);
            double q;
            if (fpu && n == 0) q = fpu_q;
            else q = w / (double)std::max(n, 1);
            double u = (double)(cp32 * node.P[i]) * (sq / (double)(1 + n));
            double s = q + u;
            if (s > best_s) {
                best_s = s;
                best = (int)i;
            }
        }
        return best;
    }

    // Returns 1 = leaf queued for evaluation, 0 = terminal (backed up), -1 = collision.
    int descend(bool virtual_loss) {
        Board& b = board_;
        int32_t id = root_;
        Leaf leaf;
        std::vector<uint64_t> line{b.hash()};
        std::vector<Move> played;
        int ep = root_ep_;

        while (nodes_[id].expanded) {
            Node& node = nodes_[id];
            int i = puct_select(node);
            leaf.path.push_back({id, i});
            if (virtual_loss) {
                if (node.VL.empty()) node.VL.assign(node.moves.size(), 0);
                node.VL[i] += 1;
            }
            const Mv& mv = node.moves[i];
            ep = ep_after(b, mv);
            b.makeMove(mv.native);
            played.push_back(mv.native);
            line.push_back(b.hash());
            int32_t c = node.children[i];
            if (c < 0) {
                c = new_node();              // may reallocate nodes_: re-index below
                nodes_[id].children[i] = c;
            }
            id = c;
        }

        auto unwind = [&]() {
            for (auto it = played.rbegin(); it != played.rend(); ++it) b.unmakeMove(*it);
        };

        if (nodes_[id].in_flight) {
            unwind();
            revert_vl(leaf.path);
            return -1;
        }

        Node& node = nodes_[id];
        double term;
        std::vector<Mv> moves;
        if (node.terminal_known) {
            term = node.terminal;
        } else {
            uint64_t key = line.back();
            bool rep = false;
            if (!leaf.path.empty() && repetition_draws_) {
                if (history_.count(key)) rep = true;
                for (size_t j = 0; j + 1 < line.size() && !rep; j++)
                    if (line[j] == key) rep = true;
            }
            if (rep) {
                term = 0.0;
            } else {
                Movelist ml;
                movegen::legalmoves(ml, b);
                moves.reserve(ml.size());
                for (const auto& m : ml) moves.push_back(make_mv(m));
                std::sort(moves.begin(), moves.end(), mv_less);
                term = terminal_value(b, (int)moves.size());
            }
            node.terminal = term;
            node.terminal_known = !std::isnan(term);
        }
        if (!std::isnan(term)) {
            unwind();
            if (virtual_loss) revert_vl(leaf.path);
            backup(leaf.path, term);
            return 0;
        }

        size_t off = planes_.size();
        planes_.resize(off + PLANES * 64);
        encode(b, ep, planes_.data() + off);
        leaf.node = id;
        leaf.moves = std::move(moves);
        leaf.white = b.sideToMove() == Color::WHITE;
        unwind();
        node.in_flight = virtual_loss;
        batch_.push_back(std::move(leaf));
        return 1;
    }

    void expand(Leaf& leaf, const float* logits, double value, bool virtual_loss) {
        Node& node = nodes_[leaf.node];
        if (virtual_loss) {
            revert_vl(leaf.path);
            node.in_flight = false;
        }
        const size_t k = leaf.moves.size();
        std::vector<float> pr(k);
        float mx = -std::numeric_limits<float>::infinity();
        for (size_t i = 0; i < k; i++) {
            pr[i] = logits[move_to_index(leaf.moves[i], leaf.white)];
            mx = std::max(mx, pr[i]);
        }
        float sum = 0.0f;
        for (size_t i = 0; i < k; i++) {
            pr[i] = std::exp(pr[i] - mx);
            sum += pr[i];
        }
        for (size_t i = 0; i < k; i++) pr[i] /= sum;
        node.moves = std::move(leaf.moves);
        node.P = std::move(pr);
        node.N.assign(k, 0);
        node.W.assign(k, 0.0f);
        node.children.assign(k, -1);
        node.expanded = true;
        backup(leaf.path, value);
    }

    void revert_vl(const std::vector<std::pair<int32_t, int32_t>>& path) {
        for (auto& [id, i] : path) nodes_[id].VL[i] -= 1;
    }

    void backup(const std::vector<std::pair<int32_t, int32_t>>& path, double leaf_value) {
        double v = leaf_value;
        for (auto it = path.rbegin(); it != path.rend(); ++it) {
            v = -v;
            Node& n = nodes_[it->first];
            n.N[it->second] += 1;
            n.W[it->second] = (float)(n.W[it->second] + (float)v);
        }
    }

    Board board_;
    int root_ep_ = -1;
    std::unordered_set<uint64_t> history_;
    std::vector<Node> nodes_;
    int32_t root_ = 0;
    std::vector<Leaf> batch_;
    std::vector<uint8_t> planes_;
    double c_puct_;
    bool use_fpu_;
    double fpu_;
    bool repetition_draws_;
};

// ---- standalone helpers (tests) ------------------------------------------------

py::array_t<uint8_t> encode_fen(const std::string& fen, int ep) {
    Board b(fen);
    py::array_t<uint8_t> out({PLANES, 8, 8});
    encode(b, ep, out.mutable_data());
    return out;
}

// legal moves (sorted uci) + index of each, for cross-checking with python-chess
py::tuple legal_moves_fen(const std::string& fen) {
    Board b(fen);
    Movelist ml;
    movegen::legalmoves(ml, b);
    std::vector<Mv> mv;
    for (auto& m : ml) mv.push_back(make_mv(m));
    std::sort(mv.begin(), mv.end(), mv_less);
    std::vector<std::string> u;
    std::vector<int> idx;
    bool white = b.sideToMove() == Color::WHITE;
    for (auto& m : mv) {
        u.push_back(mv_uci(m));
        idx.push_back(move_to_index(m, white));
    }
    double t = terminal_value(b, (int)mv.size());
    return py::make_tuple(u, idx, t);
}

}  // namespace

PYBIND11_MODULE(fastmcts, m) {
    m.doc() = "Native MCTS tree for the AlphaZero chess engine (see mcts.py)";
    py::class_<Tree>(m, "Tree")
        .def(py::init<const std::string&, const std::vector<std::string>&, int, double, py::object, bool>(),
             py::arg("start_fen"), py::arg("moves"), py::arg("root_ep"), py::arg("c_puct"),
             py::arg("fpu_reduction"), py::arg("repetition_draws"))
        .def("select_leaf", &Tree::select_leaf)
        .def("expand_backup", &Tree::expand_backup)
        .def("select_leaves", &Tree::select_leaves)
        .def("expand_leaves", &Tree::expand_leaves)
        .def("root_expanded", &Tree::root_expanded)
        .def("root_moves", &Tree::root_moves)
        .def("root_N", &Tree::root_N)
        .def("root_W", &Tree::root_W)
        .def("root_P", &Tree::root_P)
        .def("set_root_P", &Tree::set_root_P)
        .def("pv", &Tree::pv, py::arg("max_len") = 64)
        .def("depth_stats", &Tree::depth_stats)
        .def("advance", &Tree::advance)
        .def("node_count", &Tree::node_count);
    m.def("encode_fen", &encode_fen);
    m.def("legal_moves_fen", &legal_moves_fen);
}
