from collections.abc import Callable
from typing import Tuple
import random
import time
import numpy as np

from game import board, move, enums
from game.enums import MoveType, Cell


# ======================================================================
# CONSTANTS
# ======================================================================

BOARD_SIZE = 8
NUM_CELLS = BOARD_SIZE * BOARD_SIZE

NOISE_EMIT = {
    Cell.BLOCKED: {enums.Noise.SQUEAK: 0.5, enums.Noise.SCRATCH: 0.3, enums.Noise.SQUEAL: 0.2},
    Cell.SPACE:   {enums.Noise.SQUEAK: 0.7, enums.Noise.SCRATCH: 0.15, enums.Noise.SQUEAL: 0.15},
    Cell.PRIMED:  {enums.Noise.SQUEAK: 0.1, enums.Noise.SCRATCH: 0.8, enums.Noise.SQUEAL: 0.1},
    Cell.CARPET:  {enums.Noise.SQUEAK: 0.1, enums.Noise.SCRATCH: 0.1, enums.Noise.SQUEAL: 0.8},
}

DIST_OFFSETS = {-1: 0.12, 0: 0.70, 1: 0.12, 2: 0.06}

CARPET_SCORE = {1: -1, 2: 2, 3: 4, 4: 6, 5: 10, 6: 15, 7: 21}


# ======================================================================
# HELPERS
# ======================================================================

def cell_to_xy(i): return (i % 8, i // 8)
def xy_to_cell(x, y): return y * 8 + x

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_floor(loc, board_state):
    bit = 1 << xy_to_cell(*loc)
    if board_state._carpet_mask & bit:
        return Cell.CARPET
    if board_state._primed_mask & bit:
        return Cell.PRIMED
    if board_state._blocked_mask & bit:
        return Cell.BLOCKED
    return Cell.SPACE


# ======================================================================
# RAT BELIEF (UNCHANGED STRUCTURE)
# ======================================================================

class RatBelief:
    def __init__(self, T):
        self.T = np.array(T, dtype=np.float64)
        self.belief = np.ones(NUM_CELLS) / NUM_CELLS

    def predict(self):
        self.belief = self.belief @ self.T
        self.belief += 0.01 / NUM_CELLS
        self.belief /= max(self.belief.sum(), 1e-12)

    def update_noise(self, noise, board_state):
        lk = np.array([
            NOISE_EMIT[get_floor(cell_to_xy(i), board_state)][noise]
            for i in range(NUM_CELLS)
        ], dtype=np.float64)
        self.belief *= lk
        self.belief /= max(self.belief.sum(), 1e-12)

    def update_distance(self, rd, wpos):
        wx, wy = wpos
        lk = np.zeros(NUM_CELLS)
        for i in range(NUM_CELLS):
            x, y = cell_to_xy(i)
            d = abs(wx - x) + abs(wy - y)
            lk[i] = DIST_OFFSETS.get(rd - d, 0.01)
        self.belief *= lk
        self.belief /= max(self.belief.sum(), 1e-12)

    def best_cell(self):
        i = int(np.argmax(self.belief))
        return cell_to_xy(i), float(self.belief[i])


# ======================================================================
# EVALUATION (MERGED CLEAN VERSION)
# ======================================================================

def evaluate(board_state, rb: RatBelief):
    my = board_state.player_worker
    opp = board_state.opponent_worker

    score = my.get_points() - opp.get_points()

    # carpet + chain value (kept from your second agent idea)
    primed = 0
    for x in range(8):
        for y in range(8):
            if board_state.get_cell((x, y)) == Cell.PRIMED:
                primed += 1
    score += primed * 0.5

    score += len(board_state.get_valid_moves(exclude_search=True)) * 0.05

    # rat component (kept from second agent)
    if rb:
        (rx, ry), p = rb.best_cell()
        myd = manhattan(my.get_location(), (rx, ry))
        opd = manhattan(opp.get_location(), (rx, ry))
        score += p * (opd - myd)

    return score

def max_carpet_length(loc, board_state):
    max_len = 0
    px, py = loc

    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
        length = 0
        x, y = px + dx, py + dy

        while 0 <= x < 8 and 0 <= y < 8:
            if board_state.get_cell((x, y)) == Cell.PRIMED:
                length += 1
                x += dx
                y += dy
            else:
                break

        max_len = max(max_len, length)

    return max_len

# ======================================================================
# MINIMAX (SIMPLIFIED BUT CONSISTENT)
# ======================================================================

def minimax(board_state, depth, rb, time_left, maximizing=True):
    if depth == 0 or time_left() < 5:
        return evaluate(board_state, rb)

    moves = board_state.get_valid_moves(exclude_search=True)
    if not moves:
        return evaluate(board_state, rb)

    if maximizing:
        best = -float("inf")
        for m in moves:
            nb = board_state.forecast_move(m)
            if nb is None:
                continue
            best = max(best, minimax(nb, depth - 1, rb, time_left, False))
        return best
    else:
        best = float("inf")
        for m in moves:
            nb = board_state.forecast_move(m)
            if nb is None:
                continue
            best = min(best, minimax(nb, depth - 1, rb, time_left, True))
        return best


# ======================================================================
# AGENT
# ======================================================================

class PlayerAgent:

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.rb = RatBelief(transition_matrix) if transition_matrix is not None else None
        self.turn = 0

        # FIXED: proper tracking
        self.last_player_search_turn = -1

        # Stats
        self.searches = 0
        self.pos_carpet_moves = 0
        self.neg_carpet_moves = 0
        self.prime_moves = 0
        self.plain_moves = 0

    def commentate(self):
        return f"Stats: {self.searches} searches, {self.pos_carpet_moves} pos carpet moves, {self.neg_carpet_moves} neg carpet moves, {self.prime_moves} primed moves, {self.plain_moves} plain moves"

    def play(self, board: board.Board, sensor_data: Tuple, time_left: Callable):

        self.turn += 1
        noise, rd = sensor_data

        # ----------------------------
        # BELIEF UPDATE (ORDER FIXED)
        # ----------------------------
        if self.rb:
            # opponent search
            loc, found = board.opponent_search
            if loc is not None:
                self.rb.update_noise(noise, board)

            # player search FIXED (no repeated triggering bug)
            my_loc, my_found = board.player_search
            if my_loc is not None and self.turn != self.last_player_search_turn:
                self.rb.update_noise(noise, board)
                self.last_player_search_turn = self.turn

            self.rb.predict()
            if rd is not None:
                self.rb.update_distance(rd, board.player_worker.get_location())

        moves = board.get_valid_moves(exclude_search=True)
        if not moves:
            self.searches += 1
            if self.rb:
                (cx, cy), p = self.rb.best_cell()
                return move.Move.search((cx, cy))
            return move.Move.search((random.randint(0, 7), random.randint(0, 7)))

        # ----------------------------
        # SEARCH ACTION (kept simple merge behavior)
        # ----------------------------
        if self.rb:
            (cx, cy), p = self.rb.best_cell()
            if p > 0.65:
                self.searches += 1
                return move.Move.search((cx, cy))

        # ----------------------------
        # DEPTH CONTROL
        # ----------------------------
        t = time_left()
        depth = 2 if t > 30 else 1

        # ----------------------------
        # MOVE SELECTION
        # ----------------------------
        if depth == 1:
            best_move = None
            best_val = -float("inf")

            for m in moves:
                nb = board.forecast_move(m)
                if nb is None:
                    continue

                val = evaluate(nb, self.rb)

                # carpet bonus (from first agent)
                if m.move_type == MoveType.CARPET:
                    val += CARPET_SCORE.get(m.roll_length, 5) * 2

                dest = nb.player_worker.get_location()
                future_len = max_carpet_length(dest, board)
                val += CARPET_SCORE.get(future_len, 0) * 0.5

                # --- Bad priming penalty ---
                if m.move_type == MoveType.PRIME:
                    if future_len == 0:
                        val -= 2

                # search move value FIXED
                if m.move_type == MoveType.SEARCH and self.rb:
                    x, y = m.search_loc
                    p = self.rb.belief[xy_to_cell(x, y)]
                    val += 6 * p - 2

                if val > best_val:
                    best_val = val
                    best_move = m

            # Stats
            if best_move is not None:
                if best_move.move_type == MoveType.CARPET and best_val > 0:
                    self.pos_carpet_moves += 1
                elif best_move.move_type == MoveType.CARPET:
                    self.neg_carpet_moves += 1
                elif best_move.move_type == MoveType.PRIME:
                    self.prime_moves += 1
                elif best_move.move_type == MoveType.PLAIN:
                    self.plain_moves += 1
                elif best_move.move_type == MoveType.SEARCH:
                    self.searches += 1
            return best_move

        # depth 2 minimax
        best_move = None
        best_val = -float("inf")

        for m in moves:
            nb = board.forecast_move(m)
            if nb is None:
                continue

            val = minimax(nb, 1, self.rb, time_left)

            if val > best_val:
                best_val = val
                best_move = m

        # Stats
        if best_move is not None:
            if best_move.move_type == MoveType.CARPET and best_val > 0:
                self.pos_carpet_moves += 1
            elif best_move.move_type == MoveType.CARPET:
                self.neg_carpet_moves += 1
            elif best_move.move_type == MoveType.PRIME:
                self.prime_moves += 1
            elif best_move.move_type == MoveType.PLAIN:
                self.plain_moves += 1
            elif best_move.move_type == MoveType.SEARCH:
                self.searches += 1
        return best_move