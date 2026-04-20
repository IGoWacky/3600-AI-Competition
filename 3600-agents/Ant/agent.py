from collections.abc import Callable
from typing import List, Tuple, Optional
import random
import time
import numpy as np

from game import board, move, enums
from game.enums import MoveType, Cell


# ===========================================================================
# Constants
# ===========================================================================

BOARD_SIZE = 8
NUM_CELLS  = BOARD_SIZE * BOARD_SIZE

NOISE_EMIT = {
    Cell.BLOCKED: {enums.Noise.SQUEAK: 0.5,  enums.Noise.SCRATCH: 0.3,  enums.Noise.SQUEAL: 0.2},
    Cell.SPACE:   {enums.Noise.SQUEAK: 0.7,  enums.Noise.SCRATCH: 0.15, enums.Noise.SQUEAL: 0.15},
    Cell.PRIMED:  {enums.Noise.SQUEAK: 0.1,  enums.Noise.SCRATCH: 0.8,  enums.Noise.SQUEAL: 0.1},
    Cell.CARPET:  {enums.Noise.SQUEAK: 0.1,  enums.Noise.SCRATCH: 0.1,  enums.Noise.SQUEAL: 0.8},
}

DIST_OFFSETS = {-1: 0.12, 0: 0.70, 1: 0.12, 2: 0.06}
CARPET_SCORE = {1: -1, 2: 2, 3: 4, 4: 6, 5: 10, 6: 15, 7: 21}

ENDGAME_TURNS = 6  # turns at which we switch to cashout mode


class SearchTimeout(Exception):
    pass


# ===========================================================================
# Utility helpers
# ===========================================================================

def cell_to_xy(i: int) -> Tuple[int, int]:
    return (i % BOARD_SIZE, i // BOARD_SIZE)

def xy_to_cell(x: int, y: int) -> int:
    return y * BOARD_SIZE + x

def manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def get_floor(loc: Tuple[int, int], board_state) -> Cell:
    bit = 1 << xy_to_cell(loc[0], loc[1])
    if board_state._carpet_mask  & bit: return Cell.CARPET
    if board_state._primed_mask  & bit: return Cell.PRIMED
    if board_state._blocked_mask & bit: return Cell.BLOCKED
    return Cell.SPACE

def compute_rat_spawn_dist(T: np.ndarray, steps: int = 1000) -> np.ndarray:
    dist = np.zeros(NUM_CELLS, dtype=np.float64)
    dist[xy_to_cell(0, 0)] = 1.0
    for _ in range(steps):
        dist = dist @ T
    total = dist.sum()
    return dist / total if total > 1e-12 else np.ones(NUM_CELLS) / NUM_CELLS


# ===========================================================================
# RatBelief — Hidden Markov Model
# ===========================================================================

class RatBelief:
    """
    Update order per turn:
      1. update_search()  — apply search result
      2. predict() x2     — rat moves before opponent + before ours
      3. update_noise()   — reweight by noise observation
      4. update_distance()— reweight by noisy distance sensor
    """

    def __init__(self, transition_matrix):
        self.T = np.array(transition_matrix, dtype=np.float64)
        self._spawn_dist = compute_rat_spawn_dist(self.T)
        self.belief = self._spawn_dist.copy()

    def predict(self):
        self.belief = self.belief @ self.T
        self.belief += 0.001 / NUM_CELLS
        self._normalize()

    def update_noise(self, noise, board_state):
        lk = np.array([
            NOISE_EMIT[get_floor(cell_to_xy(i), board_state)].get(noise, 1e-9)
            for i in range(NUM_CELLS)
        ], dtype=np.float64)
        self.belief *= lk
        self._normalize()

    def update_distance(self, reported_dist: int, worker_pos: Tuple[int, int]):
        wx, wy = worker_pos
        lk = np.zeros(NUM_CELLS, dtype=np.float64)
        for i in range(NUM_CELLS):
            x, y = cell_to_xy(i)
            true_dist = abs(wx - x) + abs(wy - y)
            if reported_dist == 0:
                lk[i] = sum(DIST_OFFSETS[o] for o in DIST_OFFSETS
                            if true_dist + o <= 0)
            else:
                lk[i] = DIST_OFFSETS.get(reported_dist - true_dist, 0.0)
        self.belief *= lk
        self._normalize()

    def update_search(self, searched_pos: Tuple[int, int], found: bool):
        if found:
            self.belief = self._spawn_dist.copy()
        else:
            self.belief[xy_to_cell(*searched_pos)] = 0.0
            self._normalize()

    def best_cell(self) -> Tuple[Tuple[int, int], float, float]:
        idx = int(np.argmax(self.belief))
        p   = float(self.belief[idx])
        return cell_to_xy(idx), p, 6.0 * p - 2.0

    def top_cells(self, k: int) -> List[Tuple[float, Tuple[int, int]]]:
        idxs = np.argsort(self.belief)[::-1][:k]
        return [(float(self.belief[i]), cell_to_xy(int(i))) for i in idxs]

    def inverse_distance_heat(self, pos: Tuple[int, int]) -> float:
        return sum(p / (1.0 + manhattan(pos, cell)) for p, cell in self.top_cells(8))

    def expected_distance(self, pos: Tuple[int, int]) -> float:
        top = self.top_cells(8)
        weight = sum(p for p, _ in top)
        if weight <= 1e-12: return 0.0
        return sum(p * manhattan(pos, cell) for p, cell in top) / weight

    def _normalize(self):
        total = self.belief.sum()
        if total > 1e-12:
            self.belief /= total
        else:
            self.belief = self._spawn_dist.copy()


# ===========================================================================
# Direction helpers
# ===========================================================================

_DIRECTION_DELTAS = {
    enums.Direction.UP:    (0, -1),
    enums.Direction.DOWN:  (0,  1),
    enums.Direction.LEFT:  (-1, 0),
    enums.Direction.RIGHT: (1,  0),
}

def _move_destination(mv, current_pos: Tuple[int, int]) -> Tuple[int, int]:
    dx, dy = _DIRECTION_DELTAS.get(mv.direction, (0, 0))
    steps  = mv.roll_length if mv.move_type == MoveType.CARPET else 1
    return (current_pos[0] + dx * steps, current_pos[1] + dy * steps)


# ===========================================================================
# Board geometry helpers
# ===========================================================================

def _run_length_primed(loc: Tuple[int, int], dx: int, dy: int,
                       board_state) -> int:
    """Contiguous primed cells from loc (exclusive) in direction dx,dy."""
    count = 0
    nx, ny = loc[0] + dx, loc[1] + dy
    while board_state.is_valid_cell((nx, ny)):
        bit = 1 << xy_to_cell(nx, ny)
        if board_state._primed_mask & bit:
            count += 1; nx += dx; ny += dy
        else:
            break
    return count


def _run_length_open(loc: Tuple[int, int], dx: int, dy: int,
                     board_state) -> int:
    """Contiguous open (non-blocked, non-carpet) cells from loc in direction dx,dy."""
    count = 0
    nx, ny = loc[0] + dx, loc[1] + dy
    while board_state.is_valid_cell((nx, ny)):
        bit = 1 << xy_to_cell(nx, ny)
        if (board_state._blocked_mask | board_state._carpet_mask) & bit:
            break
        count += 1; nx += dx; ny += dy
    return count


def _adjacent_primed_chain(loc: Tuple[int, int], board_state) -> int:
    best = 0
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        best = max(best, _run_length_primed(loc, dx, dy, board_state))
    return best


def _max_carpet_length(loc: Tuple[int, int], board_state) -> int:
    best       = 0
    opp_loc    = board_state.opponent_worker.get_location()
    player_loc = board_state.player_worker.get_location()
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        length = 0
        nx, ny = loc[0]+dx, loc[1]+dy
        while board_state.is_valid_cell((nx, ny)):
            if (nx, ny) in (opp_loc, player_loc): break
            bit = 1 << xy_to_cell(nx, ny)
            if board_state._primed_mask & bit:
                length += 1; nx += dx; ny += dy
            else:
                break
        best = max(best, length)
    return best


def _max_carpet_potential(loc: Tuple[int, int], board_state) -> int:
    best       = 0
    opp_loc    = board_state.opponent_worker.get_location()
    player_loc = board_state.player_worker.get_location()
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        length = 0
        nx, ny = loc[0]+dx, loc[1]+dy
        while board_state.is_valid_cell((nx, ny)):
            if (nx, ny) in (opp_loc, player_loc): break
            bit = 1 << xy_to_cell(nx, ny)
            if board_state._primed_mask & bit:
                length += 1; nx += dx; ny += dy
            else:
                break
        if length > 0:
            best = max(best, CARPET_SCORE.get(length, 0))
    return best


def _best_carpet_end(loc: Tuple[int, int], board_state) -> Tuple[int, int]:
    best_end   = loc
    max_score  = 0
    opp_loc    = board_state.opponent_worker.get_location()
    player_loc = board_state.player_worker.get_location()
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        length = 0
        nx, ny = loc[0]+dx, loc[1]+dy
        while board_state.is_valid_cell((nx, ny)):
            if (nx, ny) in (opp_loc, player_loc): break
            bit = 1 << xy_to_cell(nx, ny)
            if board_state._primed_mask & bit:
                length += 1
                s = CARPET_SCORE.get(length, 0)
                if s > max_score:
                    max_score = s
                    best_end  = (nx, ny)
                nx += dx; ny += dy
            else:
                break
    return best_end


def _future_chain_potential(loc: Tuple[int, int], board_state) -> int:
    best       = 0
    opp_loc    = board_state.opponent_worker.get_location()
    player_loc = board_state.player_worker.get_location()
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        length = 0
        nx, ny = loc[0]+dx, loc[1]+dy
        while board_state.is_valid_cell((nx, ny)):
            if (nx, ny) in (opp_loc, player_loc): break
            bit = 1 << xy_to_cell(nx, ny)
            if (board_state._blocked_mask | board_state._carpet_mask) & bit: break
            length += 1; nx += dx; ny += dy
        if length > 0:
            best = max(best, CARPET_SCORE.get(min(length, 7), 0))
    return best


def _chain_continuation_bonus(dest: Tuple[int, int], dx: int, dy: int,
                               board_state) -> float:
    return _run_length_primed(dest, dx, dy, board_state) * 3.0


def _cell_potential(x: int, y: int, board_state) -> int:
    bit = 1 << (y * BOARD_SIZE + x)
    if board_state._blocked_mask & bit: return 0
    best = 0
    for dx, dy in [(0,1),(1,0)]:
        length = 1
        for sign in (1, -1):
            nx, ny = x + sign*dx, y + sign*dy
            while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                if board_state._blocked_mask & (1 << (ny * BOARD_SIZE + nx)): break
                length += 1; nx += sign*dx; ny += sign*dy
        best = max(best, length)
    return min(best, 7)


# ===========================================================================
# Carpet timing model
# ===========================================================================

def _carpet_timing_value(loc: Tuple[int, int], board_state,
                         turns_left: int) -> float:
    """
    Risk-adjusted value of the best carpetable line from loc.

    For each direction:
    - Computes the immediate cash value (current primed run)
    - Discounts it by steal risk (opponent proximity to chain end)
    - Compares against extension value, accounting for turns needed + steal risk

    This replaces the naive 'max carpet score' which always rewards longer chains
    without penalising the cost of waiting to extend them.
    """
    opp_loc    = board_state.opponent_worker.get_location()
    player_loc = board_state.player_worker.get_location()
    best_val   = 0.0

    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        # Count primed and total open cells in this direction
        primed = 0
        open_  = 0
        nx, ny = loc[0]+dx, loc[1]+dy
        counting_primed = True
        while board_state.is_valid_cell((nx, ny)):
            if (nx, ny) in (opp_loc, player_loc): break
            bit = 1 << xy_to_cell(nx, ny)
            if (board_state._blocked_mask | board_state._carpet_mask) & bit:
                break
            open_ += 1
            if counting_primed:
                if board_state._primed_mask & bit:
                    primed += 1
                else:
                    counting_primed = False
            nx += dx; ny += dy

        if open_ == 0:
            continue

        # Opponent's distance to end of current primed run
        end_of_primed = (loc[0] + dx * primed, loc[1] + dy * primed)
        opp_dist      = manhattan(opp_loc, end_of_primed)

        # Steal risk: how likely is opp to intercept before we cash?
        steal_risk = max(0.0, 1.0 - opp_dist / 5.0)

        # Value if we cash right now
        immediate_val  = max(0, CARPET_SCORE.get(primed, 0)) if primed >= 1 else 0
        safe_immediate = immediate_val * (1.0 - steal_risk * 0.65)

        # Value if we extend to full open run
        if open_ > primed:
            turns_to_finish = (open_ - primed) + 1  # extra primes + carpet turn
            if turns_left >= turns_to_finish + 1:
                extended_val  = CARPET_SCORE.get(min(open_, 7), 0)
                # Steal risk grows as we take more turns to finish
                extended_risk = min(0.95, steal_risk + (open_ - primed) * 0.15)
                safe_extended = extended_val * (1.0 - extended_risk * 0.65)
                best_val = max(best_val, safe_extended)

        best_val = max(best_val, safe_immediate)

    return best_val


# ===========================================================================
# Primed cell ownership weighting
# ===========================================================================

def _ownership_weight(cell: Tuple[int, int], my_loc: Tuple[int, int],
                      opp_loc: Tuple[int, int]) -> float:
    """
    Fraction of a primed cell's value that belongs to us.
    1.0 = only we can realistically reach it, 0.5 = equidistant.
    """
    d_me  = manhattan(my_loc, cell)
    d_opp = manhattan(opp_loc, cell)
    total = d_me + d_opp
    if total == 0: return 0.5
    return d_opp / total  # closer = higher ownership


def _owned_primed_value(my_loc: Tuple[int, int], opp_loc: Tuple[int, int],
                        board_state) -> float:
    """Sum of primed cells weighted by our ownership of each."""
    total = 0.0
    mask  = board_state._primed_mask
    while mask:
        lsb   = mask & (-mask)
        mask ^= lsb
        idx   = lsb.bit_length() - 1
        cell  = cell_to_xy(idx)
        total += _ownership_weight(cell, my_loc, opp_loc)
    return total


# ===========================================================================
# Static evaluation
# ===========================================================================

def evaluate(board_state, rat_belief: Optional[RatBelief],
             depth_simulated: int = 0,
             last_pos: Optional[Tuple[int, int]] = None) -> float:

    my      = board_state.player_worker
    opp     = board_state.opponent_worker
    my_loc  = my.get_location()
    opp_loc = opp.get_location()
    turns_left = max(1, my.turns_left or 1)

    # --- 1. Actual score differential ---
    score = 50.0 * (my.get_points() - opp.get_points())

    # --- 2. Carpet timing value (steal-risk-adjusted) ---
    my_timing  = _carpet_timing_value(my_loc,  board_state, turns_left)
    opp_timing = _carpet_timing_value(opp_loc, board_state, turns_left)
    score += 18.0 * my_timing
    score -= 18.0 * opp_timing

    # Needed for rat hunting and opponent threat sections below
    my_carpet_len  = _max_carpet_length(my_loc,  board_state)
    opp_carpet_len = _max_carpet_length(opp_loc, board_state)
    worker_dist    = manhattan(my_loc, opp_loc)

    # --- 3. Opponent threat: penalise their ready chains ---
    if opp_carpet_len >= 2 and worker_dist <= 5:
        score -= 7.0 * CARPET_SCORE.get(opp_carpet_len, 0)

    # --- 4. Ownership-weighted primed cell value ---
    # Rewards building lines only we can cash; penalises shared primed space
    score += 3.0 * _owned_primed_value(my_loc, opp_loc, board_state)

    # --- 5. Future chain potential (discounted by turns remaining) ---
    horizon    = max(0.1, turns_left / 40.0)
    my_future  = _future_chain_potential(my_loc,  board_state)
    opp_future = _future_chain_potential(opp_loc, board_state)
    score += 3.0 * my_future  * horizon
    score -= 3.0 * opp_future * horizon

    # --- 6. Cell potential ---
    my_pot  = _cell_potential(my_loc[0],  my_loc[1],  board_state)
    opp_pot = _cell_potential(opp_loc[0], opp_loc[1], board_state)
    score += 2.0 * my_pot  * horizon
    score -= 2.0 * opp_pot * horizon

    # --- 7. Stranded primed cells penalty ---
    total_primed = bin(board_state._primed_mask).count('1')
    if turns_left < total_primed:
        score -= (total_primed - turns_left) * 2.5

    # --- 8. Endgame urgency: holding uncashed runs is costly ---
    if turns_left <= ENDGAME_TURNS:
        chain_now = _adjacent_primed_chain(my_loc, board_state)
        if chain_now >= 2:
            urgency = (ENDGAME_TURNS - turns_left + 1) / ENDGAME_TURNS
            score  += urgency * CARPET_SCORE.get(chain_now, 0) * 8.0

    # --- 9. Rat hunting ---
    if rat_belief is not None:
        decay = 0.85 ** depth_simulated
        if my_carpet_len <= 2:
            score += (7.0 * rat_belief.inverse_distance_heat(my_loc)
                      - 0.4 * rat_belief.expected_distance(my_loc)) * decay
        if opp_carpet_len <= 2:
            score -= (7.0 * rat_belief.inverse_distance_heat(opp_loc)
                      - 0.4 * rat_belief.expected_distance(opp_loc)) * decay

    # --- 10. Early-game center control ---
    if turns_left >= 28:
        my_center  = (max(0, 3.5 - abs(my_loc[0]  - 3.5)) +
                      max(0, 3.5 - abs(my_loc[1]  - 3.5)))
        opp_center = (max(0, 3.5 - abs(opp_loc[0] - 3.5)) +
                      max(0, 3.5 - abs(opp_loc[1] - 3.5)))
        score += 1.2 * (my_center - opp_center)

    # --- 11. Soft oscillation penalty ---
    if last_pos is not None and my_loc == last_pos:
        score -= 15.0

    return score


# ===========================================================================
# Move ordering
# ===========================================================================

def quick_score(mv, board_state, rat_belief: Optional[RatBelief]) -> float:
    my_loc = board_state.player_worker.get_location()

    if mv.move_type == MoveType.CARPET:
        roll_score = CARPET_SCORE.get(mv.roll_length, 0)
        if roll_score < 0: return -50.0
        return 100.0 + 8.0 * roll_score

    if mv.move_type == MoveType.PRIME:
        dest   = _move_destination(mv, my_loc)
        dx, dy = _DIRECTION_DELTAS.get(mv.direction, (0, 0))
        return (10.0
                + _future_chain_potential(dest, board_state)
                + _chain_continuation_bonus(dest, dx, dy, board_state))

    if mv.move_type == MoveType.PLAIN:
        dest      = _move_destination(mv, my_loc)
        rat_bonus = 0.0
        if rat_belief is not None and _max_carpet_length(my_loc, board_state) == 0:
            cell, p, _ = rat_belief.best_cell()
            if manhattan(dest, cell) < manhattan(my_loc, cell):
                rat_bonus = p * 5.0
        return 1.0 + 0.5 * _cell_potential(dest[0], dest[1], board_state) + rat_bonus

    return -1000.0


# ===========================================================================
# PlayerAgent
# ===========================================================================

class PlayerAgent:

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.rat_belief: Optional[RatBelief] = None
        self._tm = transition_matrix

        if transition_matrix is not None:
            self.rat_belief = RatBelief(transition_matrix)

        self._turns  = 0
        self._hits   = 0
        self._misses = 0
        self._last_opp_search_turn = -1
        self._last_my_search_turn  = -1
        self._primes_done   = 0
        self._carpets_made  = 0
        self._last_turns_remaining: Optional[int] = None
        self._last_pos: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------

    def commentate(self):
        rats_searched = self._hits + self._misses
        stats = (f"Primes: {self._primes_done} | Carpets: {self._carpets_made} | "
                 f"Rats Found: {self._hits} | Searched: {rats_searched} | "
                 f"Pts Lost: {self._misses * 2}")
        if self.rat_belief is not None:
            cell, p, ev = self.rat_belief.best_cell()
            return f"Turns: {self._turns} | {stats} | Peak: {cell} p={p:.3f} EV={ev:.2f}"
        return f"Turns: {self._turns} | {stats} | (no HMM)"

    # ------------------------------------------------------------------

    def _ret(self, mv, my_loc):
        """Record stats and last position, return move."""
        self._last_pos = my_loc
        if mv is None:
            rb = self.rat_belief
            rat_cell = (rb.best_cell()[0] if rb is not None
                        else (random.randint(0, 7), random.randint(0, 7)))
            return move.Move.search(rat_cell)
        if mv.move_type == MoveType.CARPET: self._carpets_made += 1
        elif mv.move_type == MoveType.PRIME: self._primes_done += 1
        return mv

    # ------------------------------------------------------------------
    # Negamax with alpha-beta + PVS
    # ------------------------------------------------------------------

    def _negamax(self, board_state, depth: int,
                 rat_belief: Optional[RatBelief],
                 alpha: float, beta: float, end_time: float,
                 depth_simulated: int = 0,
                 last_pos: Optional[Tuple[int, int]] = None) -> float:

        if time.time() > end_time:
            raise SearchTimeout()

        if depth == 0 or (board_state.player_worker.turns_left or 0) == 0:
            return evaluate(board_state, rat_belief, depth_simulated, last_pos)

        moves = list(board_state.get_valid_moves(exclude_search=True))
        if not moves:
            return evaluate(board_state, rat_belief, depth_simulated, last_pos)

        moves.sort(key=lambda m: quick_score(m, board_state, rat_belief), reverse=True)
        if depth <= 1:   moves = moves[:8]
        elif depth <= 2: moves = moves[:12]
        else:            moves = moves[:16]

        best           = -float("inf")
        cur_player_pos = board_state.player_worker.get_location()

        for i, mv in enumerate(moves):
            if time.time() > end_time:
                raise SearchTimeout()
            child = board_state.forecast_move(mv)
            if child is None: continue
            child.reverse_perspective()

            if i == 0:
                val = -self._negamax(child, depth-1, rat_belief,
                                     -beta, -alpha, end_time,
                                     depth_simulated+1,
                                     last_pos=cur_player_pos)
            else:
                val = -self._negamax(child, depth-1, rat_belief,
                                     -alpha-1, -alpha, end_time,
                                     depth_simulated+1,
                                     last_pos=cur_player_pos)
                if alpha < val < beta:
                    val = -self._negamax(child, depth-1, rat_belief,
                                         -beta, -alpha, end_time,
                                         depth_simulated+1,
                                         last_pos=cur_player_pos)

            if val > best: best = val
            alpha = max(alpha, best)
            if alpha >= beta: break

        if best == -float("inf"):
            best = evaluate(board_state, rat_belief, depth_simulated, last_pos)

        return best

    # ------------------------------------------------------------------
    # Main play method
    # ------------------------------------------------------------------

    def play(self, board: board.Board, sensor_data: Tuple, time_left: Callable):
        my_turns = board.player_worker.turns_left

        # Detect new game
        if self._last_turns_remaining is None or my_turns > self._last_turns_remaining:
            self._turns = 0
            self._last_turns_remaining = None
            self._last_pos = None
            if self._tm is not None:
                self.rat_belief = RatBelief(self._tm)

        self._last_turns_remaining = my_turns - 1
        self._turns += 1

        # Lazy-init rat belief from board if not provided in constructor
        if self.rat_belief is None:
            tm = self._tm
            if tm is None:
                try:    tm = board.transition_matrix
                except AttributeError: pass
            if tm is not None:
                self.rat_belief = RatBelief(tm)

        rb    = self.rat_belief
        noise, reported_dist = sensor_data

        # --- HMM update ---
        if rb is not None:
            opp_loc, opp_found = board.opponent_search
            if opp_loc is not None and self._turns != self._last_opp_search_turn:
                rb.update_search(opp_loc, opp_found)
                self._last_opp_search_turn = self._turns

            my_search_loc, my_found = board.player_search
            if my_search_loc is not None and self._turns != self._last_my_search_turn:
                if my_found: self._hits += 1
                else:        self._misses += 1
                rb.update_search(my_search_loc, my_found)
                self._last_my_search_turn = self._turns

            if self._turns > 1:
                rb.predict()
                if not opp_found:
                    rb.predict()
            elif self._turns == 1:
                rb.predict()
                if board.player_worker.is_player_b:
                    rb.predict()

            if noise is not None:
                rb.update_noise(noise, board)
            if reported_dist is not None:
                try:
                    rb.update_distance(int(reported_dist),
                                       board.player_worker.get_location())
                except Exception:
                    pass

        moves      = list(board.get_valid_moves(exclude_search=True))
        turns_left = max(1, board.player_worker.turns_left)
        my_loc     = board.player_worker.get_location()
        my_score   = board.player_worker.get_points()
        opp_score  = board.opponent_worker.get_points()

        # ---------------------------------------------------------------
        # ENDGAME PHASE
        # Cash everything available; stop priming unless we can finish
        # ---------------------------------------------------------------
        if turns_left <= ENDGAME_TURNS:
            # Best available carpet
            carpet_moves = [m for m in moves
                            if m.move_type == MoveType.CARPET
                            and CARPET_SCORE.get(m.roll_length, -1) >= 2]
            if carpet_moves:
                best = max(carpet_moves,
                           key=lambda m: CARPET_SCORE.get(m.roll_length, -1))
                return self._ret(best, my_loc)

            # Rat search if high EV (more aggressive when losing)
            if rb is not None:
                rat_cell, rat_p, rat_ev = rb.best_cell()
                threshold = 0.8 if my_score < opp_score else 1.2
                if rat_ev >= threshold:
                    self._last_pos = my_loc
                    return move.Move.search(rat_cell)

            # Only prime if we have enough turns to carpet the result
            chain_now   = _adjacent_primed_chain(my_loc, board)
            prime_moves = [m for m in moves if m.move_type == MoveType.PRIME]
            if prime_moves and chain_now < turns_left - 1:
                best_prime = max(prime_moves,
                                 key=lambda m: quick_score(m, board, rb))
                return self._ret(best_prime, my_loc)

            # Plain move — move toward best carpet opportunity
            plain_moves = [m for m in moves if m.move_type == MoveType.PLAIN]
            if plain_moves:
                plain_moves.sort(key=lambda m: quick_score(m, board, rb), reverse=True)
                return self._ret(plain_moves[0], my_loc)

            return self._ret(random.choice(moves) if moves else None, my_loc)

        # ---------------------------------------------------------------
        # OPPORTUNISTIC RAT SEARCH
        # Dynamic EV threshold; don't interrupt a live carpet run >= 3
        # ---------------------------------------------------------------
        if rb is not None:
            rat_cell, rat_p, rat_ev = rb.best_cell()

            if   my_score < opp_score - 5: ev_threshold = 1.6
            elif my_score < opp_score:     ev_threshold = 1.8
            elif my_score > opp_score + 5: ev_threshold = 2.6
            else:                          ev_threshold = 2.2

            if   turns_left <= 10: ev_threshold -= 0.2
            elif turns_left <= 15: ev_threshold -= 0.1

            if rat_ev >= ev_threshold:
                my_carpet_len = _max_carpet_length(my_loc, board)
                if my_carpet_len < 3:
                    self._last_pos = my_loc
                    return move.Move.search(rat_cell)

        # ---------------------------------------------------------------
        # ITERATIVE-DEEPENING NEGAMAX WITH PVS
        # ---------------------------------------------------------------
        if moves:
            start_time  = time.time()
            safe_buffer = 1.0
            usable_time = max(0.1, time_left() - safe_buffer)
            allocated   = min(5.0, usable_time / max(1, turns_left))
            end_time    = start_time + allocated

            # Root ordering: 1-ply evaluate + carpet urgency bonus
            root_scored = []
            for mv in moves:
                child = board.forecast_move(mv)
                if child is None: continue
                child.reverse_perspective()
                extra = 0.0
                if mv.move_type == MoveType.CARPET:
                    chain_len = _adjacent_primed_chain(my_loc, board)
                    extra = 4.0 * CARPET_SCORE.get(chain_len, 0)
                root_scored.append(
                    (-evaluate(child, rb, 0, self._last_pos) + extra, mv))

            root_scored.sort(key=lambda x: x[0], reverse=True)
            moves_ordered    = [m for _, m in root_scored] or moves
            global_best_move = moves_ordered[0]

            try:
                for depth in range(1, 15):
                    if time.time() > end_time:
                        break

                    alpha = -float("inf")
                    beta  =  float("inf")
                    best_val_this_depth = -float("inf")

                    # Always search previous best first (iterative deepening trick)
                    if global_best_move in moves_ordered:
                        moves_ordered.remove(global_best_move)
                        moves_ordered.insert(0, global_best_move)

                    for i, mv in enumerate(moves_ordered):
                        if time.time() > end_time:
                            raise SearchTimeout()
                        child = board.forecast_move(mv)
                        if child is None: continue
                        child.reverse_perspective()

                        if i == 0:
                            val = -self._negamax(child, depth-1, rb,
                                                 -beta, -alpha, end_time,
                                                 last_pos=self._last_pos)
                        else:
                            val = -self._negamax(child, depth-1, rb,
                                                 -alpha-1, -alpha, end_time,
                                                 last_pos=self._last_pos)
                            if alpha < val < beta:
                                val = -self._negamax(child, depth-1, rb,
                                                     -beta, -alpha, end_time,
                                                     last_pos=self._last_pos)

                        if val > best_val_this_depth:
                            best_val_this_depth = val
                            global_best_move    = mv
                        alpha = max(alpha, best_val_this_depth)

            except SearchTimeout:
                pass

            return self._ret(global_best_move, my_loc)

        # Fallback: greedy
        if moves:
            best_mv = max(moves, key=lambda m: quick_score(m, board, rb))
            return self._ret(best_mv, my_loc)

        self._last_pos = my_loc
        rb_cell = (rb.best_cell()[0] if rb is not None
                   else (random.randint(0, 7), random.randint(0, 7)))
        return move.Move.search(rb_cell)