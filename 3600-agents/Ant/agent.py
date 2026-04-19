from collections.abc import Callable
from typing import List, Set, Tuple
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

class SearchTimeout(Exception):
    pass


# ===========================================================================
# RatBelief — Hidden Markov Model
# ===========================================================================

class RatBelief:
    """
    Update order per turn:
      1. update_search()   — apply search result (before rat moved this turn)
      2. predict() x2      — rat moved before opponent's turn + before ours
      3. update_noise()    — reweight using noise observation
      4. update_distance() — reweight using noisy distance sensor
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

    def top_cells(self, k: int) -> List[Tuple[float, Tuple[int, int]]]:
        idxs = np.argsort(self.belief)[::-1][:k]
        return [(float(self.belief[i]), cell_to_xy(int(i))) for i in idxs]

    def inverse_distance_heat(self, pos: Tuple[int, int]) -> float:
        top = self.top_cells(8)
        return sum(p / (1.0 + manhattan(pos, cell)) for p, cell in top)

    def expected_distance(self, pos: Tuple[int, int]) -> float:
        top = self.top_cells(8)
        if not top: return 0.0
        weight = sum(p for p, _ in top)
        if weight <= 1e-12: return 0.0
        return sum(p * manhattan(pos, cell) for p, cell in top) / weight

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
# Heuristic helpers
# ===========================================================================

def _max_carpet_potential(loc, board_state) -> int:
    """Best immediate carpet roll score from loc."""
    max_score  = 0
    enemy_loc  = board_state.opponent_worker.get_location()
    player_loc = board_state.player_worker.get_location()
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        length = 0
        nx, ny = loc[0]+dx, loc[1]+dy
        while board_state.is_valid_cell((nx, ny)):
            if (nx, ny) in (enemy_loc, player_loc): break
            bit = 1 << (ny * BOARD_SIZE + nx)
            if board_state._primed_mask & bit:
                length += 1; nx += dx; ny += dy
            else:
                break
        if length > 0:
            max_score = max(max_score, CARPET_SCORE.get(length, 0))
    return max_score


def _max_carpet_length(loc, board_state) -> int:
    """Raw length of the best carpet roll available."""
    best_len   = 0
    enemy_loc  = board_state.opponent_worker.get_location()
    player_loc = board_state.player_worker.get_location()
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        length = 0
        nx, ny = loc[0]+dx, loc[1]+dy
        while board_state.is_valid_cell((nx, ny)):
            if (nx, ny) in (enemy_loc, player_loc): break
            bit = 1 << (ny * BOARD_SIZE + nx)
            if board_state._primed_mask & bit:
                length += 1; nx += dx; ny += dy
            else:
                break
        best_len = max(best_len, length)
    return best_len


def _best_carpet_end(loc: Tuple[int, int], board_state) -> Tuple[int, int]:
    """End cell of the highest-scoring carpet roll from loc."""
    best_end   = loc
    max_score  = 0
    enemy_loc  = board_state.opponent_worker.get_location()
    player_loc = board_state.player_worker.get_location()
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        length = 0
        nx, ny = loc[0]+dx, loc[1]+dy
        while board_state.is_valid_cell((nx, ny)):
            if (nx, ny) in (enemy_loc, player_loc): break
            bit = 1 << (ny * BOARD_SIZE + nx)
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
    """
    Max carpet score achievable if all open+primed cells in one direction
    from loc are eventually primed. Carpet squares are treated as blockers.
    """
    best       = 0
    enemy_loc  = board_state.opponent_worker.get_location()
    player_loc = board_state.player_worker.get_location()
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        length = 0
        nx, ny = loc[0]+dx, loc[1]+dy
        while board_state.is_valid_cell((nx, ny)):
            if (nx, ny) in (enemy_loc, player_loc): break
            bit = 1 << (ny * BOARD_SIZE + nx)
            if (board_state._blocked_mask | board_state._carpet_mask) & bit: break
            length += 1; nx += dx; ny += dy
        if length > 0:
            best = max(best, CARPET_SCORE.get(min(length, 7), 0))
    return best


def _chain_continuation_bonus(dest: Tuple[int, int], dx: int, dy: int,
                               board_state) -> float:
    count = 0
    nx, ny = dest[0]+dx, dest[1]+dy
    while board_state.is_valid_cell((nx, ny)):
        bit = 1 << (ny * BOARD_SIZE + nx)
        if board_state._primed_mask & bit:
            count += 1; nx += dx; ny += dy
        else:
            break
    return count * 3.0


def _adjacent_primed_chain(loc, board_state) -> int:
    """Longest contiguous primed run adjacent to loc in any direction."""
    best = 0
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        length = 0
        nx, ny = loc[0]+dx, loc[1]+dy
        while board_state.is_valid_cell((nx, ny)):
            bit = 1 << xy_to_cell(nx, ny)
            if board_state._primed_mask & bit:
                length += 1; nx += dx; ny += dy
            else:
                break
        best = max(best, length)
    return best


def _cell_potential(x: int, y: int, board_state) -> int:
    """Max straight-line run of non-blocked cells through (x,y)."""
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
# Static evaluation  (flood fill removed — cell_potential covers it cheaply)
# ===========================================================================

def evaluate(board_state, rat_belief, depth_simulated: int = 0,
             last_pos: Tuple[int, int] | None = None) -> float:
    my      = board_state.player_worker
    opp     = board_state.opponent_worker
    my_loc  = my.get_location()
    opp_loc = opp.get_location()
    turns_left = max(1, my.turns_left or 1)

    # --- 1. Actual score differential ---
    score = 50.0 * (my.get_points() - opp.get_points())

    # --- 2. Carpet potential ---
    my_carpet      = _max_carpet_potential(my_loc,  board_state)
    opp_carpet     = _max_carpet_potential(opp_loc, board_state)
    my_carpet_len  = _max_carpet_length(my_loc,  board_state)
    opp_carpet_len = _max_carpet_length(opp_loc, board_state)
    worker_dist    = manhattan(my_loc, opp_loc)

    if worker_dist <= 2:   carpet_weight = 4.0
    elif worker_dist <= 4: carpet_weight = 14.0
    else:                  carpet_weight = 20.0

    score += carpet_weight * my_carpet
    score -= carpet_weight * opp_carpet

    # Superlinear bonus for any carpetable run >= 2, scaling with actual score.
    # This makes the search prefer cashing sooner over extending indefinitely.
    chain_now = _adjacent_primed_chain(my_loc, board_state)
    if chain_now >= 2:
        chain_score = CARPET_SCORE.get(chain_now, 0)
        if chain_score > 0:
            # Urgency grows as opponent gets closer — steal risk
            opp_dist_to_chain = manhattan(opp_loc, _best_carpet_end(my_loc, board_state))
            steal_urgency = max(1.0, 4.0 - opp_dist_to_chain * 0.5)
            score += steal_urgency * chain_score * 8.0

    if chain_now >= 4:
        score -= 6.0 * _future_chain_potential(my_loc, board_state)

    # --- 3. Steal detection: opponent adjacent to our carpetable run ---
    if my_carpet_len >= 2:
        opp_dist_to_chain = manhattan(opp_loc, _best_carpet_end(my_loc, board_state))
        steal_prob = max(0.0, 1.0 - opp_dist_to_chain / 5.0)
        score -= steal_prob * my_carpet

    if opp_carpet_len >= 2 and worker_dist <= 4:
        score -= 8.0 * CARPET_SCORE.get(opp_carpet_len, 0)

    # --- 4. Future chain potential ---
    horizon = max(0.1, turns_left / 40.0)
    score += 3.5 * _future_chain_potential(my_loc,  board_state) * horizon
    score -= 3.5 * _future_chain_potential(opp_loc, board_state) * horizon

    # --- 5. Cell potential ---
    my_pot  = _cell_potential(my_loc[0],  my_loc[1],  board_state)
    opp_pot = _cell_potential(opp_loc[0], opp_loc[1], board_state)
    score += 2.5 * my_pot  * horizon
    score -= 2.5 * opp_pot * horizon

    # --- 6. Stranded primed cells penalty ---
    total_primed = bin(board_state._primed_mask).count('1')
    if turns_left < total_primed:
        score -= (total_primed - turns_left) * 2.0

    # --- 7. Rat hunting ---
    if rat_belief is not None:
        decay = 0.85 ** depth_simulated
        if my_carpet_len <= 2:
            my_heat = rat_belief.inverse_distance_heat(my_loc)
            my_dist = rat_belief.expected_distance(my_loc)
            score += (7.0 * my_heat - 0.4 * my_dist) * decay
        if opp_carpet_len <= 2:
            opp_heat = rat_belief.inverse_distance_heat(opp_loc)
            opp_dist = rat_belief.expected_distance(opp_loc)
            score -= (7.0 * opp_heat - 0.4 * opp_dist) * decay

    # --- 8. Early-game center control ---
    if turns_left >= 25:
        my_center  = (max(0, 3.5 - abs(my_loc[0]  - 3.5)) +
                      max(0, 3.5 - abs(my_loc[1]  - 3.5)))
        opp_center = (max(0, 3.5 - abs(opp_loc[0] - 3.5)) +
                      max(0, 3.5 - abs(opp_loc[1] - 3.5)))
        score += 1.0 * (my_center - opp_center)

    # --- 9. Soft oscillation penalty ---
    # Discourage ending up back where we just were, without hard-blocking it.
    if last_pos is not None and my_loc == last_pos:
        score -= 5.0

    return score


def _primed_run_in_direction(start: Tuple[int,int], dx: int, dy: int,
                              board_state) -> int:
    """Count contiguous primed cells from start (exclusive) in direction dx,dy."""
    count = 0
    nx, ny = start[0] + dx, start[1] + dy
    while board_state.is_valid_cell((nx, ny)):
        bit = 1 << xy_to_cell(nx, ny)
        if board_state._primed_mask & bit:
            count += 1; nx += dx; ny += dy
        else:
            break
    return count


def get_active_line(loc: Tuple[int, int], board_state) -> Tuple[str, int]:
    """
    Returns (axis, length) where axis is 'H', 'V', or None.
    Only commits to an axis once it reaches 2+ cells.
    """
    h_len = (_primed_run_in_direction(loc, -1,  0, board_state) +
             _primed_run_in_direction(loc,  1,  0, board_state))
    v_len = (_primed_run_in_direction(loc,  0, -1, board_state) +
             _primed_run_in_direction(loc,  0,  1, board_state))

    if h_len >= v_len and h_len >= 2:
        return 'H', h_len
    if v_len > h_len and v_len >= 2:
        return 'V', v_len
    return None, max(h_len, v_len)


def prime_axis_penalty(mv, loc: Tuple[int, int], board_state) -> float:
    """
    Penalises a prime move that abandons the active line.
    Penalty scales with how many cells are already committed —
    the longer the existing line, the more it costs to defect.
    Returns 0 when there is no committed line yet.
    """
    if mv.move_type != MoveType.PRIME:
        return 0.0

    active_axis, committed_len = get_active_line(loc, board_state)
    if active_axis is None:
        return 0.0

    dx, dy = _DIRECTION_DELTAS.get(mv.direction, (0, 0))
    move_axis = 'H' if dy == 0 else 'V'

    if move_axis == active_axis:
        return 0.0  # Staying on the same axis — good

    # Defecting: penalty grows with committed work being abandoned
    return -15.0 * committed_len


# ===========================================================================
# Move ordering
# ===========================================================================

def quick_score(mv, board_state, rat_belief) -> float:
    if mv.move_type == MoveType.CARPET:
        roll_score = CARPET_SCORE.get(mv.roll_length, 0)
        if roll_score < 0: return -50.0
        return 200.0 + 10 * roll_score

    if mv.move_type == MoveType.PRIME:
        my_loc = board_state.player_worker.get_location()
        dest   = _move_destination(mv, my_loc)
        dx, dy = _DIRECTION_DELTAS.get(mv.direction, (0, 0))
        axis_pen = prime_axis_penalty(mv, my_loc, board_state)
        chain_len = _adjacent_primed_chain(dest, board_state) 
        if chain_len >= 3:
            axis_pen -= 10.0 * CARPET_SCORE.get(chain_len, 0)
        return (10.0
                + _future_chain_potential(dest, board_state)
                + _chain_continuation_bonus(dest, dx, dy, board_state)
                + axis_pen)

    if mv.move_type == MoveType.PLAIN:
        my_loc = board_state.player_worker.get_location()
        dest   = _move_destination(mv, my_loc)
        return 1.0 + 0.5 * _cell_potential(dest[0], dest[1], board_state)

    return -1000.0


# ===========================================================================
# PlayerAgent
# ===========================================================================

class PlayerAgent:

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.rat_belief = None
        self._tm = transition_matrix

        if transition_matrix is not None:
            self.rat_belief = RatBelief(transition_matrix)

        self._turns = 0
        self._hits   = 0
        self._misses = 0
        self._last_opp_search_turn = -1
        self._last_my_search_turn  = -1
        self._primes_done    = 0
        self._carpets_made   = 0
        self._last_turns_remaining = None
        self._last_pos: Tuple[int, int] | None = None  # for oscillation detection

    # ------------------------------------------------------------------

    def commentate(self):
        rats_searched = self._hits + self._misses
        points_lost   = self._misses * 2
        stats = (f"Primes: {self._primes_done} | Carpets: {self._carpets_made} | "
                 f"Rats Found: {self._hits} | Searched: {rats_searched} | "
                 f"Pts Lost: {points_lost}")
        if self.rat_belief is not None:
            cell, p, ev = self.rat_belief.best_cell()
            return f"Turns: {self._turns} | {stats} | Peak: {cell} p={p:.3f} EV={ev:.2f}"
        return f"Turns: {self._turns} | {stats} | (no HMM)"

    # ------------------------------------------------------------------

    def _return_and_track(self, mv):
        if mv is None:
            rb = self.rat_belief
            rat_cell = (rb.best_cell()[0] if rb is not None
                        else (random.randint(0,7), random.randint(0,7)))
            return move.Move.search(rat_cell)
        if mv.move_type == MoveType.CARPET: self._carpets_made += 1
        elif mv.move_type == MoveType.PRIME: self._primes_done += 1
        return mv

    # ------------------------------------------------------------------
    # Negamax with alpha-beta + PVS  (TT removed — low hit rate on this board)
    # ------------------------------------------------------------------

    def _negamax(self, board_state, depth: int, rat_belief,
                 alpha: float, beta: float, end_time: float,
                 depth_simulated: int = 0,
                 last_pos: Tuple[int, int] | None = None) -> float:

        if time.time() > end_time:
            raise SearchTimeout()

        if depth == 0 or (board_state.player_worker.turns_left or 0) == 0:
            return evaluate(board_state, rat_belief, depth_simulated, last_pos)

        moves = list(board_state.get_valid_moves(exclude_search=True))
        if not moves:
            return evaluate(board_state, rat_belief, depth_simulated, last_pos)

        moves.sort(key=lambda m: quick_score(m, board_state, rat_belief), reverse=True)

        if depth <= 1:   moves = moves[:6]
        elif depth <= 2: moves = moves[:10]
        else:            moves = moves[:12]

        best = -float("inf")

        # The current player's position — children that end here get the oscillation penalty
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
                # Null-window search (PVS)
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

        if self._last_turns_remaining is None or my_turns > self._last_turns_remaining:
            self._turns = 0
            self._last_turns_remaining = None
            self._last_pos = None
            if self._tm is not None:
                self.rat_belief = RatBelief(self._tm)

        self._last_turns_remaining = my_turns - 1
        self._turns += 1

        if self.rat_belief is None:
            tm = self._tm
            if tm is None:
                try:    tm = board.transition_matrix
                except AttributeError: pass
            if tm is not None:
                self.rat_belief = RatBelief(tm)

        rb = self.rat_belief
        noise, reported_dist = sensor_data

        # --- HMM update ---
        if rb is not None:
            opp_loc, opp_found = board.opponent_search
            if opp_loc is not None and self._turns != self._last_opp_search_turn:
                rb.update_search(opp_loc, opp_found)
                self._last_opp_search_turn = self._turns

            my_search_loc, my_found = board.player_search
            if my_search_loc is not None and self._turns != self._last_my_search_turn:
                if my_found:
                    self._hits += 1
                else:
                    self._misses += 1
                rb.update_search(my_search_loc, my_found)
                self._last_my_search_turn = self._turns

            if self._turns > 1:
                rb.predict()
                rb.predict()
            elif self._turns == 1 and board.player_worker.is_player_b:
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
        my_loc_now = board.player_worker.get_location()

        chain_len = _adjacent_primed_chain(my_loc_now, board)
        if chain_len >= 5:
            carpet_moves = [m for m in moves if m.move_type == MoveType.CARPET]
            if carpet_moves:
                return max(carpet_moves, key=lambda m: m.roll_length)

        # --- Last few turns: cash out aggressively ---
        if turns_left <= 3:
            carpet_moves = [m for m in moves
                            if m.move_type == MoveType.CARPET and m.roll_length >= 2]
            if carpet_moves:
                best = max(carpet_moves, key=lambda m: CARPET_SCORE.get(m.roll_length, -1))
                if CARPET_SCORE.get(best.roll_length, 0) >= 2:
                    self._last_pos = my_loc_now
                    return self._return_and_track(best)
            if rb is not None:
                rat_cell, rat_p, rat_ev = rb.best_cell()
                threshold = 1.3 if board.player_worker.get_points() >= board.opponent_worker.get_points() else 1.1
                if rat_ev >= threshold:
                    self._last_pos = my_loc_now
                    return self._return_and_track(move.Move.search(rat_cell))
            for mv in moves:
                if mv.move_type == MoveType.PRIME:
                    self._last_pos = my_loc_now
                    return self._return_and_track(mv)
            plain = [m for m in moves if m.move_type == MoveType.PLAIN]
            if plain:
                plain.sort(key=lambda m: quick_score(m, board, rb), reverse=True)
            self._last_pos = my_loc_now
            return self._return_and_track(plain[0] if plain else random.choice(moves))

        # --- Opportunistic search with dynamic thresholds ---
        if rb is not None:
            rat_cell, rat_p, rat_ev = rb.best_cell()
            my_score  = board.player_worker.get_points()
            opp_score = board.opponent_worker.get_points()

            if   my_score < opp_score - 5: ev_threshold = 1.65
            elif my_score < opp_score:     ev_threshold = 1.8
            elif my_score > opp_score + 5: ev_threshold = 2.8
            else:                          ev_threshold = 2.4

            if   turns_left <= 5:  ev_threshold -= 0.25
            elif turns_left <= 10: ev_threshold -= 0.15

            if rat_ev >= ev_threshold:
                my_carpet_len = _max_carpet_length(my_loc_now, board)
                if my_carpet_len < 3:
                    self._last_pos = my_loc_now
                    return move.Move.search(rat_cell)

        # --- Iterative-deepening negamax with PVS ---
        if moves:
            start_time  = time.time()
            safe_buffer = 1.0
            usable_time = max(0.1, time_left() - safe_buffer)

            # Simplified time allocation: single formula
            allocated = min(6.5, usable_time / max(1, turns_left))
            end_time  = start_time + allocated

            # 1-ply root ordering with carpet urgency bonus
            root_scored = []
            for mv in moves:
                child = board.forecast_move(mv)
                if child is None: continue
                child.reverse_perspective()
                extra = 0.0
                if mv.move_type == MoveType.CARPET:
                    chain_len = _adjacent_primed_chain(my_loc_now, board)
                    extra = 3.0 * CARPET_SCORE.get(chain_len, 0)
                root_scored.append((-evaluate(child, rb, 0, self._last_pos) + extra, mv))

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

            self._last_pos = my_loc_now
            return self._return_and_track(global_best_move)

        # --- Greedy fallback (simplified: just use quick_score) ---
        if moves:
            best_mv = max(moves, key=lambda m: quick_score(m, board, rb))
            self._last_pos = my_loc_now
            return self._return_and_track(best_mv)

        self._last_pos = my_loc_now
        rb_cell = (rb.best_cell()[0] if rb is not None
                   else (random.randint(0,7), random.randint(0,7)))
        return move.Move.search(rb_cell)