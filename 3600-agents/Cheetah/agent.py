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

# Transposition table bound types
TT_EXACT      = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2

MISS_COOLDOWN_TURNS = 2


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

class timeout(Exception):
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
        self.belief += 0.0001 / NUM_CELLS
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

    def top_n_ev(self, n: int = 3) -> List[Tuple[Tuple[int, int], float, float]]:
        idxs = np.argsort(self.belief)[::-1][:n]
        return [(cell_to_xy(int(i)), float(self.belief[i]), 6.0 * float(self.belief[i]) - 2.0)
                for i in idxs]

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
            self.belief[xy_to_cell(*searched_pos)] = 0.0001
            self._normalize()

    def best_cell(self) -> Tuple[Tuple[int, int], float, float]:
        idx = int(np.argmax(self.belief))
        p   = float(self.belief[idx])
        return cell_to_xy(idx), p, 6.0 * p - 2.0

    def rat_heat(self, worker_pos: Tuple[int, int]) -> float:
        idxs = np.argsort(self.belief)[-5:]
        heat = 0.0
        for i in idxs:
            p = float(self.belief[i])
            if p < 0.05: continue
            heat += p / (1.0 + manhattan(worker_pos, cell_to_xy(int(i))))
        return heat

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
    """Best immediate carpet roll SCORE from loc (primed cells only)."""
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
    """Raw length of the best carpet roll available (not its score)."""
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
    from loc are eventually primed.  Carpet squares are treated as blockers
    (from doc 11 — correct per game rules).
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
            # carpet squares block future chains (doc 11 fix)
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
    """
    Max straight-line run of non-blocked cells through (x,y).
    From doc 10 — the Carrie-style positional value metric.
    """
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


def reachable_space(loc: Tuple[int, int], board_state) -> int:
    """
    Flood-fill reachable space (from doc 11 — more accurate than O(4) check).
    Correctly allows walking on carpet squares per game rules.
    """
    visited = set()
    stack   = [loc]
    while stack:
        cur = stack.pop()
        if cur in visited: continue
        visited.add(cur)
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = cur[0]+dx, cur[1]+dy
            if not board_state.is_valid_cell((nx, ny)): continue
            bit = 1 << xy_to_cell(nx, ny)
            # Only blocked squares are walls — carpet is walkable
            if board_state._blocked_mask & bit: continue
            stack.append((nx, ny))
    return len(visited)


# ===========================================================================
# Static evaluation
# ===========================================================================

def evaluate(board_state, rat_belief: RatBelief, depth_simulated: int = 0) -> float:
    my      = board_state.player_worker
    opp     = board_state.opponent_worker
    my_loc  = my.get_location()
    opp_loc = opp.get_location()
    turns_left = max(1, my.turns_left or 1)

    # --- 1. Actual score differential ---
    score = 30.0 * (my.get_points() - opp.get_points())

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

    # Superlinear bonus for long chains (from doc 10)
    chain_now = _adjacent_primed_chain(my_loc, board_state)
    if chain_now >= 3:
        score += 6.0 * CARPET_SCORE.get(chain_now, 0)

# PANIC: If the enemy is dangerously close to our chain, plummet our score 
    # to force the Alpha-Beta tree to roll the carpet NOW before it gets stolen!
    if my_carpet_len >= 2:
        opp_dist_to_my_chain = manhattan(opp_loc, _best_carpet_end(my_loc, board_state))
        if opp_dist_to_my_chain <= 2:
            score -= 15.0 * CARPET_SCORE.get(my_carpet_len, 0)

    # HUNT/STEAL: If we are close to the ENEMY'S chain, massively boost our score!
    # This acts as a magnet, mathematically drawing your bot to block/steal their hard work.
    if opp_carpet_len >= 2:
        my_dist_to_opp_chain = manhattan(my_loc, _best_carpet_end(opp_loc, board_state))
        if my_dist_to_opp_chain <= 2:
            score += 15.0 * CARPET_SCORE.get(opp_carpet_len, 0)

    # --- 4. Future chain potential (carpet squares block, from doc 11) ---
    horizon = max(0.1, turns_left / 40.0)
    score += 3.5 * _future_chain_potential(my_loc,  board_state) * horizon
    score -= 3.5 * _future_chain_potential(opp_loc, board_state) * horizon

    # --- 5. Cell potential — Carrie-style positional value (from doc 10) ---
    my_pot  = _cell_potential(my_loc[0],  my_loc[1],  board_state)
    opp_pot = _cell_potential(opp_loc[0], opp_loc[1], board_state)
    score += 2.5 * my_pot  * horizon
    score -= 2.5 * opp_pot * horizon

    # --- 6. Reachable space — flood-fill (from doc 11, carpet walkable) ---
    my_space  = reachable_space(my_loc,  board_state)
    opp_space = reachable_space(opp_loc, board_state)
    score += 0.5 * (my_space - opp_space)
    if my_space  <= 4: score -= 15.0   # trapped penalty
    if opp_space <= 4: score += 15.0

    # --- 7. Stranded primed cells penalty (from doc 11) ---
    total_primed = bin(board_state._primed_mask).count('1')
    if turns_left < total_primed:
        score -= (total_primed - turns_left) * 2.0

    # --- 8. Rat hunting ---
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

    # --- 9. Early-game center control (from doc 10) ---
    if turns_left >= 25:
        my_center  = (max(0, 3.5 - abs(my_loc[0]  - 3.5)) +
                      max(0, 3.5 - abs(my_loc[1]  - 3.5)))
        opp_center = (max(0, 3.5 - abs(opp_loc[0] - 3.5)) +
                      max(0, 3.5 - abs(opp_loc[1] - 3.5)))
        score += 1.0 * (my_center - opp_center)

    return score


# ===========================================================================
# Move ordering
# ===========================================================================

def quick_score(mv, board_state, rat_belief: RatBelief) -> float:
    """Fast static ordering — no forecast_move calls."""
    if mv.move_type == MoveType.CARPET:
        roll_score = CARPET_SCORE.get(mv.roll_length, 0)
        if roll_score < 0: return -50.0   # never prioritise length-1 roll
        return 100.0 + roll_score

    if mv.move_type == MoveType.PRIME:
        my_loc = board_state.player_worker.get_location()
        dest   = _move_destination(mv, my_loc)
        dx, dy = _DIRECTION_DELTAS.get(mv.direction, (0, 0))
        return (10.0
                + _future_chain_potential(dest, board_state)
                + _chain_continuation_bonus(dest, dx, dy, board_state))

    if mv.move_type == MoveType.PLAIN:
        # Prefer plain moves toward high-potential cells (from doc 10)
        my_loc = board_state.player_worker.get_location()
        dest   = _move_destination(mv, my_loc)
        return 1.0 + 0.5 * _cell_potential(dest[0], dest[1], board_state)

    return -1000.0   # SEARCH excluded from tree


# ===========================================================================
# PlayerAgent
# ===========================================================================

class PlayerAgent:

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.rat_belief: RatBelief | None = None
        self._tm = transition_matrix

        if transition_matrix is not None:
            self.rat_belief = RatBelief(transition_matrix)

        self._turns = 0
        self._hits   = 0
        self._misses = 0
        self._last_opp_search_turn = -1
        self._last_my_search_turn  = -1
        self._miss_cooldown  = 0
        self._primes_done    = 0
        self._carpets_made   = 0
        self._just_caught_rat = False
        self._tt: dict = {}
        self._last_turns_remaining = None

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
    # Negamax with alpha-beta + PVS + proper TT bounds (from doc 10)
    # ------------------------------------------------------------------

    def _negamax(self, board_state, depth: int, rat_belief: RatBelief,
                 alpha: float, beta: float, end_time: float,
                 tt: dict, depth_simulated: int = 0) -> float:

        if time.time() > end_time:
            raise timeout()

        state_key = (
            board_state._primed_mask, board_state._carpet_mask,
            board_state.player_worker.get_location(),
            board_state.opponent_worker.get_location(),
            board_state.player_worker.get_points() - board_state.opponent_worker.get_points(),
            board_state.player_worker.turns_left,
        )

        cached = tt.get(state_key)
        if cached is not None and cached[0] >= depth:
            _, cached_val, cached_flag = cached
            if cached_flag == TT_EXACT:
                return cached_val
            elif cached_flag == TT_LOWERBOUND:
                alpha = max(alpha, cached_val)
            elif cached_flag == TT_UPPERBOUND:
                beta  = min(beta,  cached_val)
            if alpha >= beta:
                return cached_val

        if depth == 0 or (board_state.player_worker.turns_left or 0) == 0:
            val = evaluate(board_state, rat_belief, depth_simulated)
            tt[state_key] = (depth, val, TT_EXACT)
            return val

        moves = list(board_state.get_valid_moves(exclude_search=True))
        if not moves:
            val = evaluate(board_state, rat_belief, depth_simulated)
            tt[state_key] = (depth, val, TT_EXACT)
            return val

        moves.sort(key=lambda m: quick_score(m, board_state, rat_belief), reverse=True)

        if depth <= 1:   moves = moves[:8]
        elif depth <= 2: moves = moves[:12]
        else:            moves = moves[:16]

        orig_alpha = alpha
        best = -float("inf")

        for i, mv in enumerate(moves):
            if time.time() > end_time:
                raise timeout()
            child = board_state.forecast_move(mv)
            if child is None: continue
            child.reverse_perspective()

            if i == 0:
                val = -self._negamax(child, depth-1, rat_belief,
                                     -beta, -alpha, end_time, tt,
                                     depth_simulated+1)
            else:
                # Null-window search (PVS)
                val = -self._negamax(child, depth-1, rat_belief,
                                     -alpha-1, -alpha, end_time, tt,
                                     depth_simulated+1)
                if alpha < val < beta:
                    val = -self._negamax(child, depth-1, rat_belief,
                                         -beta, -alpha, end_time, tt,
                                         depth_simulated+1)

            if val > best: best = val
            alpha = max(alpha, best)
            if alpha >= beta: break

        if best == -float("inf"):
            best = evaluate(board_state, rat_belief, depth_simulated)

        tt_flag = (TT_UPPERBOUND if best <= orig_alpha else
                   TT_LOWERBOUND if best >= beta else TT_EXACT)
        tt[state_key] = (depth, best, tt_flag)
        return best

    # ------------------------------------------------------------------
    # Main play method
    # ------------------------------------------------------------------

    def play(self, board: board.Board, sensor_data: Tuple, time_left: Callable):
        my_turns = board.player_worker.turns_left

        if self._last_turns_remaining is None or my_turns > self._last_turns_remaining:
            self._turns = 0
            self._tt.clear()
            self._miss_cooldown   = 0
            self._just_caught_rat = False
            if self._tm is not None:
                self.rat_belief = RatBelief(self._tm)

        self._last_turns_remaining = my_turns - 1
        self._turns += 1

        if self._miss_cooldown > 0:
            self._miss_cooldown -= 1

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
                    self._just_caught_rat = True
                    self._miss_cooldown   = 0
                else:
                    self._misses += 1
                    self._miss_cooldown = MISS_COOLDOWN_TURNS
                rb.update_search(my_search_loc, my_found)
                self._last_my_search_turn = self._turns

            if self._turns > 1:
                rb.predict()
                rb.predict()

            if noise is not None:
                rb.update_noise(noise, board)
            if reported_dist is not None:
                try:
                    rb.update_distance(int(reported_dist),
                                       board.player_worker.get_location())
                except Exception:
                    pass

        moves      = board.get_valid_moves(exclude_search=True)
        turns_left = max(1, board.player_worker.turns_left)

        # --- Last few turns: cash out aggressively ---
        if turns_left <= 3:
            carpet_moves = [m for m in moves
                            if m.move_type == MoveType.CARPET and m.roll_length >= 2]
            if carpet_moves:
                return self._return_and_track(
                    max(carpet_moves, key=lambda m: CARPET_SCORE.get(m.roll_length, -1)))
            if rb is not None:
                rat_cell, rat_p, rat_ev = rb.best_cell()
                if rat_ev >= 0.9:
                    return self._return_and_track(move.Move.search(rat_cell))
            for mv in moves:
                if mv.move_type == MoveType.PRIME:
                    return self._return_and_track(mv)
            plain = [m for m in moves if m.move_type == MoveType.PLAIN]
            return self._return_and_track(plain[0] if plain else random.choice(moves))

        # --- Opportunistic search with dynamic thresholds (from doc 10) ---
        if rb is not None and self._miss_cooldown == 0:
            rat_cell, rat_p, rat_ev = rb.best_cell()
            my_score  = board.player_worker.get_points()
            opp_score = board.opponent_worker.get_points()

            if   my_score < opp_score - 5:
                ev_threshold = 1.55
            elif my_score < opp_score:
                ev_threshold = 1.7
            elif my_score > opp_score + 5:
                ev_threshold = 2.2
            else:
                ev_threshold = 1.9

            if   turns_left <= 5:
                ev_threshold -= 0.35
            elif turns_left <= 10: 
                ev_threshold -= 0.2

            if rat_ev >= ev_threshold:
                my_loc_now    = board.player_worker.get_location()
                my_carpet_len = _max_carpet_length(my_loc_now, board)
                if my_carpet_len < 3:
                    return move.Move.search(rat_cell)

        # --- Iterative-deepening negamax with PVS ---
        if moves:
            self._tt.clear()
            start_time   = time.time()
            safe_buffer  = 0.5
            usable_time  = max(0.1, time_left() - safe_buffer)
            
            # Base allocation: evenly divide remaining time by remaining turns
            base_allocated = usable_time / turns_left
            
            # Allow it to borrow up to 1.5x its base time for complex mid-game turns, 
            # but never let it spend more than 25% of its total bank on a single move.
            allocated = min(usable_time * 0.25, max(1.0, base_allocated * 1.5))
            
            end_time = start_time + allocated

            # 1-ply root ordering with carpet urgency bonus
            root_scored = []
            my_loc = board.player_worker.get_location()
            for mv in moves:
                child = board.forecast_move(mv)
                if child is None: continue
                child.reverse_perspective()
                extra = 0.0
                if mv.move_type == MoveType.CARPET:
                    chain_len = _adjacent_primed_chain(my_loc, board)
                    extra = 3.0 * CARPET_SCORE.get(chain_len, 0)
                root_scored.append((-evaluate(child, rb) + extra, mv))

            root_scored.sort(key=lambda x: x[0], reverse=True)
            moves_ordered    = [m for _, m in root_scored] or moves
            global_best_move = moves_ordered[0]
            
            completed_best_move = global_best_move
            try:
                for depth in range(1, 15):
                    if time.time() > end_time:
                        break

                    alpha = -float("inf")
                    beta  =  float("inf")
                    
                    # Store all (score, move) tuples for this depth
                    root_scores = []

                    for i, mv in enumerate(moves_ordered):
                        if time.time() > end_time:
                            raise timeout()
                            
                        child = board.forecast_move(mv)
                        if child is None: continue
                        child.reverse_perspective()

                        # PVS (Principal Variation Search)
                        if i == 0:
                            val = -self._negamax(child, depth-1, rb,
                                                 -beta, -alpha, end_time, self._tt)
                        else:
                            val = -self._negamax(child, depth-1, rb,
                                                 -alpha-1, -alpha, end_time, self._tt)
                            if alpha < val < beta:
                                val = -self._negamax(child, depth-1, rb,
                                                     -beta, -alpha, end_time, self._tt)

                        root_scores.append((val, mv))
                        alpha = max(alpha, val)

                    # 1. Sort the moves mathematically so the best move is searched first next depth
                    root_scores.sort(key=lambda x: x[0], reverse=True)
                    moves_ordered = [m for v, m in root_scores]
                    
                    if root_scores:
                        completed_best_move = root_scores[0][1]

                    # --- 2. EARLY EXIT: THE CONFIDENCE CUTOFF ---
                    # If we have searched to at least Depth 3, let's look at the scores.
                    if depth >= 3 and len(root_scores) >= 2:
                        best_score = root_scores[0][0]
                        second_score = root_scores[1][0]
                        
                        # If our #1 move beats our #2 move by more than 18 points (a massive margin),
                        # it's a "no-brainer". Stop wasting time and just play it!
                        if best_score - second_score > 18.0:
                            break

            except timeout:
                # We timed out, but completed_best_move safely holds the last fully completed depth!
                pass

            return self._return_and_track(completed_best_move)

        # --- Greedy fallback ---
        if moves:
            return self._return_and_track(self._greedy(moves, board, rb))

        rb_cell = (rb.best_cell()[0] if rb is not None
                   else (random.randint(0,7), random.randint(0,7)))
        return move.Move.search(rb_cell)

    # ------------------------------------------------------------------
    # Greedy fallback
    # ------------------------------------------------------------------

    def _greedy(self, moves, board_state, rb):
        carpet_moves = []
        prime_moves  = []
        plain_moves  = []
        my_loc = board_state.player_worker.get_location()

        for mv in moves:
            if mv.move_type == MoveType.CARPET:
                roll_score = CARPET_SCORE.get(mv.roll_length, -1)
                if roll_score > 0:
                    carpet_moves.append((roll_score, mv))
            elif mv.move_type == MoveType.PRIME:
                prime_moves.append(mv)
            elif mv.move_type == MoveType.PLAIN:
                plain_moves.append(mv)

        if carpet_moves:
            return max(carpet_moves, key=lambda x: x[0])[1]

        if prime_moves:
            def prime_key(mv):
                dest      = _move_destination(mv, my_loc)
                dx, dy    = _DIRECTION_DELTAS.get(mv.direction, (0, 0))
                future    = _future_chain_potential(dest, board_state)
                chain_b   = _chain_continuation_bonus(dest, dx, dy, board_state)
                chain_now = _adjacent_primed_chain(my_loc, board_state)
                score     = future + chain_b
                if chain_now >= 3: score -= 10
                return score
            return max(prime_moves, key=prime_key)

        if plain_moves:
            if rb is not None:
                rat_cell, rat_p, _ = rb.best_cell()
                if rat_p > 0.15:
                    plain_moves.sort(
                        key=lambda m: manhattan(_move_destination(m, my_loc), rat_cell))
                    return plain_moves[0]
            plain_moves.sort(
                key=lambda m: _future_chain_potential(
                    _move_destination(m, my_loc), board_state),
                reverse=True)
            return plain_moves[0]

        return random.choice(moves)
