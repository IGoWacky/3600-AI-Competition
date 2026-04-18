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

# HMM calibration — raised from 0.001 to reduce overconfidence  [FIX #4]
HMM_SMOOTHING     = 0.025
HMM_MAX_CELL_PROB = 0.45          # cap any single cell probability


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
# RatBelief — Hidden Markov Model  (with calibration fixes)
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
        self.belief += HMM_SMOOTHING / NUM_CELLS          # [FIX #4] was 0.001
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
            self.belief[xy_to_cell(*searched_pos)] = 0.0
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
        # [FIX #4] Cap max cell probability to prevent overconfident searches
        mx = self.belief.max()
        if mx > HMM_MAX_CELL_PROB:
            self.belief = np.clip(self.belief, 0.0, HMM_MAX_CELL_PROB)
            t2 = self.belief.sum()
            if t2 > 1e-12:
                self.belief /= t2


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

def _max_carpet_info(loc, board_state) -> Tuple[int, int]:
    """
    Returns (best_score, best_length) in one pass.          [FIX #5 — merged]
    """
    best_score = 0
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
        if length > 0:
            best_score = max(best_score, CARPET_SCORE.get(length, 0))
            best_len   = max(best_len, length)
    return best_score, best_len


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
    from loc are eventually primed.  Carpet squares are treated as blockers.
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
    """
    Max straight-line run of non-blocked cells through (x,y).
    Carrie-style positional value metric.
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


def _qmob(loc: Tuple[int, int], board_state) -> int:
    """
    O(4) quick mobility — count walkable adjacent cells.              [FIX #1]
    Replaces flood-fill inside evaluate() for massive speed gain.
    """
    count = 0
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        nx, ny = loc[0]+dx, loc[1]+dy
        if not board_state.is_valid_cell((nx, ny)): continue
        if board_state._blocked_mask & (1 << (ny * BOARD_SIZE + nx)): continue
        count += 1
    return count


def reachable_space(loc: Tuple[int, int], board_state) -> int:
    """
    Flood-fill reachable space — kept for root-only strategic use.
    NOT called inside evaluate() anymore.                             [FIX #1]
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
            if board_state._blocked_mask & bit: continue
            stack.append((nx, ny))
    return len(visited)


# ===========================================================================
# Static evaluation  (flood-fill replaced with _qmob)
# ===========================================================================

def evaluate(board_state, rat_belief: RatBelief, depth_simulated: int = 0) -> float:
    my      = board_state.player_worker
    opp     = board_state.opponent_worker
    my_loc  = my.get_location()
    opp_loc = opp.get_location()
    turns_left = max(1, my.turns_left or 1)

    # --- 1. Actual score differential ---
    score = 15 * (my.get_points() - opp.get_points())

    # --- 2. Carpet potential  [FIX #5 — single call returns both] ---
    my_carpet, my_carpet_len   = _max_carpet_info(my_loc,  board_state)
    opp_carpet, opp_carpet_len = _max_carpet_info(opp_loc, board_state)
    worker_dist = manhattan(my_loc, opp_loc)

    if worker_dist <= 2:   carpet_weight = 4.0
    elif worker_dist <= 4: carpet_weight = 14.0
    else:                  carpet_weight = 20.0

    score += carpet_weight * my_carpet
    score -= carpet_weight * opp_carpet

    # --- 3. Dynamic Chains & The "Greed Limit" ---
    my_chain  = _adjacent_primed_chain(my_loc, board_state)
    opp_chain = _adjacent_primed_chain(opp_loc, board_state)

    opp_dist_to_my_chain = manhattan(opp_loc, _best_carpet_end(my_loc, board_state))

    if opp_dist_to_my_chain <= 2:
        safe_limit = 2
    elif opp_dist_to_my_chain == 3:
        safe_limit = 4
    else:
        safe_limit = 7

    if my_chain >= 3:
        if my_chain > safe_limit:
            score -= 20.0 * CARPET_SCORE.get(my_chain, 0)
        else:
            score += 18.0 * CARPET_SCORE.get(my_chain, 0)

    if opp_chain >= 3:
        score -= 18.0 * CARPET_SCORE.get(opp_chain, 0)

    # HUNT/STEAL
    if opp_carpet_len >= 2:
        my_dist_to_opp_chain = manhattan(my_loc, _best_carpet_end(opp_loc, board_state))
        if my_dist_to_opp_chain <= 2:
            score += 15.0 * CARPET_SCORE.get(opp_carpet_len, 0)

    # --- 4. Future chain potential ---
    horizon = max(0.1, turns_left / 40.0)
    score += 3.5 * _future_chain_potential(my_loc,  board_state) * horizon
    score -= 3.5 * _future_chain_potential(opp_loc, board_state) * horizon

    # --- 5. Cell potential ---
    my_pot  = _cell_potential(my_loc[0],  my_loc[1],  board_state)
    opp_pot = _cell_potential(opp_loc[0], opp_loc[1], board_state)
    score += 2.5 * my_pot  * horizon
    score -= 2.5 * opp_pot * horizon

    # --- 6. Mobility — O(4) quick check replaces flood-fill  [FIX #1] ---
    my_mob  = _qmob(my_loc,  board_state)
    opp_mob = _qmob(opp_loc, board_state)
    score += 1.5 * (my_mob - opp_mob)
    if my_mob  == 0: score -= 20.0    # completely trapped
    if opp_mob == 0: score += 20.0
    if my_mob  == 1: score -= 10.0    # nearly trapped
    if opp_mob == 1: score += 10.0

    # --- 7. Stranded primed cells penalty ---
    total_primed = bin(board_state._primed_mask).count('1')
    if turns_left < total_primed:
        score -= (total_primed - turns_left) * 2.0

    # --- 8. Rat hunting ---
    if rat_belief is not None:
        decay = 0.85 ** depth_simulated
        if my_carpet_len <= 2:
            my_heat = rat_belief.inverse_distance_heat(my_loc)
            my_dist = rat_belief.expected_distance(my_loc)
            score += (5.0 * my_heat - 0.3 * my_dist) * decay   # reduced from 7.0/0.4
        if opp_carpet_len <= 2:
            opp_heat = rat_belief.inverse_distance_heat(opp_loc)
            opp_dist = rat_belief.expected_distance(opp_loc)
            score -= (5.0 * opp_heat - 0.3 * opp_dist) * decay

    # --- 9. Early-game center control ---
    if turns_left >= 25:
        my_center  = (max(0, 3.5 - abs(my_loc[0]  - 3.5)) +
                      max(0, 3.5 - abs(my_loc[1]  - 3.5)))
        opp_center = (max(0, 3.5 - abs(opp_loc[0] - 3.5)) +
                      max(0, 3.5 - abs(opp_loc[1] - 3.5)))
        score += 1.0 * (my_center - opp_center)

    return score


# ===========================================================================
# Move ordering (now includes killer + history bonuses)         [NEW B, C]
# ===========================================================================

def quick_score(mv, board_state, rat_belief: RatBelief,
                killers=None, ply: int = 0, history=None) -> float:
    """Fast static ordering with killer move and history heuristic bonuses."""
    s = 0.0

    if mv.move_type == MoveType.CARPET:
        roll_score = CARPET_SCORE.get(mv.roll_length, 0)
        if roll_score < 0: return -50.0
        return 200.0 + roll_score        # carpets always first

    if mv.move_type == MoveType.PRIME:
        my_loc = board_state.player_worker.get_location()
        dest   = _move_destination(mv, my_loc)
        dx, dy = _DIRECTION_DELTAS.get(mv.direction, (0, 0))
        s = (10.0
             + _future_chain_potential(dest, board_state)
             + _chain_continuation_bonus(dest, dx, dy, board_state))

    elif mv.move_type == MoveType.PLAIN:
        my_loc = board_state.player_worker.get_location()
        dest   = _move_destination(mv, my_loc)
        s = 1.0 + 0.5 * _cell_potential(dest[0], dest[1], board_state)

    else:
        return -1000.0   # SEARCH excluded from tree

    # Killer move bonus — between carpet and normal                    [NEW B]
    if killers is not None and ply < len(killers):
        k = killers[ply]
        mv_key = (mv.direction, mv.move_type)
        if k[0] is not None and (k[0].direction, k[0].move_type) == mv_key:
            s += 80.0
        elif k[1] is not None and (k[1].direction, k[1].move_type) == mv_key:
            s += 70.0

    # History heuristic bonus                                          [NEW C]
    if history is not None:
        h_key = (mv.direction, mv.move_type)
        s += min(60.0, history.get(h_key, 0) * 0.01)

    return s


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
        self._is_player_b: bool = False                            # [FIX P-B]

        # Killer move heuristic — 2 slots per ply                     [NEW B]
        self._killers = [[None, None] for _ in range(20)]
        # History heuristic table                                      [NEW C]
        self._history: dict = {}

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
    # Quiescence search — extend carpet "captures" at leaf nodes  [NEW F]
    # ------------------------------------------------------------------

    def _qsearch(self, board_state, rat_belief: RatBelief,
                 alpha: float, beta: float, end_time: float,
                 depth_simulated: int) -> float:
        if time.time() > end_time:
            raise timeout()

        stand_pat = evaluate(board_state, rat_belief, depth_simulated)
        if stand_pat >= beta:
            return beta
        if stand_pat > alpha:
            alpha = stand_pat

        # Only look at carpet moves of length >= 2 (like captures in chess)
        for mv in board_state.get_valid_moves(exclude_search=True):
            if mv.move_type != MoveType.CARPET or mv.roll_length < 2:
                continue
            child = board_state.forecast_move(mv)
            if child is None:
                continue
            child.reverse_perspective()
            val = -self._qsearch(child, rat_belief, -beta, -alpha,
                                 end_time, depth_simulated + 1)
            if val >= beta:
                return beta
            if val > alpha:
                alpha = val

        return alpha

    # ------------------------------------------------------------------
    # Negamax with alpha-beta + PVS + killers + history + LMR
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

        # Depth 0 → quiescence search instead of bare evaluate  [NEW F]
        if depth == 0 or (board_state.player_worker.turns_left or 0) == 0:
            val = self._qsearch(board_state, rat_belief, alpha, beta,
                                end_time, depth_simulated)
            # Don't store qsearch results in TT (they're variable-depth)
            return val

        moves = list(board_state.get_valid_moves(exclude_search=True))
        if not moves:
            val = evaluate(board_state, rat_belief, depth_simulated)
            tt[state_key] = (depth, val, TT_EXACT)
            return val

        # --- Null-move-style forward pruning ---                      [NEW]
        # If static eval is way above beta, this position is so good
        # that even a "free pass" wouldn't change the outcome. Prune.
        # Only at depth >= 3, not at root (depth_simulated > 0).
        if depth >= 3 and depth_simulated > 0:
            static_eval = evaluate(board_state, rat_belief, depth_simulated)
            margin = 30.0 + 10.0 * depth   # scales with depth
            if static_eval - margin >= beta:
                tt[state_key] = (depth, static_eval, TT_LOWERBOUND)
                return static_eval

        # Move ordering with killers + history                         [NEW B, C]
        moves.sort(key=lambda m: quick_score(m, board_state, rat_belief,
                                             self._killers, depth_simulated,
                                             self._history),
                   reverse=True)

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

            # --- Late Move Reduction (LMR) ---                        [NEW E]
            use_lmr = (i >= 4 and depth >= 3
                       and mv.move_type != MoveType.CARPET
                       and not self._is_killer(mv, depth_simulated))

            if i == 0:
                val = -self._negamax(child, depth-1, rat_belief,
                                     -beta, -alpha, end_time, tt,
                                     depth_simulated+1)
            else:
                # LMR: search at reduced depth first
                if use_lmr:
                    val = -self._negamax(child, depth-2, rat_belief,
                                         -alpha-1, -alpha, end_time, tt,
                                         depth_simulated+1)
                    if val <= alpha:
                        continue   # LMR confirmed — skip full search
                    # Promising — fall through to full PVS re-search

                # PVS null-window
                val = -self._negamax(child, depth-1, rat_belief,
                                     -alpha-1, -alpha, end_time, tt,
                                     depth_simulated+1)
                if alpha < val < beta:
                    val = -self._negamax(child, depth-1, rat_belief,
                                         -beta, -alpha, end_time, tt,
                                         depth_simulated+1)

            if val > best:
                best = val

            if val > alpha:
                alpha = val
                # Update history for best moves                        [NEW C]
                self._update_history(mv, depth)

            if alpha >= beta:
                # Update killer moves on beta cutoff                   [NEW B]
                self._update_killers(mv, depth_simulated)
                break

        if best == -float("inf"):
            best = evaluate(board_state, rat_belief, depth_simulated)

        tt_flag = (TT_UPPERBOUND if best <= orig_alpha else
                   TT_LOWERBOUND if best >= beta else TT_EXACT)
        tt[state_key] = (depth, best, tt_flag)
        return best

    # ------------------------------------------------------------------
    # Killer move helpers                                          [NEW B]
    # ------------------------------------------------------------------

    def _update_killers(self, mv, ply: int):
        if ply >= 20 or mv.move_type == MoveType.CARPET:
            return
        k = self._killers[ply]
        mv_key = (mv.direction, mv.move_type)
        if k[0] is not None and (k[0].direction, k[0].move_type) == mv_key:
            return   # already slot 0
        k[1] = k[0]
        k[0] = mv

    def _is_killer(self, mv, ply: int) -> bool:
        if ply >= 20:
            return False
        k = self._killers[ply]
        mv_key = (mv.direction, mv.move_type)
        return ((k[0] is not None and (k[0].direction, k[0].move_type) == mv_key) or
                (k[1] is not None and (k[1].direction, k[1].move_type) == mv_key))

    # ------------------------------------------------------------------
    # History heuristic helpers                                    [NEW C]
    # ------------------------------------------------------------------

    def _update_history(self, mv, depth: int):
        key = (mv.direction, mv.move_type)
        self._history[key] = self._history.get(key, 0) + depth * depth

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
            # Reset killers and history for new game
            self._killers = [[None, None] for _ in range(20)]
            self._history.clear()
            if self._tm is not None:
                self.rat_belief = RatBelief(self._tm)
            # Detect Player B: if opponent already used a turn, we go second [FIX P-B]
            opp_turns = board.opponent_worker.turns_left
            self._is_player_b = (opp_turns is not None and opp_turns < my_turns)

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

            # Predict rat movement since our last turn              [FIX P-B]
            # Turn 1 Player A: rat hasn't moved → skip
            # Turn 1 Player B: rat moved once (after A's turn) → predict x1
            # Turn 2+: rat moved twice (after our turn + after opp's turn) → predict x2
            if self._turns > 1:
                rb.predict()
                rb.predict()
            elif self._is_player_b:
                rb.predict()    # rat moved once before our first turn

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

        # --- Opportunistic search with dynamic thresholds ---
        if rb is not None and self._miss_cooldown == 0:
            rat_cell, rat_p, rat_ev = rb.best_cell()
            my_score  = board.player_worker.get_points()
            opp_score = board.opponent_worker.get_points()

            if   my_score < opp_score - 5:
                ev_threshold = 1.35
            elif my_score < opp_score:
                ev_threshold = 1.6
            else:
                ev_threshold = 1.9

            if   turns_left <= 5:
                ev_threshold -= 0.45
            elif turns_left <= 10:
                ev_threshold -= 0.2

            if rat_ev >= ev_threshold:
                my_loc_now    = board.player_worker.get_location()
                my_carpet_len = _max_carpet_info(my_loc_now, board)[1]
                if my_carpet_len < 3:
                    return move.Move.search(rat_cell)

        # --- Iterative-deepening negamax with PVS + aspiration  ---
        if moves:
            # TT persistence with size cap instead of clearing      [FIX #2]
            if len(self._tt) > 120_000:
                self._tt.clear()

            # Reset killers each turn (they're ply-relative)
            self._killers = [[None, None] for _ in range(20)]

            start_time   = time.time()
            safe_buffer  = 0.5
            usable_time  = max(0.1, time_left() - safe_buffer)

            base_allocated = usable_time / turns_left
            allocated = min(usable_time * 0.25, max(1.0, base_allocated * 1.5))
            end_time = start_time + allocated

            # Cheap root ordering via quick_score — saves forecast_move calls
            # (depth 1 of iterative deepening does the real 1-ply eval)
            moves_list = list(moves)
            moves_list.sort(
                key=lambda m: quick_score(m, board, rb,
                                          self._killers, 0, self._history),
                reverse=True)
            moves_ordered    = moves_list
            global_best_move = moves_ordered[0]
            completed_best_move = global_best_move

            prev_score = None                                      # [NEW D]

            try:
                for depth in range(1, 15):
                    if time.time() > end_time:
                        break

                    # Aspiration windows from depth 3+                 [NEW D]
                    if prev_score is not None and depth >= 3:
                        asp_delta = 25.0
                        alpha = prev_score - asp_delta
                        beta  = prev_score + asp_delta
                    else:
                        alpha = -float("inf")
                        beta  =  float("inf")

                    root_scores = []
                    failed_low  = False
                    failed_high = False

                    for i, mv in enumerate(moves_ordered):
                        if time.time() > end_time:
                            raise timeout()

                        child = board.forecast_move(mv)
                        if child is None: continue
                        child.reverse_perspective()

                        # PVS at root
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
                        if val > alpha:
                            alpha = val

                    # Check for aspiration window failure               [NEW D]
                    if root_scores:
                        best_this_depth = max(root_scores, key=lambda x: x[0])[0]
                        if prev_score is not None and depth >= 3:
                            if best_this_depth <= prev_score - asp_delta:
                                failed_low = True
                            elif best_this_depth >= prev_score + asp_delta:
                                failed_high = True

                        # If aspiration failed, re-search with full window
                        if failed_low or failed_high:
                            alpha = -float("inf")
                            beta  =  float("inf")
                            root_scores = []
                            for i, mv in enumerate(moves_ordered):
                                if time.time() > end_time:
                                    raise timeout()
                                child = board.forecast_move(mv)
                                if child is None: continue
                                child.reverse_perspective()
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
                                if val > alpha:
                                    alpha = val

                    # Sort moves for next iteration
                    root_scores.sort(key=lambda x: x[0], reverse=True)
                    moves_ordered = [m for v, m in root_scores]

                    if root_scores:
                        completed_best_move = root_scores[0][1]
                        prev_score = root_scores[0][0]             # [NEW D]

                    # Confidence cutoff
                    if depth >= 3 and len(root_scores) >= 2:
                        best_score  = root_scores[0][0]
                        second_score = root_scores[1][0]
                        if best_score - second_score > 18.0:
                            break

            except timeout:
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
            opp_loc = board_state.opponent_worker.get_location()

            def prime_key(mv):
                dest      = _move_destination(mv, my_loc)
                dx, dy    = _DIRECTION_DELTAS.get(mv.direction, (0, 0))
                future    = _future_chain_potential(dest, board_state)
                chain_b   = _chain_continuation_bonus(dest, dx, dy, board_state)
                chain_now = _adjacent_primed_chain(my_loc, board_state)

                opp_dist = manhattan(opp_loc, dest)
                if opp_dist <= 2:
                    safe_limit = 2
                elif opp_dist <= 4:
                    safe_limit = 4
                else:
                    safe_limit = 7

                score = future + chain_b
                if chain_now >= safe_limit:
                    score -= 50.0
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