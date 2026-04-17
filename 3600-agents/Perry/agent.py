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
NUM_CELLS = BOARD_SIZE * BOARD_SIZE

# P(noise | floor_type)  —  squeak / scratch / squeal
NOISE_EMIT = {
    Cell.BLOCKED: {enums.Noise.SQUEAK: 0.5,  enums.Noise.SCRATCH: 0.3,  enums.Noise.SQUEAL: 0.2},
    Cell.SPACE:   {enums.Noise.SQUEAK: 0.7,  enums.Noise.SCRATCH: 0.15, enums.Noise.SQUEAL: 0.15},
    Cell.PRIMED:  {enums.Noise.SQUEAK: 0.1,  enums.Noise.SCRATCH: 0.8,  enums.Noise.SQUEAL: 0.1},
    Cell.CARPET:  {enums.Noise.SQUEAK: 0.1,  enums.Noise.SCRATCH: 0.1,  enums.Noise.SQUEAL: 0.8},
}

# P(reported_dist = true_dist + offset)
DIST_OFFSETS = {-1: 0.12, 0: 0.70, 1: 0.12, 2: 0.06}

# Points for carpeting a run of length n
CARPET_SCORE = {1: -1, 2: 2, 3: 4, 4: 6, 5: 10, 6: 15, 7: 21}

# Transposition table flag types
TT_EXACT = 0
TT_LOWERBOUND = 1
TT_UPPERBOUND = 2


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
    if board_state._carpet_mask & bit:
        return Cell.CARPET
    if board_state._primed_mask & bit:
        return Cell.PRIMED
    if board_state._blocked_mask & bit:
        return Cell.BLOCKED
    return Cell.SPACE


def compute_rat_spawn_dist(T: np.ndarray, steps: int = 1000) -> np.ndarray:
    """
    Simulate the rat's 1000-step headstart from cell (0,0).
    The rat is NOT uniformly distributed at game start.
    """
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
    Tracks a probability distribution over all 64 cells for the rat's location.

    Update order per turn:
      1. update_search()   — apply last turn's search result (before rat moved)
      2. predict()         — rat moves one step according to T
      3. update_noise()    — reweight using the noise observation
      4. update_distance() — reweight using the noisy distance sensor
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
        """Return the k cells with the highest belief probability as (prob, coord) pairs."""
        idxs = np.argsort(self.belief)[::-1][:k]
        return [(float(self.belief[i]), cell_to_xy(int(i))) for i in idxs]

    def inverse_distance_heat(self, pos: Tuple[int, int]) -> float:
        """Measure how close pos is to high-probability rat cells."""
        top = self.top_cells(8)
        return sum(p / (1.0 + manhattan(pos, cell)) for p, cell in top)

    def expected_distance(self, pos: Tuple[int, int]) -> float:
        """Expected Manhattan distance to the rat mass."""
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
        p = float(self.belief[idx])
        ev = 4.0 * p - 2.0 * (1.0 - p)  # = 6p - 2
        return cell_to_xy(idx), p, ev

    def top_n_ev(self, n: int = 3) -> List[Tuple[Tuple[int, int], float, float]]:
        """Return top n cells by probability with their EV."""
        idxs = np.argsort(self.belief)[::-1][:n]
        results = []
        for i in idxs:
            p = float(self.belief[i])
            ev = 6.0 * p - 2.0
            results.append((cell_to_xy(int(i)), p, ev))
        return results

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
    steps = mv.roll_length if mv.move_type == MoveType.CARPET else 1
    return (current_pos[0] + dx * steps, current_pos[1] + dy * steps)


# ===========================================================================
# Heuristic helpers
# ===========================================================================

def _max_carpet_potential(loc, board_state) -> int:
    """
    Best immediate carpet roll score from loc.
    Only counts PRIMED cells — value we can cash in right now.
    """
    max_score = 0
    enemy_loc = board_state.opponent_worker.get_location()
    player_loc = board_state.player_worker.get_location()

    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        length = 0
        nx, ny = loc[0] + dx, loc[1] + dy
        while board_state.is_valid_cell((nx, ny)):
            if (nx, ny) == enemy_loc or (nx, ny) == player_loc:
                break
            bit = 1 << (ny * BOARD_SIZE + nx)
            if board_state._primed_mask & bit:
                length += 1
                nx += dx
                ny += dy
            else:
                break
        if length > 0:
            max_score = max(max_score, CARPET_SCORE.get(length, 0))
    return max_score


def _max_carpet_length(loc, board_state) -> int:
    """Return the length of the best carpet roll from loc (not the score, the length)."""
    best = 0
    enemy_loc = board_state.opponent_worker.get_location()
    player_loc = board_state.player_worker.get_location()

    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        length = 0
        nx, ny = loc[0] + dx, loc[1] + dy
        while board_state.is_valid_cell((nx, ny)):
            if (nx, ny) == enemy_loc or (nx, ny) == player_loc:
                break
            bit = 1 << (ny * BOARD_SIZE + nx)
            if board_state._primed_mask & bit:
                length += 1
                nx += dx
                ny += dy
            else:
                break
        best = max(best, length)
    return best


def _future_chain_potential(loc: Tuple[int, int], board_state) -> int:
    """
    Maximum carpet chain that COULD be built from loc.
    Counts SPACE + PRIMED cells in each direction until blocked.
    """
    best = 0
    enemy_loc = board_state.opponent_worker.get_location()
    player_loc = board_state.player_worker.get_location()

    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        length = 0
        nx, ny = loc[0] + dx, loc[1] + dy
        while board_state.is_valid_cell((nx, ny)):
            if (nx, ny) == enemy_loc or (nx, ny) == player_loc:
                break
            bit = 1 << (ny * BOARD_SIZE + nx)
            if board_state._blocked_mask & bit:
                break
            length += 1
            nx += dx
            ny += dy
        if length > 0:
            best = max(best, CARPET_SCORE.get(min(length, 7), 0))
    return best


def _chain_continuation_bonus(dest: Tuple[int, int], dx: int, dy: int,
                               board_state) -> float:
    """
    Count PRIMED cells immediately ahead of dest in direction (dx, dy).
    Priming dest when primed cells are already ahead = carpet roll next turn.
    """
    count = 0
    nx, ny = dest[0] + dx, dest[1] + dy
    while board_state.is_valid_cell((nx, ny)):
        bit = 1 << (ny * BOARD_SIZE + nx)
        if board_state._primed_mask & bit:
            count += 1
            nx += dx
            ny += dy
        else:
            break
    return count * 3.0


def _adjacent_primed_chain(loc, board_state):
    """
    Returns the length of the longest contiguous PRIMED chain
    starting adjacent to loc in any direction.
    """
    best = 0
    for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
        length = 0
        nx, ny = loc[0] + dx, loc[1] + dy
        while board_state.is_valid_cell((nx, ny)):
            bit = 1 << xy_to_cell(nx, ny)
            if board_state._primed_mask & bit:
                length += 1
                nx += dx
                ny += dy
            else:
                break
        best = max(best, length)
    return best


def _cell_potential(x: int, y: int, board_state) -> int:
    """
    How valuable is this cell for future carpet chain building?
    Returns the max straight-line run of non-blocked cells in any direction
    passing through (x,y). This is the Carrie-style "cell potential" metric.
    """
    bit = 1 << (y * BOARD_SIZE + x)
    if board_state._blocked_mask & bit:
        return 0

    best = 0
    for dx, dy in [(0, 1), (1, 0)]:  # Only need 2 axes (each covers both directions)
        length = 1  # count self
        # Forward
        nx, ny = x + dx, y + dy
        while 0 <= nx < 8 and 0 <= ny < 8:
            if board_state._blocked_mask & (1 << (ny * 8 + nx)):
                break
            length += 1
            nx += dx
            ny += dy
        # Backward
        nx, ny = x - dx, y - dy
        while 0 <= nx < 8 and 0 <= ny < 8:
            if board_state._blocked_mask & (1 << (ny * 8 + nx)):
                break
            length += 1
            nx += dx
            ny += dy
        best = max(best, length)
    return min(best, 7)


def _quick_mobility(loc: Tuple[int, int], board_state) -> int:
    """
    Fast O(4) mobility check — counts non-blocked cardinal neighbors.
    Replaces the expensive flood-fill reachable_space().
    """
    count = 0
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nx, ny = loc[0] + dx, loc[1] + dy
        if board_state.is_valid_cell((nx, ny)):
            bit = 1 << (ny * BOARD_SIZE + nx)
            if not (board_state._blocked_mask & bit):
                count += 1
    return count


# ===========================================================================
# Static evaluation — the core heuristic
# ===========================================================================

def evaluate(board_state, rat_belief: RatBelief, depth_simulated: int = 0) -> float:
    my = board_state.player_worker
    opp = board_state.opponent_worker
    my_loc = my.get_location()
    opp_loc = opp.get_location()

    # 1. Base Score Difference (most important — actual points earned)
    score = 30.0 * (my.get_points() - opp.get_points())

    # 2. Carpet potential — what we can cash in RIGHT NOW
    my_carpet = _max_carpet_potential(my_loc, board_state)
    opp_carpet = _max_carpet_potential(opp_loc, board_state)
    my_carpet_len = _max_carpet_length(my_loc, board_state)
    opp_carpet_len = _max_carpet_length(opp_loc, board_state)

    worker_dist = manhattan(my_loc, opp_loc)

    # Dynamic carpet weighting:
    # Low weight when threatened = prefer cashing out (rolling converts
    # chain potential into points, which are weighted at 30.0)
    # High weight when safe = patient chain building
    if worker_dist <= 2:
        carpet_weight = 4.0     # Opponent is close: roll NOW, don't hold
    elif worker_dist <= 4:
        carpet_weight = 14.0    # Medium range: moderate preference to hold
    else:
        carpet_weight = 20.0    # Safe: build long chains patiently

    score += carpet_weight * my_carpet
    score -= carpet_weight * opp_carpet

    # Bonus for having a long chain ready to roll (superlinear scoring)
    chain_now = _adjacent_primed_chain(my_loc, board_state)
    if chain_now >= 3:
        score += 6.0 * CARPET_SCORE.get(chain_now, 0)

    # 3. Opponent steal detection — if opponent can carpet our primed cells
    if opp_carpet_len >= 2 and worker_dist <= 4:
        score -= 8.0 * CARPET_SCORE.get(opp_carpet_len, 0)

    # 4. Future Chain Potential — how good is our position for building?
    turns_left = max(1, my.turns_left or 1)
    horizon = max(0.1, turns_left / 40.0)
    score += 3.5 * _future_chain_potential(my_loc, board_state) * horizon
    score -= 3.5 * _future_chain_potential(opp_loc, board_state) * horizon

    # 5. Cell Potential (Carrie-style) — value of being near high-potential cells
    my_pot = _cell_potential(my_loc[0], my_loc[1], board_state)
    opp_pot = _cell_potential(opp_loc[0], opp_loc[1], board_state)
    score += 2.5 * my_pot * horizon
    score -= 2.5 * opp_pot * horizon

    # 6. Quick Mobility — avoid getting boxed in (cheap O(4) check)
    my_mob = _quick_mobility(my_loc, board_state)
    opp_mob = _quick_mobility(opp_loc, board_state)
    score += 1.5 * my_mob
    score -= 1.5 * opp_mob

    # Severe penalty for being trapped (0 or 1 exit)
    if my_mob <= 1:
        score -= 15.0
    if opp_mob <= 1:
        score += 15.0

    # 7. Rat Hunting — position value relative to rat belief
    if rat_belief is not None:
        decay = 0.85 ** depth_simulated

        # Only factor in rat position when we don't have a huge chain to cash
        if my_carpet_len <= 2:
            my_heat = rat_belief.inverse_distance_heat(my_loc)
            my_dist = rat_belief.expected_distance(my_loc)
            score += (7.0 * my_heat - 0.4 * my_dist) * decay

        if opp_carpet_len <= 2:
            opp_heat = rat_belief.inverse_distance_heat(opp_loc)
            opp_dist = rat_belief.expected_distance(opp_loc)
            score -= (7.0 * opp_heat - 0.4 * opp_dist) * decay

    # 8. Center control bonus in early game — center cells have more potential
    if turns_left >= 25:
        my_center = max(0, 3.5 - abs(my_loc[0] - 3.5)) + max(0, 3.5 - abs(my_loc[1] - 3.5))
        opp_center = max(0, 3.5 - abs(opp_loc[0] - 3.5)) + max(0, 3.5 - abs(opp_loc[1] - 3.5))
        score += 1.0 * my_center
        score -= 1.0 * opp_center

    return score


# ===========================================================================
# Move ordering — cheap raycasts only, NO forecast_move calls
# ===========================================================================

def quick_score(mv, board_state, rat_belief: RatBelief) -> float:
    """
    Fast static move ordering to maximize alpha-beta cutoffs.
    Must NOT call forecast_move().
    """
    if mv.move_type == MoveType.CARPET:
        return 100.0 + CARPET_SCORE.get(mv.roll_length, 0)

    if mv.move_type == MoveType.PRIME:
        my_loc = board_state.player_worker.get_location()
        dest = _move_destination(mv, my_loc)
        dx, dy = _DIRECTION_DELTAS.get(mv.direction, (0, 0))
        chain_bonus = _chain_continuation_bonus(dest, dx, dy, board_state)
        future = _future_chain_potential(dest, board_state)
        return 10.0 + future + chain_bonus

    if mv.move_type == MoveType.PLAIN:
        # Prefer plain moves toward high-potential areas
        my_loc = board_state.player_worker.get_location()
        dest = _move_destination(mv, my_loc)
        pot = _cell_potential(dest[0], dest[1], board_state)
        return 1.0 + 0.5 * pot

    return -1000.0


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
        self._hits = 0
        self._misses = 0
        self._last_opp_search_turn = -1
        self._last_my_search_turn = -1
        self._miss_cooldown = 0

        self._primes_done = 0
        self._carpets_made = 0

        self._tt = {}
        self._last_turns_remaining = None
        self._just_caught_rat = False

    def commentate(self):
        rats_searched = self._hits + self._misses
        points_lost = self._misses * 2

        stats = (f"Primes: {self._primes_done} | Carpets: {self._carpets_made} | "
                 f"Rats Found: {self._hits} | Searched: {rats_searched} | Pts Lost: {points_lost}")

        if self.rat_belief is not None:
            cell, p, ev = self.rat_belief.best_cell()
            return f"Turns: {self._turns} | {stats} | Peak: {cell} p={p:.3f} EV={ev:.2f}"
        return f"Turns: {self._turns} | {stats} | (no HMM)"

    def _return_and_track(self, mv):
        """Helper to update our stats before returning a move."""
        if mv is None:
            rat_cell, rat_p, rat_ev = self.rat_belief.best_cell() if self.rat_belief is not None else ((random.randint(0, 7), random.randint(0, 7)), 0.0, -2.0)
            return move.Move.search(rat_cell)
        if mv.move_type == MoveType.CARPET:
            self._carpets_made += 1
        elif mv.move_type == MoveType.PRIME:
            self._primes_done += 1
        return mv

    # ------------------------------------------------------------------
    # Negamax with alpha-beta pruning + correct TT bound types
    # ------------------------------------------------------------------

    def _negamax(self, board_state, depth: int, rat_belief: RatBelief,
                alpha: float, beta: float, end_time: float,
                tt: dict, depth_simulated: int = 0) -> float:

        if time.time() > end_time:
            raise SearchTimeout()

        state_key = (
            board_state._primed_mask, board_state._carpet_mask,
            board_state.player_worker.get_location(), board_state.opponent_worker.get_location(),
            board_state.player_worker.get_points() - board_state.opponent_worker.get_points(),
            board_state.player_worker.turns_left
        )

        # Transposition table lookup with proper bound types
        cached = tt.get(state_key)
        if cached is not None and cached[0] >= depth:
            cached_depth, cached_val, cached_flag = cached
            if cached_flag == TT_EXACT:
                return cached_val
            elif cached_flag == TT_LOWERBOUND:
                alpha = max(alpha, cached_val)
            elif cached_flag == TT_UPPERBOUND:
                beta = min(beta, cached_val)
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

        # Adaptive move pruning — wider at shallow depths
        if depth <= 1:
            moves = moves[:8]
        elif depth <= 2:
            moves = moves[:12]
        else:
            moves = moves[:16]

        orig_alpha = alpha
        best = -float("inf")

        for i, mv in enumerate(moves):
            if time.time() > end_time:
                raise SearchTimeout()

            child = board_state.forecast_move(mv)
            if child is None:
                continue

            child.reverse_perspective()

            # Principal Variation Search (PVS)
            if i == 0:
                val = -self._negamax(child, depth - 1, rat_belief,
                            -beta, -alpha, end_time, tt,
                            depth_simulated + 1)
            else:
                # Null-window search first
                val = -self._negamax(child, depth - 1, rat_belief,
                            -alpha - 1, -alpha, end_time, tt,
                            depth_simulated + 1)
                # Re-search if it fails high within our window
                if alpha < val < beta:
                    val = -self._negamax(child, depth - 1, rat_belief,
                                -beta, -alpha, end_time, tt,
                                depth_simulated + 1)

            if val > best:
                best = val
            alpha = max(alpha, best)
            if alpha >= beta:
                break

        if best == -float("inf"):
            best = evaluate(board_state, rat_belief, depth_simulated)

        # Store with correct bound type
        if best <= orig_alpha:
            tt_flag = TT_UPPERBOUND
        elif best >= beta:
            tt_flag = TT_LOWERBOUND
        else:
            tt_flag = TT_EXACT
        tt[state_key] = (depth, best, tt_flag)

        return best

    # ------------------------------------------------------------------
    # Main play method
    # ------------------------------------------------------------------

    def play(self, board: board.Board, sensor_data: Tuple, time_left: Callable):
        my_turns = board.player_worker.turns_left

        # Continuous game reset detector
        if self._last_turns_remaining is None or my_turns > self._last_turns_remaining:
            self._turns = 0
            self._tt.clear()
            self._miss_cooldown = 0
            self._just_caught_rat = False
            if self._tm is not None:
                self.rat_belief = RatBelief(self._tm)

        self._last_turns_remaining = my_turns - 1
        self._turns += 1

        if self._miss_cooldown > 0:
            self._miss_cooldown -= 1

        # 0. Lazy-init HMM
        if self.rat_belief is None:
            tm = self._tm
            if tm is None:
                try: tm = board.transition_matrix
                except AttributeError: pass
            if tm is not None: self.rat_belief = RatBelief(tm)

        rb = self.rat_belief
        noise, reported_dist = sensor_data

        # 1. HMM Update — process searches FIRST, then predict, then observe
        if rb is not None:
            # Process opponent's search result
            opp_loc, opp_found = board.opponent_search
            if opp_loc is not None and self._turns != self._last_opp_search_turn:
                rb.update_search(opp_loc, opp_found)
                self._last_opp_search_turn = self._turns

            # Process our own search result
            my_search_loc, my_found = board.player_search
            if my_search_loc is not None and self._turns != self._last_my_search_turn:
                if my_found:
                    self._hits += 1
                    self._just_caught_rat = True
                    self._miss_cooldown = 0  # Reset cooldown on success!
                else:
                    self._misses += 1
                    self._miss_cooldown = 2  # Shorter cooldown (was 3)
                rb.update_search(my_search_loc, my_found)
                self._last_my_search_turn = self._turns

            # The rat moves twice between your turns (opponent's turn + your turn)
            if self._turns > 1:
                rb.predict()
                rb.predict()

            # Observe this turn's sensor data
            if noise is not None:
                rb.update_noise(noise, board)
            if reported_dist is not None:
                try:
                    rb.update_distance(int(reported_dist),
                                       board.player_worker.get_location())
                except Exception:
                    pass

        moves = board.get_valid_moves(exclude_search=True)
        turns_left = max(1, board.player_worker.turns_left)

        # --- LAST TURN LOGIC ---
        if turns_left == 1:
            carpet_moves = [m for m in moves if m.move_type == MoveType.CARPET]
            if carpet_moves:
                best = max(carpet_moves, key=lambda m: CARPET_SCORE.get(m.roll_length, -1))
                if best is not None and best.roll_length >= 2:
                    return self._return_and_track(best)

            if rb is not None:
                rat_cell, rat_p, rat_ev = rb.best_cell()
                if rat_ev >= 0.5:  # More aggressive on last turn
                    return self._return_and_track(move.Move.search(rat_cell))

            for mv in moves:
                if mv.move_type == MoveType.PRIME:
                    return self._return_and_track(mv)

            plain_moves = [m for m in moves if m.move_type == MoveType.PLAIN]
            if plain_moves:
                return self._return_and_track(plain_moves[0])
            return self._return_and_track(random.choice(moves))

        # 2. Opportunistic Search (smart thresholds)
        if rb is not None and self._miss_cooldown == 0:
            rat_cell, rat_p, rat_ev = rb.best_cell()
            my_score = board.player_worker.get_points()
            opp_score = board.opponent_worker.get_points()

            # Dynamic EV threshold based on game state
            if turns_left <= 5:
                ev_threshold = 0.5   # Desperate endgame: take coin-flip rat searches
            elif turns_left <= 10:
                ev_threshold = 1.0   # Late game: lower bar
            elif my_score < opp_score - 5:
                ev_threshold = 1.2   # Trailing badly: take risks
            elif my_score < opp_score:
                ev_threshold = 1.6   # Trailing slightly
            else:
                ev_threshold = 2.0   # Leading or tied: be conservative

            if rat_ev >= ev_threshold:
                my_loc_now = board.player_worker.get_location()
                my_carpet_now = _max_carpet_length(my_loc_now, board)

                # Don't waste a turn searching if we have a prime chain of 4+ ready
                if my_carpet_now < 4:
                    return move.Move.search(rat_cell)

        # 3. Iterative-deepening Alpha-Beta Search with PVS
        if moves and rb is not None:
            self._tt.clear()

            start_time = time.time()
            turns_left = max(1, board.player_worker.turns_left or 1)

            # Time budgeting
            safe_buffer = 1.0
            usable_time = max(0.1, time_left() - safe_buffer)
            if 15 <= turns_left <= 30:
                allocated = min(3.5, usable_time / 3.0)
            elif turns_left <= 5:
                allocated = min(1.5, usable_time / max(1, turns_left))
            else:
                allocated = min(2.5, usable_time / max(1, turns_left))

            end_time = start_time + allocated

            # 1-Ply Root Forecasting to prep the move order
            root_scored = []
            for mv in moves:
                child = board.forecast_move(mv)
                if child is None: continue
                child.reverse_perspective()
                if mv.move_type == MoveType.CARPET:
                    chain_len = _adjacent_primed_chain(
                        board.player_worker.get_location(), board
                    )
                    extra = 3.0 * CARPET_SCORE.get(chain_len, 0)
                else:
                    extra = 0
                root_scored.append((-evaluate(child, rb) + extra, mv))

            root_scored.sort(key=lambda x: x[0], reverse=True)
            moves_ordered = [m for v, m in root_scored] or moves

            global_best_move = moves_ordered[0]

            try:
                for depth in range(1, 15):
                    if time.time() > end_time:
                        break

                    alpha = -float("inf")
                    beta = float("inf")
                    best_val_this_depth = -float("inf")

                    # Principal Variation Sorting
                    if global_best_move in moves_ordered:
                        moves_ordered.remove(global_best_move)
                        moves_ordered.insert(0, global_best_move)

                    for i, mv in enumerate(moves_ordered):
                        if time.time() > end_time:
                            raise SearchTimeout()

                        child = board.forecast_move(mv)
                        if child is None: continue

                        child.reverse_perspective()

                        # PVS at root level too
                        if i == 0:
                            val = -self._negamax(child, depth - 1, rb,
                                        -beta, -alpha, end_time, self._tt)
                        else:
                            val = -self._negamax(child, depth - 1, rb,
                                        -alpha - 1, -alpha, end_time, self._tt)
                            if alpha < val < beta:
                                val = -self._negamax(child, depth - 1, rb,
                                            -beta, -alpha, end_time, self._tt)

                        if val > best_val_this_depth:
                            best_val_this_depth = val
                            global_best_move = mv

                        alpha = max(alpha, best_val_this_depth)

            except SearchTimeout:
                pass

            return self._return_and_track(global_best_move)

        # 4. Greedy fallback
        if moves:
            return self._return_and_track(self._greedy(moves, board, rb))

        rat_cell, rat_p, rat_ev = rb.best_cell() if rb is not None else ((random.randint(0, 7), random.randint(0, 7)), 0.0, -2.0)
        return move.Move.search(rat_cell)

    # ------------------------------------------------------------------
    # Greedy fallback move selection
    # ------------------------------------------------------------------

    def _greedy(self, moves, board_state, rb):
        """
        Priority: carpet roll > prime (chain-aware) > plain (rat or corridor)
        """
        carpet_moves = []
        prime_moves = []
        plain_moves = []

        my_loc = board_state.player_worker.get_location()

        for mv in moves:
            if mv.move_type == MoveType.CARPET:
                chain_len = _adjacent_primed_chain(my_loc, board_state)
                return_score = 3.0 * CARPET_SCORE.get(chain_len, 0)
                carpet_moves.append((return_score, mv))
            elif mv.move_type == MoveType.PRIME:
                prime_moves.append(mv)
            elif mv.move_type == MoveType.PLAIN:
                plain_moves.append(mv)

        # 1. Best scoring carpet roll
        if carpet_moves:
            best = max(carpet_moves, key=lambda x: x[0])[1]
            return best

        # 2. Prime: future chain potential + chain continuation bonus
        if prime_moves:
            def prime_key(mv):
                dest = _move_destination(mv, my_loc)
                dx, dy = _DIRECTION_DELTAS.get(mv.direction, (0, 0))

                chain_now = _adjacent_primed_chain(my_loc, board_state)
                future = _future_chain_potential(dest, board_state)
                chain_bonus = _chain_continuation_bonus(dest, dx, dy, board_state)

                score = future + chain_bonus

                if chain_now >= 3:
                    score -= 10  # discourage over-priming when cash-out is ready

                return score
            return max(prime_moves, key=prime_key)

        # 3. Plain: move toward rat if concentrated, else best open corridor
        if plain_moves:
            if rb is not None:
                rat_cell, rat_p, _ = rb.best_cell()
                if rat_p > 0.15:
                    plain_moves.sort(
                        key=lambda m: manhattan(
                            _move_destination(m, my_loc), rat_cell
                        )
                    )
                    return plain_moves[0]
            plain_moves.sort(
                key=lambda mv: _future_chain_potential(
                    _move_destination(mv, my_loc), board_state
                ),
                reverse=True,
            )
            return plain_moves[0]

        return random.choice(moves)
