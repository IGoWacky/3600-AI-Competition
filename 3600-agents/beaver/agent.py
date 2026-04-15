from collections.abc import Callable
from typing import List, Set, Tuple
import random
import time
import numpy as np

from game import board, move, enums
from game.enums import MoveType, Cell


# ===========================================================================
# Constants (from assignment spec)
# ===========================================================================

BOARD_SIZE = 8
NUM_CELLS = BOARD_SIZE * BOARD_SIZE

# P(noise | floor_type)  —  squeak / scratch / squeal
NOISE_EMIT = {
    Cell.BLOCKED: {"squeak": 0.5,  "scratch": 0.3,  "squeal": 0.2},
    Cell.SPACE:   {"squeak": 0.7,  "scratch": 0.15, "squeal": 0.15},
    Cell.PRIMED:  {"squeak": 0.1,  "scratch": 0.8,  "squeal": 0.1},
    Cell.CARPET:  {"squeak": 0.1,  "scratch": 0.1,  "squeal": 0.8},
}

# P(reported_dist = true_dist + offset)
DIST_OFFSETS = {-1: 0.12, 0: 0.70, 1: 0.12, 2: 0.06}

# Points for carpeting a run of length n
CARPET_SCORE = {1: -1, 2: 2, 3: 4, 4: 6, 5: 10, 6: 15, 7: 21}

# Search EV = 6p - 2.  Search when EV > 0  (i.e. p > 1/3).
# OLD value was 1.0 (requiring p > 0.5 — almost never triggered correctly).
SEARCH_EV_THRESHOLD = 0.0


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
    """Return the Cell enum for a location using the board's private masks."""
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

    This gives the correct initial prior over rat positions instead of a
    uniform distribution, which was wrong because the rat is NOT uniformly
    distributed — it starts at (0,0) and walks for 1000 steps.
    """
    dist = np.zeros(NUM_CELLS, dtype=np.float64)
    dist[xy_to_cell(0, 0)] = 1.0  # rat placed at (0,0) initially
    for _ in range(steps):
        dist = dist @ T
    total = dist.sum()
    return dist / total if total > 1e-12 else np.ones(NUM_CELLS) / NUM_CELLS


# ===========================================================================
# RatBelief — Hidden Markov Model for rat location
# ===========================================================================

class RatBelief:
    """
    Tracks a probability distribution over all 64 cells for the rat's location.

    Each turn:
      1. predict()         — propagate belief through transition matrix T
      2. update_noise()    — reweight using the noise observation
      3. update_distance() — reweight using the noisy distance sensor
      4. update_search()   — hard update when a search result is known
    """

    def __init__(self, transition_matrix):
        self.T = np.array(transition_matrix, dtype=np.float64)
        # Correct prior: simulate 1000-step headstart from (0,0)
        self._spawn_dist = compute_rat_spawn_dist(self.T)
        self.belief = self._spawn_dist.copy()

    def predict(self):
        """Propagate belief one step through the transition model."""
        self.belief = self.belief @ self.T

    def update_noise(self, noise: str, board_state):
        """Reweight by P(noise | floor_type of each cell)."""
        lk = np.array([
            NOISE_EMIT[get_floor(cell_to_xy(i), board_state)].get(noise, 1e-9)
            for i in range(NUM_CELLS)
        ], dtype=np.float64)
        self.belief *= lk
        self._normalize()

    def update_distance(self, reported_dist: int, worker_pos: Tuple[int, int]):
        """Reweight by P(reported_dist | true_dist) for each cell."""
        wx, wy = worker_pos
        lk = np.zeros(NUM_CELLS, dtype=np.float64)
        for i in range(NUM_CELLS):
            x, y = cell_to_xy(i)
            true_dist = abs(wx - x) + abs(wy - y)
            if reported_dist == 0:
                lk[i] = sum(DIST_OFFSETS[o] for o in DIST_OFFSETS if true_dist + o <= 0)
            else:
                lk[i] = DIST_OFFSETS.get(reported_dist - true_dist, 0.0)
        self.belief *= lk
        self._normalize()

    def update_search(self, searched_pos: Tuple[int, int], found: bool):
        """Update belief based on a search result (ours or opponent's)."""
        if found:
            # New rat spawns and does 1000 steps — reset to spawn distribution
            self.belief = self._spawn_dist.copy()
        else:
            self.belief[xy_to_cell(*searched_pos)] = 0.0
            self._normalize()

    def best_cell(self) -> Tuple[Tuple[int, int], float, float]:
        """Return (cell_xy, probability, search_EV) for the most likely rat cell."""
        idx = int(np.argmax(self.belief))
        p   = float(self.belief[idx])
        ev  = 6.0 * p - 2.0
        return cell_to_xy(idx), p, ev

    def _normalize(self):
        total = self.belief.sum()
        if total > 1e-12:
            self.belief /= total
        else:
            # Belief collapsed entirely — reset to spawn distribution
            self.belief = self._spawn_dist.copy()


# ===========================================================================
# Heuristic evaluation
# ===========================================================================

def _max_carpet_potential(loc, board_state):
    """
    Fast raycast to find the best possible carpet roll from a location.
    Only counts already-PRIMED cells — this is the immediate carpet value.
    """
    max_score = 0
    enemy_loc = board_state.opponent_worker.get_location()
    player_loc = board_state.player_worker.get_location()

    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
        length = 0
        nx, ny = loc[0]+dx, loc[1]+dy
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


def _future_chain_potential(loc: Tuple[int, int], board_state) -> int:
    """
    Estimate the maximum carpet chain that could be built from loc.

    Unlike _max_carpet_potential (which counts only primed cells),
    this counts all *available* cells — SPACE or PRIMED — in each cardinal
    direction until a BLOCKED, CARPET, or worker cell is hit.

    The result is CARPET_SCORE for the best direction, capturing how much
    long-run scoring power exists at this position.  This is the key
    heuristic term that Carrie's reference bot uses ("cell potential").
    """
    best = 0
    enemy_loc  = board_state.opponent_worker.get_location()
    player_loc = board_state.player_worker.get_location()

    for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
        length = 0
        nx, ny = loc[0]+dx, loc[1]+dy
        while board_state.is_valid_cell((nx, ny)):
            if (nx, ny) == enemy_loc or (nx, ny) == player_loc:
                break
            bit = 1 << (ny * BOARD_SIZE + nx)
            # BLOCKED or CARPET cells permanently end the chain
            if (board_state._blocked_mask | board_state._carpet_mask) & bit:
                break
            length += 1
            nx += dx
            ny += dy
        if length > 0:
            best = max(best, CARPET_SCORE.get(min(length, 7), 0))
    return best


def evaluate(board_state, rat_belief: RatBelief) -> float:
    """
    Evaluate board from the perspective of board_state.player_worker.
    Higher is better for us.

    Three components:
      1. Score differential
      2. Immediate carpet potential  (existing primed chain)
      3. Future chain potential      (buildable run of SPACE+PRIMED cells)
      4. Rat proximity / search EV   (only when belief is strong)
    """
    my  = board_state.player_worker
    opp = board_state.opponent_worker
    my_loc  = my.get_location()
    opp_loc = opp.get_location()

    # 1. Base Score
    score = float(my.get_points() - opp.get_points())

    # 2. Immediate Carpet Potential — already-primed runs ready to carpet
    my_carpet  = _max_carpet_potential(my_loc,  board_state)
    opp_carpet = _max_carpet_potential(opp_loc, board_state)
    score += 0.7 * my_carpet
    score -= 0.7 * opp_carpet

    # 3. Future Chain Potential — how long a chain *could* be built here.
    # This is the Carrie-level positional insight: a worker sitting at the
    # start of a long open corridor is worth much more than one in a corner.
    # Opponent penalty is slightly lighter: we can't actually block their run
    # (workers can't step on primed squares), so over-weighting it sends us
    # to the wrong side of the board.
    my_future  = _future_chain_potential(my_loc,  board_state)
    opp_future = _future_chain_potential(opp_loc, board_state)
    score += 0.3 * my_future
    score -= 0.2 * opp_future

    # 4. Rat Proximity — only pull toward rat when belief is meaningful.
    # Below p=0.2 the carpet income is more reliable than rat-hunting.
    rat_cell, rat_p, rat_ev = rat_belief.best_cell()
    if rat_p > 0.2:
        my_d  = manhattan(my_loc,  rat_cell)
        opp_d = manhattan(opp_loc, rat_cell)
        score += 4.0 * rat_p * (opp_d - my_d)
        if rat_ev > 0:
            score += rat_ev

    return score


# ===========================================================================
# Move ordering helper
# ===========================================================================

def quick_score(mv, rat_belief: RatBelief) -> float:
    """Fast greedy score used for move ordering in expectiminimax."""
    if mv.move_type == MoveType.CARPET:
        return float(CARPET_SCORE.get(mv.roll_length, 0))
    if mv.move_type == MoveType.PRIME:
        return 0.5
    if mv.move_type == MoveType.SEARCH:
        p = float(rat_belief.belief[xy_to_cell(*mv.search_loc)])
        return 6.0 * p - 2.0
    return 0.0   # PLAIN


# ===========================================================================
# Expectiminimax with alpha-beta pruning
# ===========================================================================

def expectiminimax(
    board_state,
    depth: int,
    rat_belief: RatBelief,
    alpha: float,
    beta: float,
    time_left: Callable,
) -> float:
    if depth == 0 or time_left() < 1.0:
        return evaluate(board_state, rat_belief)

    moves = board_state.get_valid_moves(exclude_search=True)
    if not moves:
        return evaluate(board_state, rat_belief)

    moves = sorted(moves, key=lambda m: quick_score(m, rat_belief), reverse=True)

    value = float('-inf')
    for mv in moves:
        if time_left() < 1.0:
            break
        try:
            child = board_state.forecast_move(mv)
            child.reverse_perspective()
            child_val = -expectiminimax(child, depth - 1, rat_belief, -beta, -alpha, time_left)
        except Exception:
            child_val = quick_score(mv, rat_belief)

        value = max(value, child_val)
        alpha = max(alpha, value)
        if alpha >= beta:
            break

    return value


# ===========================================================================
# PlayerAgent
# ===========================================================================

class PlayerAgent:
    """
    /you may add and modify functions, however, __init__, commentate and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):
        self.rat_belief: RatBelief | None = None
        self._tm = transition_matrix

        if transition_matrix is not None:
            self.rat_belief = RatBelief(transition_matrix)

        self._turns   = 0
        self._hits    = 0
        self._misses  = 0
        self._last_opp_search    = None
        # BUG FIX: the original code had a guard for opponent search but NOT for
        # player search. Without this, board.player_search persists across turns,
        # so the same search result gets re-applied every single turn, slowly
        # zeroing out probability mass and corrupting the entire belief.
        self._last_player_search = None

    def commentate(self):
        if self.rat_belief is not None:
            cell, p, ev = self.rat_belief.best_cell()
            return (
                f"Turns: {self._turns} | "
                f"Rat searches — hits: {self._hits}, misses: {self._misses} | "
                f"Final rat peak: {cell}  p={p:.3f}  EV={ev:.2f}"
            )
        return f"Turns: {self._turns} (no HMM — transition matrix was not provided)"

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        self._turns += 1

        # ------------------------------------------------------------------
        # 0. Lazy-init HMM if transition_matrix was not given in __init__
        # ------------------------------------------------------------------
        if self.rat_belief is None:
            tm = self._tm
            if tm is None:
                try:
                    tm = board.transition_matrix
                except AttributeError:
                    pass
            if tm is not None:
                self.rat_belief = RatBelief(tm)

        rb = self.rat_belief

        # ------------------------------------------------------------------
        # 1. Unpack sensor data  (noise_string, reported_distance)
        # ------------------------------------------------------------------
        noise, reported_dist = sensor_data

        # ------------------------------------------------------------------
        # 2. HMM update
        # ------------------------------------------------------------------
        if rb is not None:
            # 2a. Predict: rat moves one step according to T
            rb.predict()

            # 2b. Reweight by noise observation
            if noise is not None:
                rb.update_noise(noise, board)

            # 2c. Reweight by noisy distance sensor
            if reported_dist is not None:
                rb.update_distance(reported_dist, board.player_worker.get_location())

            # 2d. Incorporate opponent search result (guard against re-processing)
            opp_loc, opp_found = board.opponent_search
            if opp_loc is not None and opp_loc != self._last_opp_search:
                rb.update_search(opp_loc, opp_found)
                self._last_opp_search = opp_loc

            # 2e. Incorporate our own previous search result — only once per unique
            # search location. The original code had NO guard here, meaning the same
            # result was applied every subsequent turn, corrupting the belief.
            my_loc, my_found = board.player_search
            if my_loc is not None and my_loc != self._last_player_search:
                if my_found:
                    self._hits += 1
                else:
                    self._misses += 1
                rb.update_search(my_loc, my_found)
                self._last_player_search = my_loc

        # ------------------------------------------------------------------
        # 3. Get valid moves (searches handled separately below)
        # ------------------------------------------------------------------
        moves = board.get_valid_moves(exclude_search=True)

        # ------------------------------------------------------------------
        # 4. Opportunistic search: search when EV > 0 (p > 1/3)
        #    but only if search EV beats the best available carpet move.
        #    This prevents wasteful searches when we're sitting at a long chain.
        # ------------------------------------------------------------------
        if rb is not None:
            rat_cell, rat_p, rat_ev = rb.best_cell()
            if rat_ev >= SEARCH_EV_THRESHOLD:
                best_carpet_score = max(
                    (CARPET_SCORE.get(m.roll_length, 0) for m in moves
                     if m.move_type == MoveType.CARPET),
                    default=0
                )
                # Only search if its EV beats the best immediate carpet
                if rat_ev >= best_carpet_score:
                    return move.Move.search(rat_cell)

        # ------------------------------------------------------------------
        # 5. Iterative-deepening expectiminimax (guarded by time budget)
        # ------------------------------------------------------------------
        if moves and rb is not None:
            start_time = time.time()

            turns_left = max(1, board.player_worker.turns_left)
            allocated_time = max(0.5, min(4.0, (time_left() - 2.0) / turns_left))

            best_move = None

            for current_depth in range(1, 10):
                if time.time() - start_time > allocated_time / 2:
                    break

                moves_ordered = sorted(moves, key=lambda m: quick_score(m, rb), reverse=True)

                if best_move is None:
                    best_move = moves_ordered[0]

                depth_best_move = moves_ordered[0]
                best_val  = float('-inf')
                alpha     = float('-inf')
                beta      = float('inf')
                search_aborted = False

                for mv in moves_ordered:
                    if time.time() - start_time > allocated_time:
                        search_aborted = True
                        break

                    try:
                        child = board.forecast_move(mv)
                        child.reverse_perspective()
                        val = -expectiminimax(
                            child,
                            current_depth - 1,
                            rb,
                            -beta,
                            -alpha,
                            lambda: allocated_time - (time.time() - start_time)
                        )
                    except Exception:
                        val = quick_score(mv, rb)

                    if val > best_val:
                        best_val        = val
                        depth_best_move = mv
                    alpha = max(alpha, best_val)

                if not search_aborted:
                    best_move = depth_best_move

            if best_move is not None:
                return best_move

        # ------------------------------------------------------------------
        # 6. Greedy fallback  (carpet > prime > move-toward-rat > random)
        # ------------------------------------------------------------------
        if moves:
            return self._greedy(moves, board, rb)

        return move.Move.search((random.randint(0, 7), random.randint(0, 7)))

    # -----------------------------------------------------------------------
    # Helper: greedy move selection
    # -----------------------------------------------------------------------
    def _greedy(self, moves, board_state, rb):
        """
        Greedy fallback priority:
          1. Best-scoring carpet roll (roll_length >= 2)
          2. Prime step toward the highest future-chain-potential destination
          3. Plain step toward rat (only if p > 0.15, otherwise toward best future)
        """
        carpet_moves = []
        prime_moves  = []
        plain_moves  = []

        for mv in moves:
            if mv.move_type == MoveType.CARPET:
                carpet_moves.append(mv)
            elif mv.move_type == MoveType.PRIME:
                prime_moves.append(mv)
            elif mv.move_type == MoveType.PLAIN:
                plain_moves.append(mv)

        if carpet_moves:
            best = max(carpet_moves, key=lambda m: CARPET_SCORE.get(m.roll_length, -1))
            if CARPET_SCORE.get(best.roll_length, -1) > 0:
                return best

        my_loc = board_state.player_worker.get_location()

        if prime_moves:
            # Pick the prime direction that maximises future chain potential
            # at the destination.  This is the Carrie-level insight: don't just
            # prime randomly — position yourself at the start of the longest
            # available corridor.
            best_prime = max(
                prime_moves,
                key=lambda mv: _future_chain_potential(
                    _move_destination(mv, my_loc), board_state
                )
            )
            return best_prime

        if plain_moves:
            if rb is not None:
                rat_cell, rat_p, _ = rb.best_cell()
                if rat_p > 0.15:
                    # Navigate toward the rat when belief is strong
                    plain_moves.sort(
                        key=lambda m: manhattan(
                            _move_destination(m, my_loc), rat_cell
                        )
                    )
                    return plain_moves[0]
            # Otherwise drift toward the best future-chain-potential cell
            plain_moves.sort(
                key=lambda mv: _future_chain_potential(
                    _move_destination(mv, my_loc), board_state
                ),
                reverse=True
            )
            return plain_moves[0]

        return random.choice(moves)


# ===========================================================================
# Helper: compute destination of a plain/prime/carpet move
# ===========================================================================

_DIRECTION_DELTAS = {
    enums.Direction.UP:    (0, -1),
    enums.Direction.DOWN:  (0,  1),
    enums.Direction.LEFT:  (-1, 0),
    enums.Direction.RIGHT: (1,  0),
}

def _move_destination(mv, current_pos: Tuple[int, int]) -> Tuple[int, int]:
    """Compute where a move lands given the current position."""
    dx, dy = _DIRECTION_DELTAS.get(mv.direction, (0, 0))
    steps = mv.roll_length if mv.move_type == MoveType.CARPET else 1
    return (current_pos[0] + dx * steps, current_pos[1] + dy * steps)