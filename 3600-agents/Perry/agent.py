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

# Search EV = 6p - 2.  Break-even at p = 1/3.
# Require p >= 0.55 (EV >= 1.3) before considering a search.
# Carpet-first strategy: only hunt the rat when the payoff is compelling.
SEARCH_P_THRESHOLD = 0.55

# Don't search at all if we have a primed chain this long ready to roll.
# Finishing a 2+ cell chain is worth more than a speculative rat search.
SEARCH_SUPPRESS_IF_CARPET_GTE = 3

# After a miss, wait this many turns before searching again.
# Prevents cascade-search disasters (3 misses in a row = -6 pts).
MISS_COOLDOWN_TURNS = 3


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
        ev = 6.0 * p - 2.0
        return cell_to_xy(idx), p, ev

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
# Heuristic helpers — cheap raycasts only, no forecast_move calls
# ===========================================================================

def _max_carpet_potential(loc, board_state) -> int:
    """
    Best immediate carpet roll from loc.
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
            if (board_state._blocked_mask | board_state._carpet_mask) & bit:
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
    Each primed cell ahead = +3 move-ordering points.
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


# ===========================================================================
# Static evaluation
# ===========================================================================
def reachable_space(loc: Tuple[int, int], board_state) -> int:
    """Flood-fill to determine positional dominance and prevent getting trapped."""
    visited = set()
    stack = [loc]
    while stack:
        cur = stack.pop()
        if cur in visited: continue
        visited.add(cur)
        
        for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
            nx, ny = cur[0]+dx, cur[1]+dy
            if not board_state.is_valid_cell((nx, ny)): continue
            
            bit = 1 << xy_to_cell(nx, ny)
            if (board_state._blocked_mask | board_state._carpet_mask) & bit:
                continue
            stack.append((nx, ny))
    return len(visited)

def evaluate(board_state, rat_belief: RatBelief, depth_simulated: int = 0) -> float:
    my = board_state.player_worker
    opp = board_state.opponent_worker
    my_loc = my.get_location()
    opp_loc = opp.get_location()

    # 1. Base Score Difference
    score = 30.0 * (my.get_points() - opp.get_points())

    # 2. Dynamic Threat Multiplier & Stolen Carpet Paranoia (Symmetric)
    my_carpet = _max_carpet_potential(my_loc, board_state)
    opp_carpet = _max_carpet_potential(opp_loc, board_state)

    worker_dist = manhattan(my_loc, opp_loc)
    
    if worker_dist <= 2:
        carpet_weight = 5.0    # High threat panic: Roll instantly!
    elif worker_dist <= 4:
        carpet_weight = 17.0   # Medium threat: Cash out medium lengths
    else:
        carpet_weight = 22.0   # Safe: Patiently build massive chains

    score += carpet_weight * my_carpet
    chain_now = _adjacent_primed_chain(my_loc, board_state)
    if chain_now >= 3:
        score += 5 * CARPET_SCORE.get(chain_now, 0)
    score -= carpet_weight * opp_carpet

    # 3. Future Chain Potential
    horizon = max(0.1, (my.turns_left or 1) / 40.0)
    score += 4.0 * _future_chain_potential(my_loc, board_state) * horizon
    score -= 4.0 * _future_chain_potential(opp_loc, board_state) * horizon

    # 4. Mobility / Open Space
    my_space = reachable_space(my_loc, board_state)
    opp_space = reachable_space(opp_loc, board_state)
    score += 0.5 * my_space
    score -= 0.5 * opp_space

    # 5. Rat Hunting
    if rat_belief is not None:
        decay = 0.85 ** depth_simulated
        if my_carpet <= 2:
            my_heat = rat_belief.inverse_distance_heat(my_loc)
            my_dist = rat_belief.expected_distance(my_loc)
            score += (8.0 * my_heat - 0.5 * my_dist) * decay
            
        if opp_carpet <= 2:
            opp_heat = rat_belief.inverse_distance_heat(opp_loc)
            opp_dist = rat_belief.expected_distance(opp_loc)
            score -= (8.0 * opp_heat - 0.5 * opp_dist) * decay

    return score
# ===========================================================================
# Move ordering — cheap raycasts, NO forecast_move calls
# ===========================================================================

def quick_score(mv, board_state, rat_belief: RatBelief) -> float:
    """
    Fast static move ordering to maximize alpha-beta cutoffs.

    IMPORTANT: this must NOT call forecast_move(). Calling forecast_move here
    means O(moves) extra forecasts per interior node just for sorting — this
    kills search depth. Cheap raycasts give nearly identical ordering quality.

    PRIME moves get a chain continuation bonus: if priming in direction d puts
    us adjacent to already-primed cells in that same direction, a carpet roll
    is one move away — strongly prefer extending the chain.
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
        return 1.0

    # SEARCH is excluded from tree generation
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
        
        # New Detailed Stat Trackers
        self._primes_done = 0
        self._carpets_made = 0
        
        self._tt = {}
        self._last_turns_remaining = None

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
            return move.Move.search((random.randint(0, 7), random.randint(0, 7)))
        if mv.move_type == MoveType.CARPET:
            self._carpets_made += 1
        elif mv.move_type == MoveType.PRIME:
            self._primes_done += 1
        return mv

    def _state_key(self, board):
        """Builds a hashable key for the Transposition Table."""
        return (
            board.player_worker.get_location(),
            board.opponent_worker.get_location(),
            board.player_worker.get_points(),
            board.opponent_worker.get_points(),
            board._primed_mask,
            board._carpet_mask,
        )
    
    # ------------------------------------------------------------------
    # Negamax with alpha-beta pruning
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
        
        cached = tt.get(state_key)
        if cached is not None and cached[0] >= depth:
            return cached[1]

        if depth == 0 or (board_state.player_worker.turns_left or 0) == 0:
            val = evaluate(board_state, rat_belief, depth_simulated)
            tt[state_key] = (depth, val)
            return val

        moves = list(board_state.get_valid_moves(exclude_search=True))
        if not moves:
            val = evaluate(board_state, rat_belief, depth_simulated)
            tt[state_key] = (depth, val)
            return val

        moves.sort(key=lambda m: quick_score(m, board_state, rat_belief), reverse=True)

        if depth <= 1: moves = moves[:8]
        elif depth <= 2: moves = moves[:10]
        else: moves = moves[:14]

        best = -float("inf")
        for mv in moves:
            if time.time() > end_time:
                raise SearchTimeout()

            child = board_state.forecast_move(mv)
            if child is None: continue

            child.reverse_perspective()
            val = -self._negamax(child, depth - 1, rat_belief,
                        -beta, -alpha, end_time, tt,
                        depth_simulated + 1)

            if val > best: best = val
            alpha = max(alpha, best)
            if alpha >= beta: break

        if best == -float("inf"):
            best = evaluate(board_state, rat_belief, depth_simulated)

        tt[state_key] = (depth, best)
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
                else:
                    self._misses += 1
                rb.update_search(my_search_loc, my_found)
                self._last_my_search_turn = self._turns

            # CRITICAL FIX: The rat moves twice between your turns! 
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

        if turns_left == 1:
            carpet_moves = [m for m in moves if m.move_type == MoveType.CARPET]
            if carpet_moves:
                return max(carpet_moves, key=lambda m: CARPET_SCORE.get(m.roll_length, -1))

        # 2. Opportunistic Search (High confidence only)
        if rb is not None and self._miss_cooldown == 0:
            rat_cell, rat_p, rat_ev = rb.best_cell()
            my_score = board.player_worker.get_points()
            opp_score = board.opponent_worker.get_points()
            
            ev_threshold = 1.3  # Standard: requires ~55% probability
            if my_score < opp_score:
                ev_threshold = 0.8  # Trailing: take more risks (requires ~46% prob)
            if turns_left <= 10:
                ev_threshold = max(0.5, ev_threshold - 0.5) # Desperate endgame

            if rat_ev >= ev_threshold:
                my_loc_now = board.player_worker.get_location()
                my_carpet_now = _max_carpet_potential(my_loc_now, board)
                
                # Do not waste a turn searching if we have a prime chain of 4+ points ready
                if my_carpet_now < 4:
                    return move.Move.search(rat_cell)
                
        # 3. Iterative-deepening Alpha-Beta Search
        if moves and rb is not None:
            if not hasattr(self, '_tt'): self._tt = {}
            self._tt.clear()
            
            start_time = time.time()
            turns_left = max(1, board.player_worker.turns_left or 1)
            
            # Custom Time Budgeting: Hoard time for the mid-game (turns 15-30)
            safe_buffer = 1.2
            usable_time = max(0.1, time_left() - safe_buffer)
            if 15 <= turns_left <= 30:
                allocated = min(4.0, usable_time / 3.0)
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
                for depth in range(1, 15):  # Will automatically break via timeout
                    if time.time() > end_time:
                        break
                        
                    alpha = -float("inf")
                    beta = float("inf")
                    best_val_this_depth = -float("inf")

                    # Principal Variation Sorting: Force the best move to the front
                    if global_best_move in moves_ordered:
                        moves_ordered.remove(global_best_move)
                        moves_ordered.insert(0, global_best_move)

                    for mv in moves_ordered:
                        if time.time() > end_time:
                            raise SearchTimeout()
                            
                        child = board.forecast_move(mv)
                        if child is None: continue
                        
                        child.reverse_perspective()
                        val = -self._negamax(child, depth - 1, rb, -beta, -alpha, end_time, self._tt)

                        if val > best_val_this_depth:
                            best_val_this_depth = val
                            global_best_move = mv  # Safely lock it in!

                        alpha = max(alpha, best_val_this_depth)
                        
            except SearchTimeout:
                # We ran out of time! Cleanly break out.
                pass 

            return self._return_and_track(global_best_move)
            
        # 4. Greedy fallback
        if moves: 
            return self._return_and_track(self._greedy(moves, board, rb))
            
        return move.Move.search((random.randint(0, 7), random.randint(0, 7)))

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
                    score -= 10  # 🔥 discourage over-priming when cash-out is ready

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
