from collections.abc import Callable
from collections import Counter
from math import dist
from typing import List, Set, Tuple
import random
import numpy as np

from game import board, move, enums

NOISE_PROBS = {
    enums.Cell.BLOCKED: [0.5, 0.3, 0.2],
    enums.Cell.SPACE: [0.7, 0.15, 0.15],
    enums.Cell.PRIMED: [0.1, 0.8, 0.1],
    enums.Cell.CARPET: [0.1, 0.1, 0.8],
}

def get_dist_likelihood(actual, estimated):
    diff = estimated - actual
    if diff == 0:
        return 0.7
    elif diff == -1:
        return 0.12
    elif diff == 1:
        return 0.12
    elif diff == 2:
        return 0.06
    else:
        return 0.0

BOARD_SIZE = 8
NUM_PARTICLES = 200

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class RatBelief:
    def __init__(self, transition_matrix):
        self.T = transition_matrix
        self.particles = []

    # -------------------------
    # INIT
    # -------------------------
    def initialize_uniform(self):
        self.particles = [
            (random.randint(0, 7), random.randint(0, 7))
            for _ in range(NUM_PARTICLES)
        ]

    # -------------------------
    # PREDICT (rat motion)
    # -------------------------
    def predict(self):
        new_particles = []

        for x, y in self.particles:
            idx = y * 8 + x

            probs = self.T[idx]

            r = random.random()
            cum = 0.0

            for i, p in enumerate(probs):
                cum += p
                if r < cum:
                    nx, ny = i % 8, i // 8
                    new_particles.append((nx, ny))
                    break
                else:
                    new_particles.append((x, y))  # stay in place as fallback

        self.particles = new_particles

    # -------------------------
    # UPDATE (sensor correction)
    # -------------------------
    def update(self, board, worker_pos, sensor_data):
        noise, est_dist = sensor_data

        noise_idx = {
            enums.Noise.SQUEAK: 0,
            enums.Noise.SCRATCH: 1,
            enums.Noise.SQUEAL: 2
        }[noise]

        weights = []

        for (x, y) in self.particles:
            cell = board.get_cell((x, y))

            noise_prob = {
                enums.Cell.BLOCKED: [0.5, 0.3, 0.2],
                enums.Cell.SPACE: [0.7, 0.15, 0.15],
                enums.Cell.PRIMED: [0.1, 0.8, 0.1],
                enums.Cell.CARPET: [0.1, 0.1, 0.8],
            }[cell][noise_idx]

            actual_dist = manhattan((x, y), worker_pos)

            # distance likelihood (same as yours but inline)
            diff = est_dist - actual_dist
            if diff == 0:
                dist_lik = 0.7
            elif diff in (-1, 1):
                dist_lik = 0.12
            elif diff == 2:
                dist_lik = 0.06
            else:
                dist_lik = 0.01

            weights.append(noise_prob * dist_lik)

        # normalize weights
        total = sum(weights)
        if total == 0:
            self.initialize_uniform()
            return

        # resample
        new_particles = random.choices(
            self.particles,
            weights=weights,
            k=NUM_PARTICLES
        )

        self.particles = new_particles

    # -------------------------
    # QUERY
    # -------------------------
    def get_most_likely(self):
        return Counter(self.particles).most_common(1)[0][0]

    def get_distribution(self):
        return Counter(self.particles)
    
    def get_prob(self, x, y):
        dist = Counter(self.particles)
        return dist[(x, y)] / len(self.particles)

    def get_heatmap(self):
        heat = np.zeros((8, 8))
        for x, y in self.particles:
            heat[y][x] += 1
        return heat / len(self.particles)

def evaluate_board(board, rat_belief):
    # Heuristic evaluation function for board state.
    my_points = board.player_worker.get_points()
    opp_points = board.opponent_worker.get_points()
    score_diff = my_points - opp_points
    
    # count primed cells
    primed_count = 0
    for x in range(8):
        for y in range(8):
            if board.get_cell((x, y)) == enums.Cell.PRIMED:
                primed_count += 1
    carpet_potential = primed_count * 0.5  # Reduced weight to discourage over-priming
    
    # number of legal moves
    mobility = len(board.get_valid_moves(exclude_search=False))
    
    # Expected value of searching based on belief
    search_value = 0
    if rat_belief:
        heat = rat_belief.get_heatmap() if rat_belief else np.ones((8,8)) / 64
        max_prob = float(np.max(heat))
        search_value = 4 * max_prob - 2
        if rat_belief:
            wx, wy = board.player_worker.get_location()
        chase_value = 0
        for x in range(8):
            for y in range(8):
                prob = heat[y][x]
                d = abs(x - wx) + abs(y - wy)
                chase_value += prob * (10 - d)
        search_value += chase_value
    
    return score_diff + carpet_potential + mobility * 0.05 + search_value

class PlayerAgent:
    """
    /you may add and modify functions, however, __init__, commentate and play are the entry points for
    your program and should not be changed.
    """

    def __init__(self, board, transition_matrix=None, time_left: Callable = None):

        """
        TODO: Your initialization code below. Should be used to do any setup you want
        before the game begins (i.e. calculating priors.)
        """

        # Setting up initial belief for where the rat is and transition matrix for how the rat moves
        if transition_matrix is not None:
            self.rat_belief = RatBelief(transition_matrix)
        else:
            self.rat_belief = None

    def commentate(self):
        """
        Optional: You can use this function to print out any commentary you want at the end of the game.
        """
        return "Sekun's agent"
        
    def minimax(self, board, depth, rat_belief, time_left, maximizing=True):
        if depth == 0 or time_left() < 5:
            return evaluate_board(board, rat_belief)

        moves = board.get_valid_moves(exclude_search=False)
        if not moves:
            return evaluate_board(board, rat_belief)

        if maximizing:
            best = -float('inf')
            for m in moves:
                nb = board.forecast_move(m)
                if nb is None:
                    continue
                best = max(best, self.minimax(nb, depth-1, rat_belief, time_left, False))
            return best
        else:
            best = float('inf')
            for m in moves:
                nb = board.forecast_move(m)
                if nb is None:
                    continue
                best = min(best, self.minimax(nb, depth-1, rat_belief, time_left, True))
            return best

    def play(
        self,
        board: board.Board,
        sensor_data: Tuple,
        time_left: Callable,
    ):
        """
        TODO: Below is random mover code. Replace it with your own.
        You may do so however you like, including adding extra functions,
        variables. Return a valid move from this function.
        """
        # Update rat belief with observation
        if self.rat_belief is not None:
            worker_pos = board.player_worker.get_location()
            self.rat_belief.predict()
            self.rat_belief.update(board, worker_pos, sensor_data)
        if board.player_search[1]:
            self.rat_belief.initialize_uniform()
        
        # Decide search depth based on time
        t = time_left()
        depth = 3 if t > 60 else 2 if t > 20 else 1
        
        if depth == 1:
            # evaluate all legal moves
            moves = board.get_valid_moves(exclude_search=False)
            move_values = []
            for move in moves:
                # Forecast the move
                new_board = board.forecast_move(move)
                if new_board is None:
                    continue  # Invalid move
                value = evaluate_board(new_board, self.rat_belief)
                # Add bonus for carpet moves to encourage immediate scoring
                if hasattr(move, 'move_type') and move.move_type == enums.MoveType.CARPET:
                    value += 5  # Bonus for carpeting moves
                # Add bonus for search moves
                if hasattr(move, 'move_type') and move.move_type == enums.MoveType.SEARCH:
                    x, y = move.search_loc
                    idx = y * 8 + x
                    prob = self.rat_belief.particles.count((x, y)) / NUM_PARTICLES if self.rat_belief else 0
                    value += 6 * prob - 2  # Expected value of searching this position
                move_values.append((value, move))
            # Sort by value descending
            move_values.sort(reverse=True)
            best_move = move_values[0][1] if move_values else moves[0]
            
            return best_move
        else:
            # Minimax with depth 2
            moves = board.get_valid_moves(exclude_search=False)
            best_move = moves[0]
            best_value = -float('inf')
            
            for move in moves:
                new_board = board.forecast_move(move)
                if new_board is None:
                    continue
                value = self.minimax(new_board, 1, self.rat_belief, time_left)  # depth 1 for opponent turn
                if value > best_value:
                    best_value = value
                    best_move = move
            
            return best_move