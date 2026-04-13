from collections.abc import Callable
from math import dist
from typing import List, Set, Tuple
import random
import numpy as np

from game import board, move, enums


class RatBelief:
    """
    Hidden Markov Model for tracking rat belief distribution.
    Maintains probability distribution over 64 board positions.
    """
    
    def __init__(self, transition_matrix):
        """
        Initialize with transition matrix T (64x64 numpy array).
        """
        self.T = transition_matrix  # 64x64 transition matrix
        self.belief = np.ones(64) / 64  # uniform prior
    
    def initialize_uniform(self):
        """Reset belief to uniform distribution."""
        self.belief = np.ones(64) / 64
    
    def predict(self):
        """Predict next belief using transition matrix."""
        self.belief = self.T @ self.belief
        # Ensure normalization (in case of numerical issues)
        self.belief /= self.belief.sum()
    
    def update(self, board, worker_pos, observation):
        """
        Update belief with observation.
        observation: (noise_enum, distance_int)
        """
        noise, dist = observation
        
        # Noise probabilities from rat.py
        NOISE_PROBS = {
            enums.Cell.BLOCKED: (0.5, 0.3, 0.2),
            enums.Cell.SPACE: (0.7, 0.15, 0.15),
            enums.Cell.PRIMED: (0.1, 0.8, 0.1),
            enums.Cell.CARPET: (0.1, 0.1, 0.8),
        }
        
        # Distance error probabilities from rat.py
        DISTANCE_ERROR_PROBS = (0.12, 0.7, 0.12, 0.06)
        DISTANCE_ERROR_OFFSETS = (-1, 0, 1, 2)
        
        new_belief = np.zeros(64)
        
        for i in range(64):
            x, y = i % 8, i // 8
            pos = (x, y)
            
            # P(noise | position)
            cell = board.get_cell(pos)
            p_noise = NOISE_PROBS[cell][noise.value]
            
            # P(distance | position)
            true_dist = abs(pos[0] - worker_pos[0]) + abs(pos[1] - worker_pos[1])
            p_dist = 0.0
            for offset, prob in zip(DISTANCE_ERROR_OFFSETS, DISTANCE_ERROR_PROBS):
                reported = max(0, true_dist + offset)
                if reported == dist:
                    p_dist += prob
            
            # Update belief
            new_belief[i] = self.belief[i] * p_noise * p_dist
        
        total = new_belief.sum()
        if total > 0:
            self.belief = new_belief / total
        else:
            # Fallback to uniform if all probabilities zero
            self.initialize_uniform()
    
    def get_most_likely_position(self):
        """Return (x, y) of most likely rat position."""
        idx = np.argmax(self.belief)
        return (idx % 8, idx // 8)
    
    def get_belief_distribution(self):
        """Return current belief array."""
        return self.belief


def evaluate_board(board, rat_belief):
    """
    Heuristic evaluation function for board state.
    Higher values are better for the current player.
    """
    my_points = board.player_worker.get_points()
    opp_points = board.opponent_worker.get_points()
    score_diff = my_points - opp_points
    
    # Carpet potential: count primed cells (future scoring opportunities)
    primed_count = 0
    for x in range(8):
        for y in range(8):
            if board.get_cell((x, y)) == enums.Cell.PRIMED:
                primed_count += 1
    carpet_potential = primed_count * 0.5  # Weight for potential
    
    # Mobility: number of legal moves
    mobility = len(board.get_valid_moves())
    
    # Distance to rat belief hotspots
    rat_dist_penalty = 0
    if rat_belief:
        my_pos = board.player_worker.get_location()
        belief = rat_belief.get_belief_distribution()
        for i, prob in enumerate(belief):
            x, y = i % 8, i // 8
            dist = abs(x - my_pos[0]) + abs(y - my_pos[1])
            rat_dist_penalty += prob * dist
    rat_dist_penalty *= 0.2  # Weight
    
    # Expected value of searching (simplified: based on max belief prob)
    search_value = 0
    if rat_belief:
        belief = rat_belief.get_belief_distribution()
        max_prob = max(belief)
        search_value = max_prob * 4 + (1 - max_prob) * (-2)
    search_value *= 0.5  # Weight
    
    return score_diff + carpet_potential + mobility * 0.1 - rat_dist_penalty + search_value


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
        if transition_matrix is not None:
            self.transition_matrix = transition_matrix
            self.rat_belief = RatBelief(transition_matrix)
        else:
            self.transition_matrix = None
            self.rat_belief = None
        
    def commentate(self):
        """
        Optional: You can use this function to print out any commentary you want at the end of the game.
        """
        return "I'm a believer!"

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
        if self.rat_belief:
            worker_pos = board.player_worker.get_location()
            self.rat_belief.predict()
            self.rat_belief.update(board, worker_pos, sensor_data)
        if board.player_search[1]:
            self.rat_belief.initialize_uniform()
        
        # Greedy agent: evaluate all legal moves
        moves = board.get_valid_moves()
        best_move = moves[0]
        best_value = -float('inf')
        
        for move in moves:
            # Forecast the move
            new_board = board.forecast_move(move)
            if new_board is None:
                continue  # Invalid move, skip
            value = evaluate_board(new_board, self.rat_belief)
            if value > best_value:
                best_value = value
                best_move = move
        
        return best_move
