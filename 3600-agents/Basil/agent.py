from collections.abc import Callable
from math import dist
from typing import List, Set, Tuple
import random
import jax.numpy as jnp

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

class RatBelief:
    def __init__(self, transition_matrix):
        self.transition_matrix = jnp.array(transition_matrix)
        self.belief = jnp.ones(64) / 64

    def predict(self):
        self.belief = self.transition_matrix @ self.belief

    def update(self, board, worker_pos, sensor_data):
        noise, estimated_distance = sensor_data
        noise_idx = {enums.Noise.SQUEAK: 0, enums.Noise.SCRATCH: 1, enums.Noise.SQUEAL: 2}[noise]
        likelihood = jnp.zeros(64)
        for i in range(64):
            x, y = i % 8, i // 8
            cell_type = board.get_cell((x, y))
            noise_prob = NOISE_PROBS[cell_type][noise_idx]
            actual_dist = abs(x - worker_pos[0]) + abs(y - worker_pos[1])
            dist_lik = get_dist_likelihood(actual_dist, estimated_distance)
            likelihood = likelihood.at[i].set(noise_prob * dist_lik)
        self.belief = self.belief * likelihood
        total = jnp.sum(self.belief)
        if total > 0:
            self.belief = self.belief / total
        else:
            self.belief = jnp.ones(64) / 64  # reset if all zero

    def get_belief_distribution(self):
        return self.belief

    def initialize_uniform(self):
        self.belief = jnp.ones(64) / 64

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
        belief = rat_belief.get_belief_distribution()
        max_prob = jnp.max(belief)
        search_value = 6 * max_prob - 2
    search_value *= 10.0
    
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
        
    def minimax(self, board, depth, rat_belief, time_left):
        if depth == 0 or time_left() < 5:
            return evaluate_board(board, rat_belief)
        
        moves = board.get_valid_moves(exclude_search=False)
        if not moves:
            return evaluate_board(board, rat_belief)
        
        if depth % 2 == 1:  # My turn (maximizing)
            best = -float('inf')
            for move in moves:
                new_board = board.forecast_move(move)
                if new_board is None:
                    continue
                val = self.minimax(new_board, depth - 1, rat_belief, time_left)
                best = max(best, val)
            return best
        else:  # Opponent's turn (minimizing)
            # Swap players to get opponent moves
            temp_player = board.player_worker
            temp_opp = board.opponent_worker
            board.player_worker = temp_opp
            board.opponent_worker = temp_player
            opp_moves = board.get_valid_moves(exclude_search=False)
            board.player_worker = temp_player
            board.opponent_worker = temp_opp
            
            best = float('inf')
            for move in opp_moves:
                # Swap for forecasting
                board.player_worker = temp_opp
                board.opponent_worker = temp_player
                new_board = board.forecast_move(move)
                board.player_worker = temp_player
                board.opponent_worker = temp_opp
                if new_board is None:
                    continue
                # Swap back in new_board
                new_board.player_worker = temp_player
                new_board.opponent_worker = temp_opp
                val = evaluate_board(new_board, rat_belief)
                best = min(best, val)
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
        depth = 2 if time_left() > 30 else 1
        
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
                if hasattr(move, 'move_type') and move.move_type == enums.MoveType.CARPET_ROLL:
                    value += 5  # Bonus for carpeting moves
                # Add bonus for search moves
                if hasattr(move, 'move_type') and move.move_type == enums.MoveType.SEARCH:
                    x, y = move.position
                    idx = x * 8 + y
                    prob = self.rat_belief.belief[idx]
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