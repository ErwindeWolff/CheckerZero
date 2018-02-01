from __future__ import division
from game import *
from trees import *
from random import randint, random
from math import sqrt, log
import copy

""" Classes that define the Random player and the MCTS player (baseline players) """


""" Class that defines the random player """
class RandomPlayer():

	def __init__(self):
		
		self.name = 'Random'
		self.wins = 0

	'''
                Resets the random player (does nothing, needed for Netplayer)
	'''
	def reset(self):
		return

	'''
                Returns a random move from the possible moves
	'''
	def get_move(self, game):
		# Get possible moves
		moves = game.get_moves()
		
		# Select random move from list
		return moves[randint(0, len(moves)-1)]

	'''
                Update the state of the random player (does nothing, needed for Netplayer)
	'''
	def update_state(self, move):
		return


""" Class that defines the MCTS player """
class MCTSPlayer():

	def __init__(self, nr_samples, C=1):
		
		self.name = 'MCTS'
		self.wins = 0
		
		# Specific values
		self.nr_samples = nr_samples
		self.C = C

	'''
                Resets the MCTS player (does nothing, needed for Netplayer)
	'''
	def reset(self):
		return

	'''
                Returns the best move based on the MCTS samples
	'''
	def get_move(self, game):
	
		state = Node()
		
		moves = game.get_moves()
		if len(moves) == 1:
			return moves[0]
		
		for _ in range(self.nr_samples):
			new_game = copy.deepcopy(game)
			self.MCTS_sample(new_game, state)
		
		final_move = moves[0]
		max_value = -9999
		for (move, child) in state.children:
		
			to_maximise = child.value*child.current_player*state.current_player
		
			if to_maximise > max_value:
				max_value = to_maximise
				final_move = move
		
		return final_move

	'''
                Draws a MCTS sample from the current state
	'''
	def MCTS_sample(self, game, state):
	
		# Get move and current player from game
		moves = game.get_moves()
		current_player = game.current_player
		state.current_player = current_player
		
		# Increment the visits in this node
		state.visits += 1
		
		# If the node is in a terminal state
		if game.game_over():
			# Get score from game itself
			winner = game.get_score()

		# If all children of node are expanded
		elif state.is_fully_expanded(moves):
		
			# Get best move via exploration/exploitation trade-off
			(move, next_state) = self.UCB_sample(state)
			# Apply this move
			game.move(move)
			# Recursively look for leaf node
			winner = self.MCTS_sample(game, next_state)
			
		else:
			# Expand children
			(move, new_state) = state.expand(game, moves)
			game.move(move)
			new_state.current_player = game.current_player
			
			# Play random game from the new state
			winner = self.random_playout(game)
		
		# Update the value based on the winner
		self.update_value(state, winner, current_player)
		return winner
		
	'''
              Sample move by applying the UCB formula  
	'''
	def UCB_sample(self, state):
		
		# Get weights of all children
		weights = list()
		sum_weights = 0
		for (_, child) in state.children:
			# Apply formula
			w = child.value/child.visits + self.C * sqrt(log(state.visits) / child.visits)
			weights.append(w)
			sum_weights += w
		
		# Normalize weights into probability distribution
		distribution = [w / sum_weights for w in weights]
		
		# Select a random move to play
		choice = random()	
		# Default value
		(move, next_state) = state.children[0]			
		for i, prob in enumerate(distribution):
			# If randomly selected
			if prob > choice:
				# Set values and break out of loop
				(move, next_state) = state.children[i]
				break
		return (move, next_state)
		
		
	'''
                Update value of the state based on if current player wins in this sampling
	'''		
	def update_value(self, state, winner, current_player):
		# If the current player won the random game, increment the value
		if winner == current_player:
			state.value += 1
		# Else if it was a draw, increments less so
		elif winner == 0:
			state.value += 0.5
		# In case of loss don't adjust the value


	'''
                After a move has been chosen by MCTS, finish the game randomly
	'''
	def random_playout(self, game):
		# Create random player
		p1 = RandomPlayer()
		p2 = RandomPlayer()
		
		# Play game until end
		while not game.game_over():
			if game.current_player == 1:	
				move = p1.get_move(game)
			else:
				move = p2.get_move(game)
			game.move(move)
		# Return score of game
		return game.get_score()

	'''
                Update the state of the random player (does nothing, needed for Netplayer)
	'''
	def update_state(self, move):
		return
