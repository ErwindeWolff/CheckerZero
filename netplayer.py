from __future__ import division
from game import *
from nettrees import *
from network import *
from random import random
from math import sqrt
from chainer.serializers import load_npz
import copy
import numpy as np

""" Player class hat uses the checkerzero network to play the game """

class Netplayer():

	def __init__(self, C=1, nr_samples=1000, nondeterm_moves=15):

		# Set name, C and number of samples for the MTCS
		self.name = 'CheckerZero'
		self.C = C
		self.nr_samples = nr_samples
		
		# State that is kept in check
		self.played_moves = 0
		self.state = False
		
		# Number of moves before player choose deterministically
		self.nondeterm_moves = nondeterm_moves
		self.played_moves = 0
		
		# Initialise the network
		self.network = Classifier(32, 5, 4096)
		
		# Encodes what value corresponds to what
		# index, here: value 1 encoding to index 0,
		# value 3 to index 1 etc.
		self.value_links = dict()
		self.value_links['1'] = 0
		self.value_links['3'] = 1
		self.value_links['2'] = 2
		self.value_links['4'] = 3
		self.nr_links = 4
		
		print("CheckerZero reporting for duty")
	
	
	'''
		Sets the player to its basic settings for new game
	'''
	def reset(self):
		self.played_moves = 0
		self.state = Node()
	
	
	'''
		Finds a move given the game state and the current player to move
		
		- game: the game state
		- current_state: the player to move in the position
		
		Returns the move to play
	'''
	def get_move(self, game, additional_info=False):
	
		# Get legal moves
		moves = game.get_moves()
		
		# Turn board into network input
		net_input = self.board_to_input(game)
		
		# Determine prior policy (and value)
		if not self.state:
			self.state = Node()
		
		# Perform MCTS
		for i in range(self.nr_samples):
			new_game = copy.deepcopy(game)
			self.MCTS_sample(new_game, self.state)
		
		# Determine prior policy
		p = [branch.prob for branch in self.state.branches]
		
		# Determine posterior policy
		sum_visits = sum([branch.visits for branch in self.state.branches])
		pi = [branch.visits /sum_visits for branch in self.state.branches]
		
		# Sample next state from policy distribution
		if (self.played_moves < self.nondeterm_moves):
			move = self.sample_distribution(moves, pi)
		else:
			move = max([(p, m) for p, m in zip(pi, moves)])[1]
		
		# If wanted return info about prior, posterior and value
		if additional_info:
			return (move, net_input, moves, pi, game.current_player)
			
		# Else return selected move only
		else:		
			return move
	
		
	'''
		Updates the head of the player
		search tree to the most recent move.
		Also adds a play to the played moves.
	'''
	def update_state(self, move):

				# Create new leaf
		next_state = Node()

		# Try to find branches that match the played move
		try:
			for branch in self.state.branches:
				if branch.move == move:
					next_state = branch
					break
		except:
			pass
		self.state = next_state
		
		# Increments played moves by one ply (half move)
		self.played_moves += 0.5
			
		
	'''
		Turns a game state into input for the network
		
		- game: the game being played
		
		Returns input for the network
	'''
	def board_to_input(self, game):
		
		# Input size: number of different possibles pieces
		# (as encoded by value_links) by board size
		net_input = np.zeros((self.nr_links+1, game.columns, game.rows))
		
		# For each location on the board:
		for y in range(game.rows):
			for x in range(game.columns):
				index = y*game.columns + x
				
				# Get value from the board
				value = game.board[index]
				
				# Add pieces in their associated input layer
				if value != '0':
					l = self.value_links[value]
					net_input[l, x, y] = 1

		# Make final layer colour of player whose turn it is
		net_input[self.nr_links,:,:] = game.current_player
	
		return net_input


	'''
				Performs the actual Monte Carlo Tree search from a certain state in a game
				- game: the game being played
		- state: state of the board
		
		Returns the winner for this particular game and state
	'''
	def MCTS_sample(self, game, state):
	
		# Set state to current player
		state.current_player = game.current_player
		
		# If the node is in a terminal state
		if game.game_over():
			# Get score from game itself
			winner = game.get_score()

		# If state is leaf, apply expand and evaluate
		elif state.is_leaf():
		
			# Get move and current player from game
			moves = game.get_moves()
			
			# Convert board state to proper input for network
			net_input = self.board_to_input(game)
		
			# Get prior policy information and value estimation from network
			padded_input = np.zeros( (1, net_input.shape[0], net_input.shape[1], net_input.shape[2]))
			padded_input[0, :, :, :] = net_input
			(policy, value) = self.network.get_policy_and_value(padded_input.astype(np.float32))

			# Add noise over policy
			policy = [(0.75*pol+0.25*random()) for pol in policy]
			
			# Get relevant scores from posterior policy
			distribution = self.get_policy_dist(game, moves, policy)
			
			# Create branches for each move
			for move, prob in zip(moves, distribution):
				branch = Node()
				branch.move = move
				branch.prob = prob
				state.branches.append(branch)
			
			winner = value
			
			
		else: # Node was expanded, selection still in progress
			next_state = False
			value = 0
			# Find branch with highest sum of Q and U values
			for branch, u_value in zip(state.branches, self.get_u_values(state)):
				v = branch.mean_value + u_value
				if (not next_state) or (v > value):
					next_state = branch
					value = v
					
			# Play this move in the game
			game.move(next_state.move)
			
			# Recursively determine winner
			winner = self.MCTS_sample(game, next_state)
		

		# Backup phase of MCTS
		state.visits += 1
		state.total_value += winner
		state.mean_value = state.total_value/state.visits
		
		return winner

	
	'''
		Determines U(s, a) for all a in the current state
		
		- state: s
		- policy: a vector P(s,a)
		
		Returns list of U(s,a)
	'''
	def get_u_values(self, state):
		# Determine sum of all children
		sum_child_visits = sum([b.visits for b in state.branches])
		
		# Find U-value for all actions in state
		U_values = [self.C * b.prob * (sqrt(sum_child_visits)/(b.visits + 1)) 
						for b in state.branches]
		
		return U_values


	'''
		Get the distribution over the moves from the network output
		
		- moves: the list of legal moves (x1, y1, x2, y2)
		- policy: the policy output of the network
		
		Returns probability vector for each of the legal moves
	'''
	def get_policy_dist(self, game, moves, policy):
		
		# Variables to keep score
		distribution = list()
		dist_sum = 0
		
		# For each move, get the value in the policy output
		for move in moves:
			(x1, y1, x2, y2) = [x-1 for x in move]
			index = ((y1*game.columns) + x1) * (game.rows*game.columns) + ((y2*game.columns) + x2)				
			distribution.append(policy[index])
			dist_sum += policy[index]
			
		# Return normalized probabilities
		return [p/dist_sum for p in distribution]		
			
			
	'''
		Single sampling algorithm from distribution. It requires that the to_sample is a list
		as long as distribution, and that the values in distribution sum to 1.
		
		- to_sample: list of elements to sample from.
		- distribution: probablities relating to elements in to_sample
		
		Returns single sample from to_sample
	'''		
	def sample_distribution(self, to_sample, distribution):
		choice = random()
		sample = to_sample[0]
		for i, prob in enumerate(distribution):
			if prob > choice:
				sample = to_sample[i]
				break
		return sample


		'''
				Loads a trained model in the player
		'''

	def load_network(self, path, save_name):
		save_path = path + save_name
		serializers.load_npz(save_path, self.network, path='', strict=True)
