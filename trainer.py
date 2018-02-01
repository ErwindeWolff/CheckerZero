from __future__ import division
from game import *
from window import *
from player import *
from netplayer import *
from math import ceil
from chainer import optimizers, serializers, Variable
import os, os.path

""" Class to train a CheckerZero network for a certain amount of epochs and MCTS samples (call is at the bottom of this file) """

class Trainer():

	def __init__(self, epochs, mcts_samples, path, filename, load_network=False):
	
		# Create player
		self.player = Netplayer(C=1, nr_samples=mcts_samples, nondeterm_moves=15)

		# Create game
		self.game = Game(rows=8, columns=8)

                # Set model
		self.model = self.player.network
		
		# Set epochs
		self.epochs = epochs

		# Set batch size
		self.batch_size = 1
		
		# Set paths
		self.path = path
		self.filename = filename
		
		# If a network must be loaded
		if load_network:
			self.load_network()
			
		# Create optimizer
		self.optimizer = optimizers.MomentumSGD(lr=0.1)
		self.optimizer.setup(self.model)
	

        '''
                Trains the network of the player via mini batches 
                and writes the loss to a file
        '''
	def train(self):

                # Train for given amounts of epochs (equal to finished games)
		for i in range(self.epochs):
			print(i+1)
			
                        # Initialize input and targets of network
			x = list()
			t_policy = list()
			t_value = list()
			current_players = list()

			# Play a full game
			while not self.game.game_over():

                                # Get move from the network player and append to input of network variable x
				(move, net_input, moves, pi, current_player) = self.player.get_move(self.game, additional_info = True)
				x.append(net_input)
				
				# Save the policy in the network output shape
				pi_vector = np.zeros((4096, 1))
				for m, p in zip(moves, pi):
					(x1, y1, x2, y2) = [n-1 for n in m]
					index = ((y1*self.game.columns) + x1) * (self.game.rows*self.game.columns) + ((y2*self.game.columns) + x2)
					pi_vector[index] = p

				t_policy.append(pi_vector)
				
				# Remember whose turn it was
				current_players.append(current_player)
				
				# Apply move in game and update state of player
				self.game.move(move)
				self.player.update_state(move)

			# Get score to be positive if that player won, negative otherwise
			t_value = [self.game.get_score() * cur_player for cur_player in current_players]

                        loss = []
                        # Create loop for mini-batches
                        for mb in range(0, int(ceil(len(t_value)/self.batch_size))):
                                
                                # Create batches
                                fro = self.batch_size*mb
                                to = self.batch_size*(mb+1)

                                if to >= len(t_value):
                                        to = len(t_value)
                                        
                                # Train the network on the moves
                                loss.append(self.update_network(x[fro:to], t_policy[fro:to], t_value[fro:to]))
                                
                        # Reset player and board
                        self.player.reset()
                        self.game.init_board()
                                
                        # Save every 10 games
                        if ((i+1)%10 == 0):
                                self.save_network()

                        # Save loss every game and write to file	
                        f = open("loss_results.txt", "a")
                        f.write(str(np.mean(loss)) + "\n")
                        f.close()
	
	'''
                Function that handles the actual updating of the network by
                clearing the gradients, performing backpropagation and letting
                the optimizer update the model
        '''
	def update_network(self, x, t_policy, t_value):
	
		# Cast parameters as numpy arrays 
		# with proper types and shapes
		x = np.asarray(x).astype(np.float32)
		t_policy = np.asarray(t_policy).squeeze()
		t_policy = (np.reshape(t_policy, (t_policy.shape[0]))).astype(np.float32)
		t_value = np.asarray(t_value)
		t_value = np.reshape(t_value, (t_value.shape[0],1)).astype(np.float32)
	
		# Clear the gradients of the model
		self.model.cleargrads()
		
		# Determine loss of game
		loss = self.model(x, t_policy, t_value)
		
		# Update network via backpropagation
		loss.backward()
		self.optimizer.update()

                # Return loss for printing
                return loss.data
	
        '''
                Loads a trained model in the player
        '''		
	def load_network(self):
		nr_files = len([name for name in os.listdir(self.path) if os.path.isfile(self.path+name)])
		save_name = self.path + self.filename + "_" + str(nr_files)
		serializers.load_npz(save_name, self.player.network)
	
	
	'''
                Saves a trained model from the player
        '''
	def save_network(self):
		nr_files = len([name for name in os.listdir(self.path) if os.path.isfile(self.path+name)])
		save_name = self.path + self.filename + "_" + str(nr_files+1)
		serializers.save_npz(save_name, self.model)


# Start the training process
trainer = Trainer(500, 400, "E:/KI/Master/2e_jaar/CCN/Project/networks/", "checkerzero")
trainer.train()
