from __future__ import division
from game import *
from window import *
from player import *
from netplayer import * 
from time import time
import random
import copy

""" Main loop to let two players play checkers against each other """

# Set random seed for reproducibility
random.seed(42)

# Set game
game = Game(rows=8, columns=8)

# Set player 1. If network player: choose network to load (think of the path)
p1 = MCTSPlayer(nr_samples=1000)
#p1 = Netplayer(C=1, nr_samples=40, nondeterm_moves=15)
#p1.load_network("/home/endarmir/Downloads/Project/","checkerzero_1")

# Set player 2. If network player: choose network to load (think of the path)
#p2 = Netplayer(C=1, nr_samples=40, nondeterm_moves=15)
#p2.load_network("/home/endarmir/Downloads/Project/","checkerzero_20")
p2 = RandomPlayer()
players = [p1, p2]

# Play epochs amount of games
epochs = 10

#window = GameWindow(game)

# Keep score of moves and wins
played_moves = 0
possible_moves = 0
p1_wins = 0.0
p2_wins = 0.0

# Start timer and main loop
t = time()
for i in range(epochs):
	print(i+1)
	while not game.game_over():

		# Switch beginning player every game
		if(game.current_player == 1 and epochs%2 == 0):
			move = p1.get_move(game)
		elif(game.current_player == -1 and epochs%2 == 0):
			move = p2.get_move(game)
		elif(game.current_player == 1 and epochs%2 == 1):
			move = p2.get_move(game)
		else:
                        move = p1.get_move(game)
                        
		# Turn this on to print moves being played
		#print(move)

		# Update the game and players with the move information
		game.move(move)
		if p1 == p2:
			p1.update_state(move)
		else:
			p1.update_state(move)
			p2.update_state(move)
			
                # Increment played moves
		played_moves += 1

        # Reset players after finishing a game
	for player in players:
		player.reset()

        # Get score and increment the wins of the winning player
	score = game.get_score()
	print(score)
	if(score == 1):
		if(epochs%2 == 0):
			p1_wins += 1.0
			print("P1,1")
		else:
			p2_wins += 1.0
			print("P2,1")
	elif(score == -1):
		if(epochs%2 == 0):
			p2_wins += 1.0
			print("P2,-1")
		else:
			p1_wins += 1.0
			print("P1,-1")

	# Reinitialize board for a new game
	game.init_board()

# Check end time
t2 = time()

# Print statements
print("The game ended after an average of {0} moves".format((played_moves/2)/epochs))
print("The games took {0} seconds in total to play,\nfor an average of {1} seconds per game".format( (t2-t), (t2-t)/epochs))
print("The computer looked at {0} moves each second\n".format( (played_moves)/(t2-t)))
print("The {0} Player won {1}% of the time".format(p1.name, (p1_wins/epochs)*100))
print("The {0} Player won {1}% of the time".format(p2.name, (p2_wins/epochs)*100))
print("The game was a tie {0}% of the time".format(((epochs-p1_wins-p2_wins)/epochs)*100))
