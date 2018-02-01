""" Class that handles the nodes of the tree belonging to the MCTS player """

class Node():

	def __init__(self):
	
		self.current_player = None
	
		self.visits = 1
		self.value = 0
		self.children = []
		
	'''
                Returns whether a tree is fully expanded
                (When the amount of children is equal to amount of possible moves)
	'''		
	def is_fully_expanded(self, moves):
		return len(self.children) == len(moves)
		
	'''
                Add new node to tree
	'''		
	def expand(self, game, moves):
	
		# Select nth move (so order should always be the same)
		move = moves[len(self.children)]
		
		# Add new leaf to node
		new_state = Node()
		self.children.append( (move, new_state) )
		return (move, new_state)
		
		
