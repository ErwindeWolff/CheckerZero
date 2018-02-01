""" Class that handles the nodes of the tree belonging to the MCTS search in the CheckerZero network """

class Node():

	def __init__(self):
	
		self.current_player = None
		
		self.visits = 0
		self.total_value = 0
		self.mean_value = 0
		
		self.move = None
		self.prob = None
	
		self.branches = list()
		

	'''
                Returns whether a node is a leaf node
	'''
	def is_leaf(self):
		return len(self.branches) == 0


	'''
                Returns the amount of nodes that belong in the branches of this node
	'''
	def get_size(self):
		return 1 + sum([branch.get_size() for branch in self.branches])
