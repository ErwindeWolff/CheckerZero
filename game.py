from __future__ import division
from piece import *

""" Class that determines the rules and structure of the Checkers game """

class Game():

	def __init__(self, rows=8, columns=8, set_board=True):

                # Set structure of the board
		self.rows = rows
		self.columns = columns

                # Reset board to begin position
		if set_board:
			self.init_board()
		
	'''
                Initializes the board with the begin position
        '''	
	def init_board(self):
	
		s = ""
		for _ in range(self.rows):
			for _ in range(self.columns):
				s += "0"
		self.board = s

		self.pieces_one = list()
		self.pieces_two = list()
		for x in range(1, self.rows, 2):
			for j in range(1, 4):
			
				if (j == 2):
					i = x+1
				else:
					i = x
				
				self.pieces_one.append( Piece(i, j, 1) )
				self.set(i, j, '1')
			
				self.pieces_two.append( Piece(self.columns - (i-1), self.rows - (j-1), -1) )
				self.set(self.columns - (i-1), self.rows - (j-1) , '2')
		
		self.current_player = 1
		self.moves_since_hit = 0
		self.path = dict()


	'''
                Sets the position x,y on the board with value
        '''
	def set(self, x, y, value):
		cut = (self.rows - (y))*self.columns + (x-1)
		self.board = self.board[0:cut] + value + self.board[cut+1::]


	'''
                Gets on position x,y the value on the board
        '''
	def get(self, x, y):
		if (x-1 >= 0 and x-1 < self.columns and (self.rows - y) >= 0 
			and (self.rows - y) < self.rows):
			
			index = (self.rows - (y))*self.columns + (x-1)
			return self.board[index]
		return ''
	
	
	'''
                Returns a list of every piece on the board
        '''	
	def get_pieces(self):
		return self.pieces_one + self.pieces_two

	
	'''
                Moves a piece from its initial position to the current position on the board
        '''	
	def move(self, move):
	
		# Unpack move
		(x1, y1, x2, y2) = move
			
		# Determine allies and enemy pieces
		if self.current_player == 1:
			own_pieces = self.pieces_one
			enemy_pieces = self.pieces_two
		else:
			own_pieces = self.pieces_two
			enemy_pieces = self.pieces_one
			
			
		# Find what piece moved
		piece = own_pieces[0]
		for p in own_pieces:
				if (p.x == x1) and (p.y == y1):
					piece = p
					break
					

		# Increment moves since hit (checks for draw)					
		self.moves_since_hit += 1
		
		
		# If forced, determine which piece is killed
		if self.forced_move:
			self.moves_since_hit = 0
		
			# Coordinates of to kill
			(a, b) = (x1 + int((x2 - x1)/2), y1 + int((y2-y1)/2))
			self.set(a, b, '0')

			# Remove that piece
			to_remove = enemy_pieces[0]
			for p in enemy_pieces:
				if (p.x == a) and (p.y == b):
					to_remove = p
					break
			enemy_pieces.remove(to_remove)


		# Perform actual move
		self.set(x1, y1, '0')
		if (self.current_player == 1):
			self.set(x2, y2, '1')
		else:
			self.set(x2, y2, '2')


		# Remember that this state was seen (another time)
		self.path[self.board] = self.path.get(self.board, 0) + 1
		
		# Update coordinates of piece
		piece.x = x2
		piece.y = y2
		
		# Crown the stone if reached the opposite side
		if ( (self.current_player == 1 and y2 == self.rows) or
				(self.current_player == -1 and y2 == 1)):
			piece.is_king = True

		# If move was forced find if still forced moves left
		if self.forced_move:
			(_, forced_moves) = self.get_piece_moves(piece)
		
			# Only switch turn if current player cannot keep on hitting
			if len(forced_moves) <= 0:
				self.current_player = self.current_player*-1
		else:
			self.current_player = self.current_player*-1

	
	'''
                Returns a list of all possible moves in the current game state
        '''	
	def get_moves(self):
	
		free_moves = list()
		forced_moves = list()
		
		if self.current_player == 1:
			pieces = self.pieces_one
		else:
			pieces = self.pieces_two
			
			
		for piece in pieces:			
			(free, forced) = self.get_piece_moves(piece)
			free_moves += free
			forced_moves += forced
			
		if len(forced_moves) > 0:
			self.forced_move = True
			return forced_moves
		else:
			self.forced_move = False
			return free_moves
	
	
	'''
                Returns all possible moves for a specific piece in the current game state
        '''	
	def get_piece_moves(self, piece):

		(x, y) = (piece.x, piece.y)
			
		if piece.is_king:
			moves = [(1, 1), (-1, 1), (1, -1), (-1, -1)] 
		else:
			moves = [(1, piece.value), (-1, piece.value)]
		
		free_moves = []		# Moves if there are no forced moves
		forced_moves = []	# Will make sure piece will capture for sure
		
		for (a, b) in moves:	
			if self.get(x+a, y+b) == '0':
				free_moves.append( (x, y, x+a, y+b) )
			elif (piece.value == 1 
					and (self.get(x + a, y + b) == '2' 
					and self.get(x+2*a, y+2*b) == '0')):
				forced_moves.append( (x, y, x+2*a, y+2*b))
			
			elif (piece.value == -1
					and (self.get(x + a, y + b) == '1' 
					and self.get(x+2*a, y+2*b) == '0')):
				forced_moves.append( (x, y, x+2*a, y+2*b))
		
		return (free_moves, forced_moves)




	'''
                Returns a boolean whether this game state is the end of the game
        '''
	def game_over(self):
		return (len(self.pieces_one) == 0 
				or len(self.pieces_two) == 0 
				or len(self.get_moves()) == 0
				or self.moves_since_hit >= 40
				or self.path.get(self.board, 0) >= 3)
				
				
	'''
                Returns the winner of the ended game state
        '''
	def get_score(self):
		# If no move has been killed for 40 moves
		# or a 3-repeat of the same position
		if self.moves_since_hit >= 40 or self.path.get(self.board, 0) >= 3:
			return 0
			
		# If the current player has no legal moves, 
		# that means that the other player won
		elif len(self.get_moves()) == 0:
			return self.current_player * -1
			
		elif (self.current_player == 1) and (len(self.pieces_one) == 0):
			return -1
			
		else:
			return 1
	
