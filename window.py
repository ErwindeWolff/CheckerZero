from __future__ import division
import sys, pygame, pygame.gfxdraw

from piece import Piece
from square import Square

'''
Class that defines the Checker GUI
'''

class GameWindow():

	def __init__(self, game):
	
		self.game = game
		
		
		screen_size = self.width, self.height = 1440, 900
		
		self.root = pygame.display.set_mode(screen_size)
		self.root.fill([222, 184, 135])
		self.root.fill([48, 48, 48])
		
		
		self.square_edge = int((min(self.width, self.height)-200) 
								/ max(self.game.rows, self.game.columns))	
		(self.cx, self.cy) = (int(self.width/2), int(self.height/2))
			
		self.set_squares()
		self.set_pieces()
		self.loop()	

	
	def set_squares(self):
	
		self.squares = list()
		for x in range(int(-self.game.columns/2), int(self.game.columns/2)):
			for y in range(int(-self.game.rows/2), int(self.game.rows/2)):
				if (x+y) % 2 == 0:
					color = [255 ,218 ,185]
				else:
					color = [160, 82, 45]
			
				area = pygame.Rect(self.cx + x*self.square_edge, self.cy + y*self.square_edge,
									self.square_edge, self.square_edge)
					
				a = x + int(self.game.columns/2) + 1
				b = self.game.rows - (y + int(self.game.columns/2))
									
				self.squares.append( Square(self.root, area, color, a, b))
	
	
	def set_pieces(self):
		# Draw white pieces on the game
		self.pieces = self.game.get_pieces()
		for piece in self.pieces:
		
			# Set root of piece
			piece.root = self.root
		
			# Coordinates where piece should draw itself
			(x, y) = (piece.x, piece.y)
			(a,b) = (x - int(self.game.rows/2) -1, y - int(self.game.columns/2))
			piece.area = pygame.Rect(self.cx + a*self.square_edge + 5, 
					self.cy - b*self.square_edge + 5, self.square_edge - 10, self.square_edge - 10)
			
			# Set color of piece
			if piece.value == 1:		
				piece.color = [255, 239, 219]
			else:
				piece.color = [41, 36, 33]

			
			
			
	
	def draw_board(self, moves):
	
		# Draw area surrounding board
		area = pygame.Rect( int((self.width-self.height)/2) + 50, 50, 
							self.height - 100, self.height-100)
		pygame.draw.rect(self.root, [139, 115, 85], area)
		pygame.draw.rect(self.root, [205, 155, 29], area, 3)
		
		
		for square in self.squares:
			square.draw(self.square_edge)
					
				
				
			
	def loop(self):
	
		change = True
		moves = []
		mouse_click = pygame.mouse.get_pressed()	
		
		while (True):
			
			########################################
			# EVENT CATCHING
			########################################
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					pygame.quit(); 
					sys.exit();
					
			#######################
			# Redrawing of screen #
			#######################
			
			if (change):
				self.root.fill([48, 48, 48])
				
				self.draw_board(moves)
					
				for piece in self.pieces:	
					piece.draw(self.square_edge)
				
				pygame.display.flip()
			
			##############################
			# Mouse activation of pieces #		
			##############################
			
			mouse_pos = pygame.mouse.get_pos()
			mouse_click = pygame.mouse.get_pressed()
			moves = []
			for piece in self.pieces:
			
				
			
				if (piece.value == self.game.current_player 
						and piece.pressed(mouse_pos, mouse_click, self.pieces)):
					
					moves = [(x2, y2) for (x1, y1, x2, y2) in self.game.get_moves() 
								if (x1 == piece.x and y1==piece.y)]
					
					for square in self.squares:
						if (square.x, square.y) in moves and piece.is_pressed:
							square.is_selected = True
						else:
							square.is_selected = False
						
					change = True
			
			
			for square in self.squares:
				if (square.is_selected and square.pressed(mouse_pos, mouse_click)):
					
					(x2, y2) = (square.x, square.y)
					(x1, y1) = (-1, -1)

					for piece in self.pieces:
						if piece.is_pressed:
							(x1, y1) = (piece.x, piece.y)

						piece.is_pressed = False
						piece.can_change = True
					
					for sq in self.squares:
						sq.is_selected = False
						
					self.game.move((x1, y1, x2, y2))
					
					if self.game.game_over():
						self.game.init_board()
					
					moves = []
					self.set_pieces()
					change = True
					
					for piece in self.pieces:
						if piece.x == x2 and piece.y == y2:
							piece.can_change = False
					
					break

			# Wait (60 fps)
			pygame.time.wait(16)
