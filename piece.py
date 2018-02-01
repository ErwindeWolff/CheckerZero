from __future__ import division
import pygame, pygame.gfxdraw

"""Class that defines a board piece in the class Game and in the Checker GUI """

class Piece():

	def __init__(self, x, y, value):
		
		# Variables used by the game
		self.x = x
		self.y = y
		self.value = value
		self.is_king = False
		
		# Variables ONLY set or used by the UI window
		self.root = None
		self.area = None
		self.color = None
		self.is_pressed = False
		self.can_change = True
		
		
	'''
                Draws the piece in the given square on the board in the GUI
	'''		
	def draw(self, square_edge):

		# Draw piece itself
		pygame.draw.ellipse(self.root, self.color, self.area)
		for dr in range(-6, -3):
			r = int(square_edge/2) + dr
			pygame.gfxdraw.aacircle(self.root, self.area.centerx, 
									self.area.centery, r, [205, 155, 29])
		
		# If king, draw another piece on top, slightly askew							
		if self.is_king:
			pygame.gfxdraw.filled_circle(self.root, self.area.centerx + 7,
									self.area.centery - 7, r, self.color)
			for dr in range(-6, -3):
				r = int(square_edge/2) + dr
				pygame.gfxdraw.aacircle(self.root, self.area.centerx + 7, 
									self.area.centery - 7, r, [205, 155, 29])
	
	
	'''
               Returns whether in the GUI this piece is presed by the user
	'''	
	def pressed(self, mouse_pos, mouse_click, pieces):
		
		hovered = (mouse_pos[0] >= self.area.left and mouse_pos[0] <= self.area.right
					and mouse_pos[1] >= self.area.top and mouse_pos[1] <= self.area.bottom)
		is_pressed = (hovered and mouse_click[0] == 1)

		
		if is_pressed and self.can_change:		
			# Switch on!
			for p in pieces:
				if (self != p):
					p.is_pressed = False
					p.can_change = True
			
			self.is_pressed = not(self.is_pressed)
			self.can_change = False
				
			return True
			
	
		else:
			self.can_change = (mouse_click[0] == 0)
			return False
