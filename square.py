import pygame

'''
Class that defines the squares on the board of the Checker GUI
'''

class Square():

	def __init__(self, root, area, color, x, y):
	
		self.root = root
		self.area = area
		self.color = color
		
		self.x = x
		self.y = y
		
		self.is_selected = False
		
	
	def draw(self, square_edge):
	
		if self.is_selected:
			color = [50, 200, 30]
		else:
			color = self.color
			
		pygame.draw.rect(self.root, color, self.area)
		pygame.draw.rect(self.root, [205, 155, 29], self.area, 3)
		
	
	
	def pressed(self, mouse_pos, mouse_click):
		
		hovered = (mouse_pos[0] >= self.area.left and mouse_pos[0] <= self.area.right
					and mouse_pos[1] >= self.area.top and mouse_pos[1] <= self.area.bottom)
		
		return (hovered and mouse_click[0] == 1)

		
		if is_pressed:
			
			# Switch on!
			for p in pieces:
				if (self != p):
					p.is_pressed = False
					p.can_change = True
			
			self.is_pressed = not(self.is_pressed)
			self.can_change = False
				
			return self.is_pressed
			
	
		else:
			self.can_change = (mouse_click[0] == 0)
			return self.is_pressed
