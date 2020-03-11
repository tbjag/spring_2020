#import pandas as pd

#define class person with position 
class Person:
	def __init__(self, pos_x, pos_y):
		self.x = pos_x
		self.y = pos_y
		self.is_sick = False

p1 = Person(3,5)

print(p1.is_sick)


