import pandas as pd
import random
import time
import math

#define move set
MOVES = [0,1,2]
#define area
BOUND_X_MIN = 0
BOUND_Y_MIN = 0
BOUND_X_MAX = 100
BOUND_Y_MAX = 100

def print_state(person):
	print("%d is at (%d,%d) and is %ssick" %( person.id, person.x, person.y, "" if person.is_sick else "not "))

#define class person with position 
class Person:
	def __init__(self, id, pos_x, pos_y):
		self.id = id
		self.x = pos_x
		self.y = pos_y
		self.is_sick = False
		self.goto_x = random.randrange(BOUND_X_MAX)
		self.goto_y = random.randrange(BOUND_Y_MAX)
	def move(self):
		#random move set, if at edge, then do nothing
		if(self.x == self.goto_x and self.y == self.goto_y):
			#set new random coords if arrived at destination
			self.goto_x = random.randrange(BOUND_X_MAX)
			self.goto_y = random.randrange(BOUND_Y_MAX)
		elif(self.x == self.goto_x):
			self.y += 1
		elif(self.y == self.goto_y):
			self.x += 1
		else:
			move = random.choice(MOVES)
			if(move == 0):
				if ((self.goto_x - self.x) > 0):
					self.x += 1
				else:
					self.x -= 1
			elif(move == 1):
				if ((self.goto_y - self.y) > 0):
					self.y += 1
				else:
					self.y -= 1				

		

#calculate manhattan distance abs(posx - posx)
def within_area(person1, person2, prox):
	if(abs(person1.x-person2.x) <= prox and abs(person1.y-person2.y) <= prox):
		return True
	else:
		return False

def main():
	arr = []
	for i in range(30):
		arr.append(Person(i, 0, i*2))
		print_state(arr[i])
		#create a bunch of classes
	#make one person sicl
	arr[0].is_sick = True
	
	for time_step in range(100):
		for i in range(30):
			arr[i].move()
		#can optimize this part
		for lol in range(30):
			for gey in range(30):
				if(lol != gey):
					if(within_area(arr[lol], arr[gey], 1)):
						if(arr[lol].is_sick or arr[gey].is_sick):
							arr[lol].is_sick = True
							arr[gey].is_sick = True
		#print out states
		for i in range(30):
			print_state(arr[i])


main()