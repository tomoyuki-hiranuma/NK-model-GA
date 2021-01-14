import numpy as np
import copy
from Individual import Individual
from NKModel import NKModel

class Population:
	def __init__(self, population_size, individual_size):
		self.individual_size = individual_size
		self.array = np.array([ Individual(individual_size) for i in range(population_size)])


	def print_array(self):
		for i in range(len(self.array)):
			print(self.array[i].gene)

	

if __name__ == '__main__':
	pop_size = 10
	ind_size = 5
	population = Population(pop_size, ind_size)
	population.print_array()
	print("crossover")
	family = population.crossover()
	for ind in family:
		print(ind.gene)