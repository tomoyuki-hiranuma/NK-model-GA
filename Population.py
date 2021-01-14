import numpy as np
from Individual import Individual

class Population:
	def __init__(self, population_size, individual_size):
		self.array = np.array([ Individual(individual_size) for i in range(population_size)])

	def crossover(self, parent1, parent2):
		pass

	def mutation(self, ind):
		pass

	def do_one_generation(self):
		

	def print_array(self):
		for i in range(len(self.array)):
			print(self.array[i].gene)

	

if __name__ == '__main__':
	pop_size = 10
	ind_size = 4
	population = Population(pop_size, ind_size)
	population.print_array()