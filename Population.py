import numpy as np
import copy
from Individual import Individual

class Population:
	def __init__(self, population_size, individual_size):
		self.individual_size = individual_size
		self.array = np.array([ Individual(individual_size) for i in range(population_size)])

	def crossover(self):
		point = np.random.randint(1, self.individual_size)
		parent1_index = np.random.randint(len(self.array))
		parent2_index = np.random.randint(len(self.array))
		while parent1_index == parent2_index:
			parent2_index = np.random.randint(len(self.array))

		parent1 = self.array[parent1_index]
		parent2 = self.array[parent2_index]
		child1 = Individual(self.individual_size)
		child2 = Individual(self.individual_size)
		child1.gene = parent1.gene[0:point] + parent2.gene[point:self.individual_size]
		child2.gene = parent2.gene[0:point] + parent1.gene[point:self.individual_size]
		# 突然変異
		child1.mutation(0.1)
		child2.mutation(0.1)
		return np.array([parent1.gene, parent2.gene, child1.gene, child2.gene])
		# family = np.array([parent1, parent2, child1, child2])
		# sorted_family = sort_fitness(family)
		# elite_gene = sorted_family[0]
		# random_index = np.random.randint(1, len(sorted_family))
		# random_gene = sorted_family[random_index]
		# return elite_gene, random_gene

	def do_one_generation(self):
		pass

	def print_array(self):
		for i in range(len(self.array)):
			print(self.array[i].gene)

	

if __name__ == '__main__':
	pop_size = 10
	ind_size = 5
	population = Population(pop_size, ind_size)
	population.print_array()
	print("crossover")
	print(population.crossover())
	# print(population.array[1].gene)
	# print(population.mutation(population.array[1].gene))