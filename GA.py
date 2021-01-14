from Population import Population
from NKModel import NKModel
import numpy as np

class GeneticAlgorithm:
	def __init__(self, N, K, population_size):
		self.population = Population(population_size, N)
		self.nk_model = NKModel(N, K)

	def do_one_generation(self):
		parent1_index = np.random.randint(len(self.population.array))
		parent2_index = np.random.randint(len(self.population.array))
		while parent1_index == parent2_index:
			parent2_index = np.random.randint(len(self.population.array))
		
		parent1 = self.population.array[parent1_index]
		parent2 = self.population.array[parent2_index]
        # 交叉
		child1, child2 = self.crossover(parent1, parent2)
		# 突然変異
		child1.mutation(0.1)
		child2.mutation(0.1)
		family = np.array([parent1, parent2, child1, child2])
		selected_individual1, selected_individual2 = select_individuals(family)

	def crossover(self, parent1, parent2):
		point = np.random.randint(1, self.individual_size)
		child1 = Individual(self.individual_size)
		child2 = Individual(self.individual_size)
		child1.gene = parent1.gene[0:point] + parent2.gene[point:self.individual_size]
		child2.gene = parent2.gene[0:point] + parent1.gene[point:self.individual_size]
		return np.array[child1, child2]
		

	def sort_fitness(self, population):
		population.print_array()
		for individual in population.array:
			individual.fitness = self.nk_model.calc_eval(individual.gene)
		sorted_population = Population(len(population.array), len(population.array[0].gene))
		sorted_population.array = sorted(population.array, key=lambda x: x.fitness)[::-1] #降順のインデックス
		print("=====sorted====")
		sorted_population.print_array()
		return sorted_population

	def select_individuals(self, family):
		sorted_family = sort_fitness(family)
		elite_gene = sorted_family[0]
		random_index = np.random.randint(1, len(sorted_family))
		random_gene = sorted_family[random_index]
		return elite_gene, random_gene


if __name__ == '__main__':
	N = 5
	K = 0
	Population_size = 10

	ga = GeneticAlgorithm(N, K, Population_size)
	print(ga.sort_fitness(ga.population))