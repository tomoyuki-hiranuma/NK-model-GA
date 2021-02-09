import numpy as np
from NKModel import NKModel
import copy

class Individual:
	def __init__(self, individual_size):
		self.gene = np.random.randint(2, size=individual_size)
		self.fitness = 0.0

	def mutation(self, mutation_rate):
		ind2 = copy.deepcopy(self.gene)
		for i in range(len(ind2)):
			if np.random.random() < mutation_rate:
				ind2[i] = 1 - ind2[i]
		self.gene = ind2


if __name__ == '__main__':
	N = 4
	K = 0
	individual = Individual(N)
	function = NKModel(N, K)
	print(individual.gene)
	individual.fitness = function.calc_eval(individual.gene)
	print(individual.fitness)
	individual.mutation(0.05)
	print(individual.gene)