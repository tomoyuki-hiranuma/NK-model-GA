import numpy as np
from NKModel import NKModel

class Individual:
	def __init__(self, individual_size):
		self.gene = f'{np.random.randint(2**individual_size):0{individual_size}b}'
		self.fitness = 0.0


if __name__ == '__main__':
	N = 4
	K = 0
	individual = Individual(N)
	function = NKModel(N, K)
	print(individual.gene)
	individual.fitness = function.calc_eval(individual.gene)
	print(individual.fitness)