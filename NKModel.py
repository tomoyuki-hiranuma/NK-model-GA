import numpy as np

class NKModel:
	def __init__(self, N ,K):
		self.N = N
		self.K = K
		self.nk_landscape = self._create_NK_landscape()
		self.best_eval = 0.0
		self.best_gene = ""

	def _create_NK_landscape(self):
		# np.random.seed(1)
		index = [ f'{i:0{self.K+1}b}' for i in range(2**(self.K+1)) ]
		rand_array = np.random.random(2**(self.K+1))
		return dict(zip(index, rand_array))

	# 適応度を計算する
	def calc_eval(self, gene):
		fitness = 0.0
		long_genes = np.hstack((gene, gene))
		for i in range(len(gene)):
			element = ""
			for j in long_genes[i:i+self.K+1]:
				element += str(j)
			fitness += self.nk_landscape[element]
		fitness /= len(gene)
		return fitness

	# 最適解計算
	def calc_optimization(self):
		best_gene = ""
		best_eval = 0.0
		all_genes = np.array([ f'{i:0{self.N}b}' for i in range(2**(self.N)) ])
		all_genes = self._to_np_int(all_genes)
		for gene in all_genes:
			fitness = self.calc_eval(gene)
			if best_eval <= fitness:
				best_eval = fitness
				best_gene = gene
		self.best_eval = best_eval
		self.best_gene = best_gene

	def _to_np_int(self, binary_array):
		new_genes = []
		for gene in binary_array:
			gene = list(gene)
			genes = []
			for num in gene:
				genes.append(int(num))
			new_genes.append(np.array(genes))
		return np.array(new_genes)

	@property
	def get_best_eval(self):
		return self.best_eval

	@property
	def get_best_gene(self):
		return self.best_gene

	def get_optimized_solution(self):
		return self.best_gene, self.best_eval


if __name__ == '__main__':
	N = 4
	K = 0
	NK_model = NKModel(N, K)
	print(NK_model.nk_landscape)
	NK_model.calc_optimization()
	print(NK_model.get_optimized_solution())