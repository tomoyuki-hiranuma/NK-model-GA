# coding: utf8
import numpy as np
import random

'''
1: Kによるテーブル作成（0,1）一様乱数
2: Nとテーブルから最適解を探す
'''

'''
 使い方
    NK_landscape[f'{1:0{K+1}b}']
  これでテーブルの中身が呼び出せる
'''
def create_NK_landscape(N, K):
    np.random.seed(0)
    index = [ f'{i:0{K+1}b}' for i in range(2**(K+1)) ]
    rand_array = np.random.random(2**(K+1))
    return dict(zip(index, rand_array))

def calc_eval(gene, NK_landscape):
    fitness = 0.0
    for i in range(len(gene)):
        if i+K+1 < N:
            fitness += NK_landscape[gene[i:(i+K+1)]]
        else:
            fitness += NK_landscape[gene[i:(i+K+1)] + gene[0:(i+K+1)%N]]
    fitness /= N
    return fitness


def get_best_gene(N, K, NK_landscape):
    best_gene = ""
    best_eval = 0.0
    all_genes = np.array([ f'{i:0{N}b}' for i in range(2**(N)) ])
    # print(all_genes)
    for gene in all_genes:
        fitness = calc_eval(gene, NK_landscape)
        if best_eval <= fitness:
            best_eval = fitness
            best_gene = gene
  
    return best_gene, best_eval

def init_population(POPULATION_SIZE, N):
    return np.array([f'{np.random.randint(2**N):0{N}b}' for i in range(POPULATION_SIZE)])
    # return np.random.randint(2, size=(POPULATION_SIZE, N))

if __name__ == '__main__':
    N = 3
    K = 0
    POPULATION_SIZE = 10
    # NK_landscape = create_NK_landscape(N, K)
    # print(NK_landscape)
    # best_gene, best_eval = get_best_gene(N, K, NK_landscape)
    # print("====BEST GENE INFO====")
    # print("Best gene:", best_gene)
    # print("Best evaluation:", best_eval)
    population = init_population(POPULATION_SIZE, N)
    print(population)

  