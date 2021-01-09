# coding: utf8
import matplotlib.pyplot as plt
import random
import copy
import numpy as np

# パラメーター
N = 10 # 0/1リスト長（遺伝子長）
K = 0

POPULATION_SIZE = 10 # 集団の個体数
GENERATION = 50 # 世代数
MUTATE_RATE = 0.1 # 突然異変の確率
SELECT_RATE = 0.9 # 選択割合

def create_NK_landscape(N, K):
    np.random.seed(1)
    index = [ f'{i:0{K+1}b}' for i in range(2**(K+1)) ]
    rand_array = np.random.random(2**(K+1))
    return dict(zip(index, rand_array))

# 適応度を計算する
def calc_eval(gene):
    return np.sum(gene)

# 集団を適応度順にソートする
def sort_fitness(population):
    fp = np.array([calc_eval(x) for x in population])
    sorted_index = np.argsort(fp) #昇順のインデックス
    sorted_population = population[sorted_index] #昇順に並び替え
    return sorted_population[::-1]

# エリート選択（適応度の高い個体を残す）
def selection(population):
    sorted_population = sort_fitness(population)
    n = int(POPULATION_SIZE * SELECT_RATE)

    return sorted_population[0 : n]

# 1点交叉
def crossover(ind1, ind2):
    r1 = random.randint(0, N -1)
    # r2 = random.randint(r1 + 1, N)
    child1 = copy.deepcopy(ind1)
    child2 = copy.deepcopy(ind2)
    child1[0:r1] = ind2[0:r1]
    child2[0:r1] = ind1[0:r1]
    return child1, child2

# 突然変異（10%の確率で遺伝子を変化）
def mutation(ind1):
    ind2 = copy.deepcopy(ind1)
    for i in range(N):
        if random.random() < MUTATE_RATE:
            ind2[i] =  random.randint(0,1)
    return ind2

def init_population():
    return np.random.randint(2, size=(POPULATION_SIZE, N))

def do_one_generation(population):
    #選択
    population = selection(population)

    # 少なくなった分の個体を交叉と突然変異によって生成
    """
    todo: 総入れ替え×　→ 多様性維持の世代交代モデル 
    """
    n = POPULATION_SIZE - len(population)
    for i in range(n):
        r1 = random.randint(0, len(population) -1)
        r2 = random.randint(0, len(population) -1)
        while r1 == r2:
            r2 = random.randint(0, len(population) -1)
        # 交叉
        child1, child2 = crossover(population[r1], population[r2])
        # 突然変異
        child1 = mutation(child1)
        # child2 = mutation(child2)
        # 集団に追加
        population = np.vstack((population, child1))
    return population

def print_population(population):
    for individual in population:
        print(individual)
        
def get_best_individual(population):
    better_eval = 0.0
    better_gene = []
    for individual in population:
        fitness = calc_eval(individual)
        if better_eval <= fitness:
            better_eval = fitness
            better_gene = individual
    return better_gene, better_eval

def get_optimization(N, K):
    best_gene = ""
    best_eval = 0.0
    all_genes = np.array([ f'{i:0{N}b}' for i in range(2**(N)) ])
    for gene in all_genes:
        fitness = calc_eval(gene)
        if best_eval <= fitness:
            best_eval = fitness
            best_gene = gene
    
    return best_gene, best_eval

# メイン処理
NK_landscape = create_NK_landscape(N, K)
# BEST_GENE, BEST_EVAL = get_optimization(N, K)
BEST_GENE = np.array([1 for i in range(N)])
BEST_EVAL = N
# 初期集団を生成（ランダムに0/1を10個ずつ並べる）
if __name__ == '__main__':
    population = init_population()
    print("0世代")
    print_population(population)
    generation_count = 0
    best_eval = 0.0

    x = []
    y = []
    while generation_count < GENERATION:
        print(str(generation_count + 1) + u"世代")
        population = do_one_generation(population)
        best_gene, best_eval = get_best_individual(population)
        print_population(population)
        print("best gene: {}\nbest evaluation: {}".format(best_gene, best_eval))
        generation_count += 1

        ### 出力用
        x.append(generation_count)
        y.append(best_eval)
    
    ## グラフ
    plt.plot(x, y)
    plt.show()
