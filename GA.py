# coding: utf8
import matplotlib.pyplot as plt
import random
import copy
import numpy as np

# パラメーター
N = 15 # 0/1リスト長（遺伝子長）
K = 0

ROUND = 8 # 総取り替えするであろう回数
PARENTS_SIZE = 2
POPULATION_SIZE = 75 # 集団の個体数
GENERATION = POPULATION_SIZE * ROUND // PARENTS_SIZE # 世代数
MUTATE_RATE = 0.1 # 突然異変の確率

def create_NK_landscape(N, K):
    np.random.seed(1)
    index = [ f'{i:0{K+1}b}' for i in range(2**(K+1)) ]
    rand_array = np.random.random(2**(K+1))
    return dict(zip(index, rand_array))

# 適応度を計算する
def calc_eval(gene):
    fitness = 0.0
    for i in range(len(gene)):
        index = str(gene[i])
        for j in range(i+1, i+K+1):
            index += str(gene[j%N])
        fitness += NK_landscape[index]
    fitness /= N
    return fitness
    # return np.sum(gene)

# 集団を適応度順にソートする
def sort_fitness(population):
    fp = np.array([calc_eval(x) for x in population])
    sorted_index = np.argsort(fp)[::-1] #降順のインデックス
    sorted_population = population[sorted_index] #降順に並び替え
    return sorted_population

# 1点交叉
def crossover(ind1, ind2):
    r1 = random.randint(0, N -1)
    # r2 = random.randint(r1 + 1, N)
    child1 = copy.deepcopy(ind1)
    child2 = copy.deepcopy(ind2)
    child1[0:r1] = ind2[0:r1]
    child2[0:r1] = ind1[0:r1]
    child1 = mutation(child1)
    child2 = mutation(child2)
    family = np.array([ind1, ind2, child1, child2])
    sorted_family = sort_fitness(family)
    elite_gene = sorted_family[0]
    random_index = np.random.randint(1, len(sorted_family))
    random_gene = sorted_family[random_index]
    return elite_gene, random_gene

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
    r1 = random.randint(0, len(population) -1)
    r2 = random.randint(0, len(population) -1)
    while r1 == r2:
        r2 = random.randint(0, len(population) -1)
    # 交叉
    child1, child2 = crossover(population[r1], population[r2])
    # 突然変異
    # child1 = mutation(child1)
    # child2 = mutation(child2)
    # 集団に追加
    population[r1] = child1
    population[r2] = child2
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
  
def get_mean_eval(population):
    sum_eval = 0.0
    for individual in population:
        sum_eval += calc_eval(individual)
    return sum_eval/len(population)

# メイン処理
NK_landscape = create_NK_landscape(N, K)
print(NK_landscape)
BEST_GENE, BEST_EVAL = get_optimization(N, K)
# 初期集団を生成（ランダムに0/1を10個ずつ並べる）
if __name__ == '__main__':
    population = init_population()
    print("0世代")
    print_population(population)
    generation_count = 0
    best_eval = 0.0

    generations = []
    elites_evals = []
    mean_evals = []
    while generation_count < GENERATION:
        print(str(generation_count + 1) + u"世代")
        population = do_one_generation(population)
        best_gene, best_eval = get_best_individual(population)
        print_population(population)
        print("best gene: {}\nbest evaluation: {}".format(best_gene, best_eval))
        generation_count += 1

        ### 出力用
        generations.append(generation_count)
        elites_evals.append(best_eval)
        mean_evals.append(get_mean_eval(population))
    
    print("opt gene: {}\nopt evaluation: {}".format(BEST_GENE, BEST_EVAL))
    ## グラフ
    plt.plot(generations, elites_evals)
    plt.plot(generations, mean_evals)
    plt.plot([0, GENERATION], [BEST_EVAL, BEST_EVAL])
    plt.show()
