# coding: utf8
import matplotlib.pyplot as plt
import random
import copy
import numpy as np

# パラメーター

ROUND = 8 # 総取り替えするであろう回数
PARENTS_SIZE = 2
POPULATION_SIZE = 100 # 集団の個体数
GENERATION = POPULATION_SIZE * ROUND // PARENTS_SIZE # 世代数
MUTATE_RATE = 0.01 # 突然異変の確率

MAX_EVALUATION_NUMBER = 2*POPULATION_SIZE*10000

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
    # 突然変異
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
    # 交叉&突然変異
    child1, child2 = crossover(population[r1], population[r2])
    # 集団に追加
    population[r1] = child1
    population[r2] = child2
    return population

def print_population(population):
    for individual in population:
        print(individual)
        
def get_best_worst_evals(population):
    better_eval = 0.0
    worse_eval = 1.0
    for individual in population:
        fitness = calc_eval(individual)
        if better_eval <= fitness:
            better_eval = fitness
        if worse_eval >= fitness:
            worse_eval = fitness
    return better_eval, worse_eval

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
NK_landscape = ""
# 初期集団を生成（ランダムに0/1を10個ずつ並べる）
if __name__ == '__main__':
    N = 20 # 0/1リスト長（遺伝子長）
    Ks = np.arange(0, N-1, 3)
    for K in Ks:
    # create NK model
        NK_landscape = create_NK_landscape(N, K)
        print(NK_landscape)
        BEST_GENE, BEST_EVAL = get_optimization(N, K)

        # GA開始
        population = init_population()
        print("0世代")
        print_population(population)
        generation_count = 0
        eval_number = PARENTS_SIZE * generation_count
        best_eval = 0.0

        generations = []
        elites_evals = []
        mean_evals = []
        worst_evals = []
        mean_eval = 0.0
        while eval_number < MAX_EVALUATION_NUMBER and BEST_EVAL-mean_eval >= 0.001:
            print(str(generation_count + 1) + u"世代")
            population = do_one_generation(population)
            best_eval, worst_eval = get_best_worst_evals(population)
        
            generation_count += 1
            eval_number = POPULATION_SIZE * generation_count
            mean_eval = get_mean_eval(population)

            print("best evaluation: {}".format(best_eval))
            print("mean eval: {}".format(mean_eval))
            print("worst eval: {}".format(worst_eval))

            ### 出力用
            generations.append(eval_number)
            elites_evals.append(best_eval)
            mean_evals.append(mean_eval)
            worst_evals.append(worst_eval)
        
        print("opt gene: {}\nopt evaluation: {}".format(BEST_GENE, BEST_EVAL))
        ## グラフ
        plt.plot(generations, elites_evals, label="best eval in population")
        plt.plot(generations, mean_evals, label="mean eval in population")
        plt.plot([0, eval_number], [BEST_EVAL, BEST_EVAL], label="Opt Evaluation", color='red', linestyle='--')
        plt.plot(generations, worst_evals, label="bad eval in population")
        plt.legend()
        plt.xlabel("number of evaluation")
        plt.ylabel("evaluation value")
        plt.title("NK model N={} K={} POP_SIZE={}".format(N, K, POPULATION_SIZE))
        plt.savefig("./NK_Model_png/N={}_K={}_POP_SIZE={}.png".format(N, K, POPULATION_SIZE))
        plt.clf()
    # plt.show()
