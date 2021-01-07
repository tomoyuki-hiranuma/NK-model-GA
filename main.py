# coding: utf8
import matplotlib.pyplot as plt
import random
import copy
import numpy as np

### Todo: numpyに書き換える

# パラメーター
N = 10 # 0/1リスト長（遺伝子長）
K = 0

POPULATION_SIZE = 10 # 集団の個体数
GENERATION = 50 # 世代数
MUTATE_RATE = 0.1 # 突然異変の確率
SELECT_RATE = 0.5 # 選択割合

# 適応度を計算する
def calc_fitness(individual):
    return sum(individual) # リスト要素の合計

# 集団を適応度順にソートする
"""
 Todo: 評価値としてNKモデルでの評価値を採用
"""
def sort_fitness(population):
    fp = []
    for individual in population:
        fitness = calc_fitness(individual)
        fp.append((fitness, individual))
    fp.sort(reverse=True)  # 適応度でソート（降順）

    sorted_population = []
    # リストに入れ直す
    for fitness,  individual in fp:
        sorted_population.append(individual)
    return sorted_population

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
    population = []
    for i in range(POPULATION_SIZE):
        individual =  []
        for j in range(N):
            individual.append(random.randint(0,1))
        population.append(individual)
    return population

def do_one_gengeration(population):
    #選択
    population = selection(population)

    # 少なくなった分の個体を交叉と突然変異によって生成
    n = POPULATION_SIZE - len(population)
    for i in range(n//2):
        r1 = random.randint(0, len(population) -1)
        r2 = random.randint(0, len(population) -1)
        while r1 == r2:
            r2 = random.randint(0, len(population) -1)
        # 交叉
        child1, child2 = crossover(population[r1], population[r2])
        # 突然変異
        child1 = mutation(child1)
        child2 = mutation(child2)
        # 集団に追加
        population.append(child1)
        population.append(child2)
    return population

def print_population(population):
    for individual in population:
        print(individual)
        
def get_best_individual(population):
    better_eval = 0.0
    better_gene = []
    for individual in population:
        fitness = calc_fitness(individual)
        if better_eval <= fitness:
          better_eval = fitness
          better_gene = individual
    return better_gene, better_eval

# メイン処理
best_gene = [0 for i in range(N)]
best_eval = 0.0
# 初期集団を生成（ランダムに0/1を10個ずつ並べる）
if __name__ == '__main__':
    population = init_population()
    generation_count = 0

    x = []
    y = []

    while generation_count < GENERATION:
        # print(str(generation_count + 1) + u"世代")
        population = do_one_gengeration(population)
        print_population(population)
        best_gene, best_eval = get_best_individual(population)
        print("best gene: {}\nbest evaluation: {}".format(best_gene, best_eval))
        generation_count += 1

        ### 出力用
        x.append(generation_count)
        y.append(best_eval)
    
    ## グラフ
    plt.plot(x, y)
    plt.show()
