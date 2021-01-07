# coding: utf8
import random
import copy

# パラメーター
LIST_SIZE = 10 # 0/1リスト長（遺伝子長）

POPULATION_SIZE = 10 # 集団の個体数
GENERATION = 50 # 世代数
MUTATE_RATE = 0.1 # 突然異変の確率
SELECT_RATE = 0.5 # 選択割合

# 適応度を計算する
def calc_fitness(individual):
    return sum(individual) # リスト要素の合計

# 集団を適応度順にソートする
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

# 選択（適応度の高い個体を残す）
def selection(population):
    sorted_population = sort_fitness(population)
    n = int(POPULATION_SIZE * SELECT_RATE)
    return sorted_population[0 : n]

# 交叉（ランダムな範囲をr1,r2に置き換える）
def crossover(ind1, ind2):
    r1 = random.randint(0, LIST_SIZE -1)
    # r2 = random.randint(r1 + 1, LIST_SIZE)
    child1 = copy.deepcopy(ind1)
    child2 = copy.deepcopy(ind2)
    child1[0:r1] = ind2[0:r1]
    child2[0:r1] = ind1[0:r1]
    return child1, child2

# 突然変異（10%の確率で遺伝子を変化）
def mutation(ind1):
    ind2 = copy.deepcopy(ind1)
    for i in range(LIST_SIZE):
        if random.random() < MUTATE_RATE:
            ind2[i] =  random.randint(0,1)
    return ind2

def init_population():
    population = []
    for i in range(POPULATION_SIZE):
        individual =  []
        for j in range(LIST_SIZE):
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

# メイン処理
# 初期集団を生成（ランダムに0/1を10個ずつ並べる）
if __name__ == '__main__':
    population = init_population()
    print_population(population)
    generation_count = 0
    while generation_count <= GENERATION:
        print(str(generation_count + 1) + u"世代")
        population = do_one_gengeration(population)
        print_population(population)
        generation_count += 1

        