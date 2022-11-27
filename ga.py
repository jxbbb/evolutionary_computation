import config as conf
import random
import numpy as np

def copy_list(old_arr: [int]):
    new_arr = []
    for element in old_arr:
        new_arr.append(element)
    return new_arr


class Individual:
    def __init__(self, params, city_dist_mat, genes=None):
        self.city_dist_mat = city_dist_mat
        self.params = params
        if genes is None:
            genes = [i for i in range(self.params['gene_len'])]
            random.shuffle(genes)
        self.genes = genes
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        fitness = 0.0
        for i in range(self.params['gene_len'] - 1):
            from_idx = self.genes[i]
            to_idx = self.genes[i + 1]
            fitness += self.city_dist_mat[from_idx, to_idx]
        fitness += self.city_dist_mat[self.genes[-1], self.genes[0]]
        return fitness


class Ga:
    def __init__(self, params, input_):
        self.params = params
        self.city_dist_mat = input_
        self.best = None
        self.individual_list = []
        self.result_list = []
        self.fitness_list = []

        selection_probability = np.random.dirichlet(np.ones(2*self.params['individual_num']), size=1)[0]
        self.selection_probability = sorted(selection_probability, reverse=True)


    def cross(self):
        new_gen = []
        random.shuffle(self.individual_list)
        for i in range(0, self.params['individual_num'] - 1, 2):
            genes1 = copy_list(self.individual_list[i].genes)
            genes2 = copy_list(self.individual_list[i + 1].genes)
            index1 = random.randint(0, self.params['gene_len'] - 2)
            index2 = random.randint(index1, self.params['gene_len'] - 1)
            pos1_recorder = {value: idx for idx, value in enumerate(genes1)}
            pos2_recorder = {value: idx for idx, value in enumerate(genes2)}
            for j in range(index1, index2):
                value1, value2 = genes1[j], genes2[j]
                pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
                genes1[j], genes1[pos1] = genes1[pos1], genes1[j]
                genes2[j], genes2[pos2] = genes2[pos2], genes2[j]
                pos1_recorder[value1], pos1_recorder[value2] = pos1, j
                pos2_recorder[value1], pos2_recorder[value2] = j, pos2
            new_gen.append(Individual(self.params, self.city_dist_mat, genes1))
            new_gen.append(Individual(self.params, self.city_dist_mat, genes2))
        return new_gen

    def mutate(self, new_gen):
        for individual in new_gen:
            if random.random() < self.params['mutate_prob']:
                old_genes = copy_list(individual.genes)
                index1 = random.randint(0, self.params['gene_len'] - 2)
                index2 = random.randint(index1, self.params['gene_len'] - 1)
                genes_mutate = old_genes[index1:index2]
                genes_mutate.reverse()
                individual.genes = old_genes[:index1] + genes_mutate + old_genes[index2:]
        self.individual_list += new_gen

    def select(self):
        # 选择排序法
        # fitness_list = [individual.fitness for individual in self.individual_list]
        # rank = np.argsort(fitness_list)
        # selection_probability = np.array(self.selection_probability)[rank]

        # selected_winners = []
        # i = 0
        # while i < self.params['individual_num']:
        #     selected = np.random.choice(self.individual_list, p=selection_probability)
        #     if selected not in selected_winners:
        #         selected_winners.append(selected)
        #         i+=1
        #     else:
        #         continue
        
        # self.individual_list = selected_winners

        # 锦标赛
        group_num = 10
        group_size = 10
        group_winner = self.params['individual_num'] // group_num
        winners = []
        for i in range(group_num):
            group = []
            for j in range(group_size):
                player = random.choice(self.individual_list)
                player = Individual(self.params, self.city_dist_mat, player.genes)
                group.append(player)
            group = Ga.rank(group)
            winners += group[:group_winner]
        self.individual_list = winners

    @staticmethod
    def rank(group):
        for i in range(1, len(group)):
            for j in range(0, len(group) - i):
                if group[j].fitness > group[j + 1].fitness:
                    group[j], group[j + 1] = group[j + 1], group[j]
        return group

    def next_gen(self):
        new_gen = self.cross()
        self.mutate(new_gen)
        self.select()
        for individual in self.individual_list:
            if individual.fitness < self.best.fitness:
                self.best = individual

    def train(self):
        self.individual_list = [Individual(self.params, self.city_dist_mat) for _ in range(self.params['individual_num'])]
        self.best = self.individual_list[0]
        for i in range(self.params['gen_num']):
            self.next_gen()
            result = copy_list(self.best.genes)
            result.append(result[0])
            self.result_list.append(result)
            self.fitness_list.append(self.best.fitness)
        return self.result_list, self.fitness_list
