import random
import numpy as np

def copy_list(old_arr: [int]):
    new_arr = []
    for element in old_arr:
        new_arr.append(element)
    return new_arr


class Individual:
    def __init__(self, params, distance_matrix, sequence_encodding=None):
        self.distance_matrix = distance_matrix
        self.params = params
        if sequence_encodding is None:
            sequence_encodding = [i for i in range(self.params['sequence_len'])]
            random.shuffle(sequence_encodding)
        self.sequence_encodding = sequence_encodding
        fitness = 0.0
        for i in range(self.params['sequence_len'] - 1):
            from_idx = self.sequence_encodding[i]
            to_idx = self.sequence_encodding[i + 1]
            fitness += self.distance_matrix[from_idx, to_idx]
        fitness += self.distance_matrix[self.sequence_encodding[-1], self.sequence_encodding[0]]
        self.fitness = fitness


class SGA:
    def __init__(self, params, input_):
        self.params = params
        self.distance_matrix = input_
        self.best = None
        self.ind_list = []
        self.result_list = []
        self.fitness_list = []

        selection_probability = np.random.dirichlet(np.ones(2*self.params['ind_num']), size=1)[0]
        self.selection_probability = sorted(selection_probability, reverse=True)


    def cross(self):
        new_gen = []
        random.shuffle(self.ind_list)
        for i in range(0, self.params['ind_num'] - 1, 2):
            sequence1 = copy_list(self.ind_list[i].sequence_encodding)
            sequence2 = copy_list(self.ind_list[i + 1].sequence_encodding)
            index1 = random.randint(0, self.params['sequence_len'] - 2)
            index2 = random.randint(index1, self.params['sequence_len'] - 1)
            pos1_recorder = {value: idx for idx, value in enumerate(sequence1)}
            pos2_recorder = {value: idx for idx, value in enumerate(sequence2)}
            for j in range(index1, index2):
                value1, value2 = sequence1[j], sequence2[j]
                pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
                sequence1[j], sequence1[pos1] = sequence1[pos1], sequence1[j]
                sequence2[j], sequence2[pos2] = sequence2[pos2], sequence2[j]
                pos1_recorder[value1], pos1_recorder[value2] = pos1, j
                pos2_recorder[value1], pos2_recorder[value2] = j, pos2
            new_gen.append(Individual(self.params, self.distance_matrix, sequence1))
            new_gen.append(Individual(self.params, self.distance_matrix, sequence2))
        return new_gen

    def mutate(self, new_gen):
        for ind in new_gen:
            if random.random() < self.params['mutate_prob']:
                old_genes = copy_list(ind.sequence_encodding)
                index1 = random.randint(0, self.params['sequence_len'] - 2)
                index2 = random.randint(index1, self.params['sequence_len'] - 1)
                genes_mutate = old_genes[index1:index2]
                genes_mutate.reverse()
                ind.sequence_encodding = old_genes[:index1] + genes_mutate + old_genes[index2:]
        self.ind_list += new_gen

    def select(self):
        # 选择排序法
        # fitness_list = [ind.fitness for ind in self.ind_list]
        # rank = np.argsort(fitness_list)
        # selection_probability = np.array(self.selection_probability)[rank]

        # selected_winners = []
        # i = 0
        # while i < self.params['ind_num']:
        #     selected = np.random.choice(self.ind_list, p=selection_probability)
        #     if selected not in selected_winners:
        #         selected_winners.append(selected)
        #         i+=1
        #     else:
        #         continue
        
        # self.ind_list = selected_winners

        # 锦标赛
        choose_num = 10
        group_size = 10
        group_winner = self.params['ind_num'] // choose_num
        winners = []
        for i in range(choose_num):
            group = []
            for j in range(group_size):
                player = random.choice(self.ind_list)
                player = Individual(self.params, self.distance_matrix, player.sequence_encodding)
                group.append(player)
            group = SGA.rank(group)
            winners += group[:group_winner]
        self.ind_list = winners

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
        for ind in self.ind_list:
            if ind.fitness < self.best.fitness:
                self.best = ind

    def train(self):
        self.ind_list = [Individual(self.params, self.distance_matrix) for _ in range(self.params['ind_num'])]
        self.best = self.ind_list[0]
        for i in range(self.params['gen_num']):
            self.next_gen()
            result = copy_list(self.best.sequence_encodding)
            result.append(result[0])
            self.result_list.append(result)
            self.fitness_list.append(self.best.fitness)
        return self.result_list, self.fitness_list
