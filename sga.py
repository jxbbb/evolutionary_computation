import random
import numpy as np
from copy import copy


class SGA:
    def __init__(self, params, distance_matrix):
        self.params = params
        self.distance_matrix = distance_matrix
        self.best = None
        self.ind_list = []
        self.result_list = []
        self.fitness_list = []

        selection_probability = np.random.dirichlet(np.ones(2*self.params['ind_num']), size=1)[0]
        self.selection_probability = sorted(selection_probability, reverse=True)

    def iteration(self):
        self.ind_list = [Individual(self.params, self.distance_matrix) for _ in range(self.params['ind_num'])]
        self.best = self.ind_list[0]
        for _ in range(self.params['iteration_number']):
            self.next_gen()
            result = copy(self.best.sequence_encodding)
            result.append(result[0])
            self.result_list.append(result)
            self.fitness_list.append(self.best.fitness)
        return self.result_list, self.fitness_list

    def cross(self):
        new_gen = []
        random.shuffle(self.ind_list)
        for individ in range(0, self.params['ind_num'] - 1, 2):
            sequence1 = copy(self.ind_list[individ].sequence_encodding)
            sequence2 = copy(self.ind_list[individ + 1].sequence_encodding)
            start_pos = random.randint(0, self.params['sequence_len'] - 2)
            end_pos = random.randint(start_pos, self.params['sequence_len'] - 1)
            pos1_recorder = {value: idx for idx, value in enumerate(sequence1)}
            pos2_recorder = {value: idx for idx, value in enumerate(sequence2)}
            for sequence_pos in range(start_pos, end_pos):
                value1, value2 = sequence1[sequence_pos], sequence2[sequence_pos]
                pos1, pos2 = pos1_recorder[value2], pos2_recorder[value1]
                sequence1[sequence_pos], sequence1[pos1] = sequence1[pos1], sequence1[sequence_pos]
                sequence2[sequence_pos], sequence2[pos2] = sequence2[pos2], sequence2[sequence_pos]
                pos1_recorder[value1], pos1_recorder[value2] = pos1, sequence_pos
                pos2_recorder[value1], pos2_recorder[value2] = sequence_pos, pos2
            new_gen.append(Individual(self.params, self.distance_matrix, sequence1))
            new_gen.append(Individual(self.params, self.distance_matrix, sequence2))
        return new_gen

    def mutate(self, new_gen):
        for ind in new_gen:
            if random.random() < self.params['mutate_rate']:
                old_sequence = copy(ind.sequence_encodding)
                start_pos = random.randint(0, self.params['sequence_len'] - 2)
                end_pos = random.randint(start_pos, self.params['sequence_len'] - 1)
                sequence_mutate = old_sequence[start_pos:end_pos]
                sequence_mutate.reverse()
                ind.sequence_encodding = old_sequence[:start_pos] + sequence_mutate + old_sequence[end_pos:]
        self.ind_list += new_gen

    def select(self):
        # 选择排序法，目前还有bug
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
        competition_num = 15
        competitor_number = 15
        winner_number = self.params['ind_num'] // competition_num
        winners = []
        for _ in range(competition_num):
            competitors = []
            for _ in range(competitor_number):
                competitor = random.choice(self.ind_list)
                competitor = Individual(self.params, self.distance_matrix, competitor.sequence_encodding)
                competitors.append(competitor)
            winners += sorted(competitors, key=lambda x:x.fitness)[:winner_number]
        self.ind_list = winners

    def next_gen(self):
        new_gen = self.cross()
        self.mutate(new_gen)
        self.select()
        for ind in self.ind_list:
            if ind.fitness < self.best.fitness:
                self.best = ind


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
