import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import random
from copy import copy
from tqdm import tqdm

seed = 777

def seed_everything(seed):
    """
        Ensure the different methods have the same data.
    """
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    else:
        rank = 0
    seed = (rank * 100000) + seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

seed_everything(seed)

def check_params(params):
    assert params['city_numbers'] >= 10

def TSP_init():
    params = {
        "city_numbers": 30,
    }
    check_params(params)
    city_positions = np.random.rand(params['city_numbers'], 2)

    city_numbers = params['city_numbers']
    distance_matrix = np.zeros([city_numbers, city_numbers])
    for i in range(city_numbers):
        for j in range(i + 1, city_numbers):
            d = city_positions[i, :] - city_positions[j, :]
            distance_matrix[i, j] = np.dot(d, d)
            distance_matrix[j, i] = distance_matrix[i, j]
    return params, city_positions, distance_matrix  

def ga_init():
    params = {
        'sequence_len': 30,
        "ind_num": 60,
        "iteration_number": 800,
        "mutate_rate": 0.3,
        "method": 'sga',
        "select_method": 'championship'
    }
    return params

def ant_init():
    params = {
        "ind_num": 60,
        "iteration_number": 800,
        "method": 'ant',
    }
    return params


class SGA:
    def __init__(self, params, distance_matrix):
        self.params = params
        self.distance_matrix = distance_matrix
        self.best = None
        self.ind_list = []
        self.result_list = []
        self.fitness_list = []

        # Probability of Sorting Selection Method
        selection_probability = np.random.dirichlet(np.ones(2*self.params['ind_num']), size=1)[0]
        self.selection_probability = sorted(selection_probability, reverse=True)

    def iteration_process(self):
        self.ind_list = [Individual(self.params, self.distance_matrix) for _ in range(self.params['ind_num'])]
        self.best = self.ind_list[0]
        for _ in tqdm(range(self.params['iteration_number'])):
            self.iter()
            result = copy(self.best.sequence_encodding)
            result.append(result[0])
            self.result_list.append(result)
            self.fitness_list.append(self.best.fitness)
        print("********** sga done\t**********")
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
        if self.params['select_method'] == 'sort':
            # 选择排序法
            fitness_list = [ind.fitness for ind in self.ind_list]
            rank = np.argsort(fitness_list)
            selection_probability = np.array(self.selection_probability)[rank]

            selected_winners = []
            i = 0
            while i < self.params['ind_num']:
                selected = np.random.choice(self.ind_list, p=selection_probability)
                if selected not in selected_winners:
                    selected_winners.append(selected)
                    i+=1
                else:
                    continue
            
            self.ind_list = selected_winners
        elif self.params['select_method'] == 'championship':
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
        else:
            raise Exception("Select method not implemented")

    def iter(self):
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

def visualizer(city_positions, results, training_processes):
    plt.figure()
    plt.scatter(city_positions[:, 0],city_positions[:, 1])
    plt.title("start")
    plt.show()
    plt.savefig("start.jpg")

    plt.figure()
    plt.plot(results[:, 0], results[:, 1])
    plt.title("route")
    plt.show()
    plt.savefig("results.jpg")

    plt.figure()
    plt.plot(training_processes)
    plt.title("training_process")
    plt.show()
    plt.savefig("training_process.jpg")



if __name__ == '__main__':

    methods = ['sga', 'ant']

    params, city_positions, distance_matrix  = TSP_init()

    for method in methods:
        print(f"********** method: {method}\t**********")
        if method == 'sga':
            params.update(ga_init())
            sga = SGA(params, distance_matrix)
            result_list, training_processes = sga.iteration_process()
            results = city_positions[result_list[-1], :]
        elif method == 'ant':
            params.update(ant_init())
            # not implemented
        else:
            raise Exception("Methods not implemented")

        visualizer(city_positions, results, training_processes)
