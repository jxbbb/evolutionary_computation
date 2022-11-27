import numpy as np
from sga import SGA
import matplotlib.pyplot as plt
import argparse
import torch
import random


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
    }
    return params

def ant_init():
    params = {
        "ind_num": 60,
        "iteration_number": 800,
        "method": 'ant',
    }
    return params


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
    plt.title("fit ourte")
    plt.show()
    plt.savefig("sga.jpg")



if __name__ == '__main__':

    methods = ['sga', 'ant']

    params, city_positions, distance_matrix  = TSP_init()
    print(city_positions)
    print(distance_matrix)


    for method in methods:
        if method == 'sga':
            params.update(ga_init())
            sga = SGA(params, distance_matrix)
            result_list, training_processes = sga.iteration()
            result = result_list[-1]
            results = city_positions[result, :]
        elif method == 'ant':
            params.update(ant_init())
            # not implemented
        else:
            raise Exception("Methods not implemented")

        visualizer(city_positions, results, training_processes)
