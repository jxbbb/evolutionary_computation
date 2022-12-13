import numpy as np
import matplotlib.pyplot as plt
import argparse
import random
from copy import copy
from tqdm import tqdm
import os

from sga import SGA
from hopfield import HM
from som import SOM

seed = 456

def seed_everything(seed):
    """
        Ensure the different methods have the same data.
    """
    if seed >= 10000:
        raise ValueError("seed number should be less than 10000")

    np.random.seed(seed)
    random.seed(seed)

seed_everything(seed)

def check_params(params):
    assert params['city_numbers'] >= 10

def TSP_init():
    params = {
        "city_numbers": 25,
    }
    check_params(params)
    city_positions = np.random.rand(params['city_numbers'], 2)*10

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
        'sequence_len': 15,
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

def hm_init():
    params = {
        "ind_num": 60,
        "iteration_number": 800,
        "method": 'ant',
    }
    return params

def visualizer(city_positions, results, training_processes=None):
    plt.figure()
    plt.scatter(city_positions[:, 0],city_positions[:, 1])
    plt.title("start")
    # plt.show()
    plt.savefig("result/start.jpg")

    plt.figure()
    plt.plot(results[:, 0], results[:, 1])
    plt.title("route")
    # plt.show()
    plt.savefig("result/results.jpg")

    if training_processes is not None:
        plt.figure()
        plt.plot(training_processes)
        plt.title("training_process")
        # plt.show()
        plt.savefig("result/training_process.jpg")



if __name__ == '__main__':

    methods = ['sga', 'ant', 'hopfield', 'som', 'all']
    methods = ['hopfield']
    params, city_positions, distance_matrix  = TSP_init()

    os.makedirs("./result", exist_ok=True)

    for method in methods:
        print(f"********** method: {method}\t**********")
        if method == 'sga':
            params.update(ga_init())
            sga = SGA(params, city_positions, distance_matrix)
            result_list, training_processes = sga.iteration_process()
            results = city_positions[result_list[-1], :]
            visualizer(city_positions, results, training_processes)
        elif method == 'ant':
            params.update(ant_init())
            # TODO: implemented
        elif method == 'hopfield':
            params.update(ga_init())
            hm = HM(params, city_positions, distance_matrix)
            result_list = hm.hopfield()
            results = city_positions[result_list, :]
            visualizer(city_positions, results, training_processes=None)
        elif method == 'som':
            params.update(ga_init())
            hm = SOM(params, city_positions, distance_matrix)
            final_path = hm.som()
            visualizer(city_positions, final_path, training_processes=None)
        else:
            raise Exception("Methods not implemented")

        