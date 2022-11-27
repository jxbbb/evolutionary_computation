import numpy as np
from ga import Ga
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
    assert params['n'] >= 10

def TSP_init():
    params = {
        "n": 30,
    }
    check_params(params)
    city_pos_list = np.random.rand(params['n'], 2)
    # 城市距离矩阵
    city_dist_mat = build_dist_mat(params, city_pos_list)
    return params, city_pos_list, city_dist_mat  

def ga_init():
    params = {
        'gene_len': 30,
        "individual_num": 60,
        "gen_num": 800,
        "mutate_prob": 0.3,
        "method": 'ga',
    }
    return params

def ant_init():
    params = {
        "individual_num": 60,
        "gen_num": 800,
        "method": 'ant',
    }
    return params

def build_dist_mat(params, input_list):
    n = params['n']
    dist_mat = np.zeros([n, n])
    for i in range(n):
        for j in range(i + 1, n):
            d = input_list[i, :] - input_list[j, :]
            # 计算点积
            dist_mat[i, j] = np.dot(d, d)
            dist_mat[j, i] = dist_mat[i, j]
    return dist_mat


def visualizer(city_pos_list, result_pos_list, fitness_list):
    plt.figure()
    plt.scatter(city_pos_list[:, 0],city_pos_list[:, 1])
    plt.title("start")
    # plt.legend()
    plt.show()
    plt.savefig("start.jpg")

    plt.figure()
    plt.plot(result_pos_list[:, 0], result_pos_list[:, 1])
    plt.title("route")
    # plt.legend()
    plt.show()
    plt.savefig("results.jpg")

    plt.figure()
    plt.plot(fitness_list)
    plt.title("fit ourte")
    # plt.legend()
    plt.show()
    plt.savefig("ga_0.9.jpg")



if __name__ == '__main__':

    methods = ['ga', 'ant']

    params, city_pos_list, city_dist_mat  = TSP_init()
    print(city_pos_list)
    print(city_dist_mat)


    for method in methods:
        if method == 'ga':
            params.update(ga_init())
            # 遗传算法运行
            ga = Ga(params, city_dist_mat)
            result_list, fitness_list = ga.train()
            result = result_list[-1]
            result_pos_list = city_pos_list[result, :]
        elif method == 'ant':
            params.update(ant_init())
        else:
            raise Exception("Methods not implemented")

        visualizer(city_pos_list, result_pos_list, fitness_list)
