import numpy as np
import random
import math
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

class SOM:
    def __init__(self, params, city_positions, distance_matrix):
        self.params = params
        self.city_positions = city_positions
        self.distance_matrix = distance_matrix


    def som(self):
        citys = []
        for i in range(len(self.city_positions)):
            citys.append([i, self.city_positions[i][0], self.city_positions[i][1]])

        lr = 0.8
        n = len(citys)*10
        citys = np.array(citys)
        temp_citys = citys.copy()
    
        #normalize
        x_min, y_min = citys.min(axis=0)[1:]
        x_max, y_max = citys.max(axis=0)[1:]
        citys[:,1] = (citys[:,1]-x_min)/(x_max-x_min)
        citys[:,2] = (citys[:,2]-y_min)/(y_max-y_min)
        # for i in range(len(citys)):
        #     citys[i][1], citys[i][2] = (citys[i][1]-min_c)/(max_c-min_c), (citys[i][2]-min_c)/(max_c-min_c)
    
        network = np.random.rand(n, 2)
        iterations = 100000
        os.makedirs("./result", exist_ok=True)
        for i in tqdm(range(iterations)):
            select_city = random.randint(0,len(citys)-1)
            city = citys[select_city][1:]
    
            nearest_n = -1
            min_dis = float('inf')
            for j in range(len(network)):
                dis = math.sqrt(sum(pow(city - network[j], 2)))
                if dis < min_dis:
                    min_dis = dis
                    nearest_n = j
    
            radix = n // 10
            if radix < 1:
                radix = 1
            
            deltas = np.absolute(nearest_n - np.arange(network.shape[0]))
            distances = np.minimum(deltas, network.shape[0] - deltas)
            
            gaussian = np.exp(-(distances*distances) / (2*(radix*radix)))
            network += gaussian[:, np.newaxis] * lr * (city - network)
            n = n * 0.9997
            lr = lr * 0.99997
    
            if i%1000 == 0:
                plt.scatter(citys[:,1], citys[:,2])
                plt.plot(network[:,0], network[:,1])
                plt.savefig('./result/iter{}.jpg'.format(i))
                plt.close()
    
    
            if n < 1:
                print('Radius has completely decayed, finishing execution',
                'at {} iterations'.format(i))
                break
            if lr < 0.001:
                print('Learning rate has completely decayed, finishing execution',
                'at {} iterations'.format(i))
                break
    
    
        new_citys = []
        for i in range(len(citys)):
            city = citys[i][1:]
            nearest_city = -1
            min_dis = float('inf')
            for j in range(len(network)):
                dis = math.sqrt(sum(pow(city - network[j], 2)))
                if dis < min_dis:
                    min_dis = dis
                    nearest_city = j
            new_citys.append([i, nearest_city])
    
        new_citys_ = sorted(new_citys, key=lambda x:x[1])
        new_citys_.append(new_citys_[0])
        new_citys_ = np.array(new_citys_)
        final_path = temp_citys[new_citys_[:,0],:][:,1:]
        path_lenght = 0
        for i in range(len(final_path)-1):
            path_lenght += math.sqrt(sum(pow(final_path[i] - final_path[i+1], 2)))
        print('final distance:{}'.format(path_lenght))
        return final_path
 
 
 
