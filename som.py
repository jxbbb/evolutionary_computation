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
        self.lr = 0.8
        self.num_neurons = 10*len(city_positions)
        
        cities = []
        for i in range(len(self.city_positions)):
            cities.append([i, self.city_positions[i][0], self.city_positions[i][1]])

        cities = np.array(cities)
        self.cities = cities

    def update_lr(self):
        self.lr = self.lr * 0.99997

    def update_neurons(self):
        self.num_neurons = self.num_neurons * 0.9997

    def init_som(self):
        x_max = self.city_positions[:,0].max(axis=0)
        y_max = self.city_positions[:,1].max(axis=0)
    
        network1 = np.random.rand(self.num_neurons, 1)*x_max
        network2 = np.random.rand(self.num_neurons, 1)*y_max

        return np.concatenate((network1, network2), axis=1)

    def get_route_length(self, path):
        dis = 0.0
        for i in range(len(path) - 1):
            dis += self.distance_matrix[path[i]][path[i+1]]
        return dis

    def city_neuron_match(self, som):
        city_neuron_matches = []
        for i in range(len(self.city_positions)):
            city_position = self.city_positions[i]
            nearest_city = -1
            min_dis = float('inf')
            for neuron in range(len(som)):
                dis = math.sqrt(sum((city_position - som[neuron])**2))
                if dis < min_dis:
                    min_dis = dis
                    nearest_city = neuron
            city_neuron_matches.append([i, nearest_city])
        city_neuron_matches = sorted(city_neuron_matches, key=lambda x:x[1])
        city_neuron_matches.append(city_neuron_matches[0])
        return [city_neuron_matches[j][0] for j in range(len(city_neuron_matches))]

    def som(self):    
        som = self.init_som()     

        iterations = 50000
        os.makedirs("./result", exist_ok=True)
        for i in tqdm(range(iterations)):
            select_city = random.randint(0,len(self.city_positions)-1)
            city = self.city_positions[select_city]
    
            nearest_neuron = -1
            min_dis = float('inf')
            for neuron in range(len(som)):
                dis = math.sqrt(sum((city - som[neuron])**2))
                if dis < min_dis:
                    min_dis = dis
                    nearest_neuron = neuron
                
            deltas = np.absolute(nearest_neuron - np.arange(som.shape[0]))
            distances = np.minimum(deltas, som.shape[0] - deltas)
            
            filter = np.exp(-(distances*distances) / (2*(max(self.num_neurons // 10, 1)**2)))
            som += filter[:, np.newaxis] * self.lr * (city - som)

            self.update_neurons()
            self.update_lr()

            if i%1000 == 0:
                plt.scatter(self.city_positions[:,0], self.city_positions[:,1])
                plt.plot(som[:,0], som[:,1])
                plt.savefig('./result/iter{}.jpg'.format(i))
                plt.close()
    
    
            if self.num_neurons < 1 or self.lr < 0.001:
                break
    
        new_path = self.city_neuron_match(som)
        return self.city_positions[new_path]
 
 
 
