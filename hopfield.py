import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

class HM:
    def __init__(self, params, city_positions, distance_matrix):
        self.params = params
        self.city_positions = city_positions
        self.distance_matrix = distance_matrix
        
        self.city_numbers = len(city_positions)
        self.rho_1 = self.city_numbers * self.city_numbers
        self.rho_2 = self.city_numbers / 2

    def swap(self, v):
        tmp_v = v.copy()
        for i in range(len(tmp_v) - 1):
            tmp_v[:, i] = tmp_v[: ,i + 1]
        tmp_v[:, -1] = v[:, 0]
        return tmp_v

    def get_route_length(self, path):
        dis = 0.0
        for i in range(len(path) - 1):
            dis += self.distance_matrix[path[i]][path[i+1]]
        return dis


    def du_func(self, V):
        a = np.sum(V, axis=0) - 1
        b = np.sum(V, axis=1) - 1
        city_numbers = self.city_numbers
        rho_1 = self.rho_1
        rho_2 = self.rho_2
        t1 = np.zeros((city_numbers, city_numbers))
        t2 = np.zeros((city_numbers, city_numbers))
        for i in range(city_numbers):
            for j in range(city_numbers):
                t1[i, j] = a[j]
        for i in range(city_numbers):
            for j in range(city_numbers):
                t2[j, i] = b[j]
        c = self.swap(V)
        c = np.dot(self.distance_matrix, c)
        return -rho_1 * (t1 + t2) - rho_2 * c

    def energy_func(self, V):
        city_numbers = self.city_numbers
        rho_1 = self.rho_1
        rho_2 = self.rho_2
        t1 = np.sum(np.power(np.sum(V, axis=0) - 1, 2))
        t2 = np.sum(np.power(np.sum(V, axis=1) - 1, 2))
        Vt = self.swap(V)
        t3 = self.distance_matrix * Vt
        t3 = np.sum(np.sum(np.multiply(V, t3)))
        energy = 0.5 * (rho_1 * (t1 + t2) + rho_2 * t3)
        return energy

    def get_tsp_path(self, V):
        city_numbers = self.city_numbers
        newV = np.zeros([city_numbers, city_numbers])
        route = []
        for i in range(city_numbers):
            mm = np.max(V[:, i])
            for j in range(city_numbers):
                if V[j, i] == mm:
                    newV[j, i] = 1
                    route += [j]
                    break
        return route, newV

    def draw_energys(self, energys):
        plt.plot(np.arange(0, len(energys), 1), energys)
        plt.title("energy")
        # plt.show()
        plt.savefig("result/energy.jpg")

    def hopfield(self):
        U0 = 0.001
        step = 0.0003
        num_iter = 10000
        U = 1 / 2 * U0 * np.log(self.city_numbers - 1) + (2 * (np.random.random((self.city_numbers, self.city_numbers))) - 1)
        V = 1 / 2 * (1 + np.tanh(U / U0))
        energys = np.array([0.0 for x in range(num_iter)])
        best_distance = np.inf
        best_route = []
        for n in tqdm(range(num_iter)):
            du = self.du_func(V)
            U = U + du * step
            V = 1 / 2 * (1 + np.tanh(U / U0))
            energys[n] = self.energy_func(V)
            route, newV = self.get_tsp_path(V)
            if len(np.unique(route)) == self.city_numbers:
                route.append(route[0])
                dis = self.get_route_length(route)
                if dis < best_distance:
                    best_route.clear()
                    best_distance = dis
                    for i in range(len(route)):
                        best_route.append(route[i])
                    best_route.append(route[0])

        if len(best_route) > 0:
            self.draw_energys(energys)
            return best_route
        else:
            print('No optimal solution')
            return None
        



# hopfield()
