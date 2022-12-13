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


    def price_cn(self, vec1, vec2):
        return np.linalg.norm(np.array(vec1) - np.array(vec2))

    def calc_distance(self, path):
        dis = 0.0
        for i in range(len(path) - 1):
            dis += self.distance_matrix[path[i]][path[i+1]]
        return dis


    def calc_du(self, V, distance):
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
        c_1 = V[:, 1:city_numbers]
        c_0 = np.zeros((city_numbers, 1))
        c_0[:, 0] = V[:, 0]
        c = np.concatenate((c_1, c_0), axis=1)
        c = np.dot(distance, c)
        return -rho_1 * (t1 + t2) - rho_2 * c

    def calc_U(self, U, du, step):
        return U + du * step

    def calc_V(self, U, U0):
        return 1 / 2 * (1 + np.tanh(U / U0))

    def calc_energy(self, V, distance):
        city_numbers = self.city_numbers
        rho_1 = self.rho_1
        rho_2 = self.rho_2
        t1 = np.sum(np.power(np.sum(V, axis=0) - 1, 2))
        t2 = np.sum(np.power(np.sum(V, axis=1) - 1, 2))
        idx = [i for i in range(1, city_numbers)]
        idx = idx + [0]
        Vt = V[:, idx]
        t3 = distance * Vt
        t3 = np.sum(np.sum(np.multiply(V, t3)))
        e = 0.5 * (rho_1 * (t1 + t2) + rho_2 * t3)
        return e

    def check_path(self, V):
        city_numbers = self.city_numbers
        rho_1 = self.rho_1
        rho_2 = self.rho_2
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
        citys = self.city_positions
        # print(citys)
        distance = self.distance_matrix

        U0 = 0.0009
        step = 0.0001
        num_iter = 10000
        U = 1 / 2 * U0 * np.log(self.city_numbers - 1) + (2 * (np.random.random((self.city_numbers, self.city_numbers))) - 1)
        V = self.calc_V(U, U0)
        energys = np.array([0.0 for x in range(num_iter)])
        best_distance = np.inf
        H_path = []
        for n in tqdm(range(num_iter)):
            du = self.calc_du(V, distance)
            U = self.calc_U(U, du, step)
            V = self.calc_V(U, U0)
            energys[n] = self.calc_energy(V, distance)
            route, newV = self.check_path(V)
            if len(np.unique(route)) == self.city_numbers:
                route.append(route[0])
                dis = self.calc_distance(route)
                if dis < best_distance:
                    H_path = []
                    best_distance = dis
                    best_route = route
                    [H_path.append((route[i], route[i + 1])) for i in range(len(route) - 1)]

        if len(H_path) > 0:
            final_path = []
            for h_path in H_path:
                final_path.append(h_path[0])
            final_path.append(H_path[0][0])
            self.draw_energys(energys)
            return final_path
        else:
            print('No optimal solution')
            return None
        



# hopfield()
