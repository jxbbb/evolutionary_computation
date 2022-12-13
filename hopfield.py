import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

class HM:
    def __init__(self, params, city_positions, distance_matrix):
        self.params = params
        self.city_positions = city_positions
        self.distance_matrix = distance_matrix
        
        self.distance = None
        self.N = None
        self.A = None
        self.D = None


    def price_cn(self, vec1, vec2):
        return np.linalg.norm(np.array(vec1) - np.array(vec2))
    def calc_distance(self, path):
        dis = 0.0
        for i in range(len(path) - 1):
            dis += self.distance[path[i]][path[i+1]]
        return dis

    def get_distance(self, citys):
        N = len(citys)
        distance = np.zeros((N, N))
        for i, curr_point in enumerate(citys):
            line = []
            [line.append(self.price_cn(curr_point, other_point)) if i != j else line.append(0.0) for j, other_point in enumerate(citys)]
            distance[i] = line
        self.distance = distance
        return distance

    def calc_du(self, V, distance):
        a = np.sum(V, axis=0) - 1
        b = np.sum(V, axis=1) - 1
        N = self.N
        A = self.A
        D = self.D
        t1 = np.zeros((N, N))
        t2 = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                t1[i, j] = a[j]
        for i in range(N):
            for j in range(N):
                t2[j, i] = b[j]
        c_1 = V[:, 1:N]
        c_0 = np.zeros((N, 1))
        c_0[:, 0] = V[:, 0]
        c = np.concatenate((c_1, c_0), axis=1)
        c = np.dot(distance, c)
        return -A * (t1 + t2) - D * c

    def calc_U(self, U, du, step):
        return U + du * step

    def calc_V(self, U, U0):
        return 1 / 2 * (1 + np.tanh(U / U0))

    def calc_energy(self, V, distance):
        N = self.N
        A = self.A
        D = self.D
        t1 = np.sum(np.power(np.sum(V, axis=0) - 1, 2))
        t2 = np.sum(np.power(np.sum(V, axis=1) - 1, 2))
        idx = [i for i in range(1, N)]
        idx = idx + [0]
        Vt = V[:, idx]
        t3 = distance * Vt
        t3 = np.sum(np.sum(np.multiply(V, t3)))
        e = 0.5 * (A * (t1 + t2) + D * t3)
        return e

    def check_path(self, V):
        N = self.N
        A = self.A
        D = self.D
        newV = np.zeros([N, N])
        route = []
        for i in range(N):
            mm = np.max(V[:, i])
            for j in range(N):
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
        print(citys)
        distance = self.get_distance(citys)
        self.N = len(citys)
        self.A = self.N * self.N
        self.D = self.N / 2
        U0 = 0.0009
        step = 0.0001
        num_iter = 10000
        U = 1 / 2 * U0 * np.log(self.N - 1) + (2 * (np.random.random((self.N, self.N))) - 1)
        V = self.calc_V(U, U0)
        energys = np.array([0.0 for x in range(num_iter)])
        best_distance = np.inf
        best_route = []
        H_path = []
        for n in tqdm(range(num_iter)):
            du = self.calc_du(V, distance)
            U = self.calc_U(U, du, step)
            V = self.calc_V(U, U0)
            energys[n] = self.calc_energy(V, distance)
            route, newV = self.check_path(V)
            if len(np.unique(route)) == self.N:
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
            print('没有找到最优解')
            return None
        



# hopfield()