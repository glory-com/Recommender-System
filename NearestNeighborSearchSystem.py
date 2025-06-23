import matplotlib.pyplot as plt 
import pandas as pd 
import time 
import numpy as np 

df = pd.read_csv("data.csv")
target_x = 7
target_y = -3

def calc_dis(x, y):
    x = np.array(x)
    y = np.array(y)
    if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
        return float('inf')  
    return 1 - (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

class priority_queue:
    def __init__(self, df=None, MAX_VOLUMN=3):
        self.item = []
        self.vol = MAX_VOLUMN
        self.df = df

    def pop(self):
        return self.item.pop()

    def add(self, x):
        if len(self.item) < self.vol:
            self.item.append(x)
            self.item.sort()
        elif x[0] < self.item[-1][0]:
            self.item[-1] = x
            self.item.sort()

    def show(self):
        for point in self.item:
            print(f"第 {point[1]} 个点，余弦距离为 {point[0]}，点的坐标为 "
                  f"{(float(self.df.iloc[point[1]]['x']), float(self.df.iloc[point[1]]['y']))}")

class NearestSystem:
    def __init__(self, data, target_x, target_y, parts=360):
        self.data = data
        self.target_x = target_x
        self.target_y = target_y
        self.parts = parts
        self.points = [[] for _ in range(self.parts)]  
        self.best_points = []

    def part_points(self):
        xs = self.data['x'].values
        ys = self.data['y'].values
        indices = self.data.index.values

        angles = (np.arctan2(ys, xs) + 2 * np.pi) % (2 * np.pi)
        sector_indices = (angles / (2 * np.pi / self.parts)).astype(int)

        for i in range(self.parts):
            mask = (sector_indices == i)
            sector_points = np.column_stack((xs[mask], ys[mask], indices[mask]))
            self.points[i] = sector_points.tolist() 

    def avg_vec(self):
        mn_dis = float('inf')
        mn_idx = -1

        for i in range(self.parts):
            sector = self.points[i]
            if not sector:
                continue
            sector_array = np.array(sector)
            avg_x = np.mean(sector_array[:, 0])
            avg_y = np.mean(sector_array[:, 1])
            dis = calc_dis([avg_x, avg_y], [self.target_x, self.target_y])
            if dis < mn_dis:
                mn_dis = dis
                mn_idx = i

        self.best_points = self.points[mn_idx] if mn_idx != -1 else []

    def find_nearest(self):
        pri_q = priority_queue(self.data)
        if not self.best_points:
            pri_q.show()
            return

        best_points_array = np.array(self.best_points)
        coords = best_points_array[:, :2]
        indices = best_points_array[:, 2].astype(int)

        target = np.array([self.target_x, self.target_y])

        dot_products = np.einsum('ij,j->i', coords, target)
        norms = np.linalg.norm(coords, axis=1) * np.linalg.norm(target)
        distances = 1 - dot_products / norms

      
        for dis, idx in zip(distances, indices):
            pri_q.add([dis, int(idx)])

        pri_q.show()

def NNSS():
    start = time.time()
    nnss = NearestSystem(df, target_x, target_y)
    nnss.part_points()
    nnss.avg_vec()
    nnss.find_nearest()
    end = time.time()
    print(f"NNSS用时 {round((end - start), 3)}秒")

def Enumeration():
    pri_q = priority_queue(df, 3)
    start = time.time()

    xs = df['x'].values
    ys = df['y'].values
    indices = df.index.values

    points = np.column_stack((xs, ys))
    target = np.array([target_x, target_y])
    for i in range(len(indices)):
        x = points[i]
        dis = calc_dis(x, target)  
        pri_q.add([dis, indices[i]])

    pri_q.show()

    end = time.time()
    print(f"Enumeration用时 {round(end - start, 3)}秒")

NNSS()
Enumeration()