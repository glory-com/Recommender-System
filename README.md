# 推荐系统设计

---
## 一、认识推荐系统


### 1、衡量推荐系统和和好坏的标准：北极星指标。

什么是北极星指标：北极星指标（North Star Metric），也叫作第一关键指标（One Metric That Matters），是指在产品的当前阶段与业务/战略相关的绝对核心指标，一旦确立就像北极星一样闪耀在空中，指引团队向同一个方向迈进（提升这一指标）。

主要北极星指标
- 日活数（DAD）/月活数（MAU）：每一个不同的用户使用一次系统，就记作一个日活数，同理对于月活数
- 消费（广义）：使用系统的时长，消费数额等
- 发布：用户的发布作品的数量


### 2、推荐系统基本流程
```mermaid
graph TD 
    a[数据库] --> A 
    A[几亿条数据] --召回通道（1）--> B[几百条数据]
    A --召回通道（2）--> C[几百条数据]
    A --召回通道（3）--> D[几百条数据]

    B --粗排/细排--> E[几百条数据]
    C --粗排/细排--> E
    D --粗排/细排--> E

    E --随机抽样--> F[几十条数据]
    F --> G[用户界面]
```

召回是为了快速减少数据量，方便后续筛选。

粗排和细排通过ai模型等快速对数据打分，粗派使用简单模型，减少时间，精排用复杂模型，提高精度，最后通过随机抽样来推送用户。

----

## 二、推荐系统的算法

### 1.基于物品的协同过滤(ItemCF)

ItemCF：物品协同过滤（Item-based Collaborative Filtering，简称 ItemCF）是一种基于物品相似性的推荐算法。它的核心思想是：如果用户喜欢某个物品，那么用户可能会喜欢与该物品相似的其他物品。

$$
\text{sim}(i, j) = \frac{\sum_{u \in U} r(u, i) \cdot r(u, j)}{\sqrt{\sum_{u \in U} r(u, i)^2} \cdot \sqrt{\sum_{u \in U} r(u, j)^2}}
$$


*举例子*

|      | ItemX  | ItemY  | ItemZ  |
|------|------|------|------|
| UserA  | 1    | 0    | 1    |
| UserB  | 1    | 1    | 0    |
| UserC  | 0    | 1    | 0    |

解释：有三位用户A,B,C，有三个物品X,Y,Z(1表示喜欢，0表示不喜欢)


$$
sin(X,Y) = \frac{1\cdot0+1\cdot1+0\cdot1}{\sqrt{1^2+1^2+0^2}\cdot\sqrt{0^2+1^2+1^2}} = \frac{1}{2}
$$

$$
sin(X,Z) = \frac{1\cdot1+1\cdot0+0\cdot0}{\sqrt{1^2+1^2+0^2}\cdot\sqrt{1^2+0^2+0^2}} = \frac{1}{\sqrt{2}}
$$

$$
sin(Y,Z) = \frac{0\cdot1+1\cdot0+1\cdot0}{\sqrt{0^2+1^2+1^2}\cdot\sqrt{1^2+0^2+0^2}} = 0 
$$

$$
\text{简单地，可以得到，计算物品x,y的相似度时，取Item的向量，}\vec{x}\text{和}\vec{y}
\text{则}
$$

$$
\sin(\vec{x}, \vec{y}) = \frac{\vec{x} \cdot \vec{y}}{||\vec{x}||\cdot||\vec{y}||}
$$

python程序
```python
import numpy as np
def calc_similarity(x,y) :
    norm_x = np.linalg.norm(x)
    norm_y = np.linalg.norm(y)
    dot_xy = np.dot(x , y)
    if (norm_x == 0) or (norm_y == 0) :
        return 0 
    return dot_xy / (norm_x * norm_y) 
```

预估目标用户对于物品x的喜爱程度

$$
\text{like}(\text{user},\text{item}) = \sum_j \text{like}(\text{user}, \text{item}_j) \cdot \text{sim}(\text{item}_j, \text{item})
$$

返回几百条喜爱程度最大的数据，作为一个召回通道

### 2.Swing模型

$$
\text{sim}(x,y) = \sum_{u_1 \in \mathcal{V}} \sum_{u_2 \in \mathcal{V}} \frac{1}{a + \text{overlap}(u_1, u_2)} （a为非负超参数）
$$

*举例子*

|      | ItemX  | ItemY  |
|------|------|------|
| User1  | 1    | 1    | 
| User2  | 0    | 0    | 
| User3  | 0    | 0    | 


重叠矩阵为：

$$
\text{overlap} =
\begin{bmatrix}
2 & 1 & 1 \\
1 & 1 & 0 \\
1 & 0 & 1
\end{bmatrix}
$$

平滑参数设为：

$$
a = 1
$$

$$
\frac{1}{a + \text{overlap}(u_1, u_1)} = \frac{1}{1 + 2} = \frac{1}{3}
$$

$$
\frac{1}{a + \text{overlap}(u_1, u_2)} = \frac{1}{1 + 1} = \frac{1}{2}
$$

$$
\frac{1}{a + \text{overlap}(u_1, u_3)} = \frac{1}{1 + 1} = \frac{1}{2}
$$

类似地，其他用户对的贡献可以计算如下：

$$
\text{sim}(i_1, i_2) = \frac{1}{3} + \frac{1}{2} + \frac{1}{2} + \frac{1}{2} + \frac{1}{2} + 1 + \frac{1}{2} + 1 + \frac{1}{2}
$$

化简结果：

$$
\text{sim}(i_1, i_2) = \frac{1}{3} + \frac{5}{2} + 2 = \frac{1}{3} + \frac{10}{3} = \frac{11}{3}
$$

最终结果：

$$
\text{sim}(i_1, i_2) \approx 5.3333
$$

```python
def swing(X , a):
    sim = 0 
    for i in range(len(X)) :
        for j in range(len(X)) :
            useri = X[i] 
            userj = X[j]
            sim += 1 / (a + (useri[0] + userj[0] == 2) + (useri[1] + userj[1] == 2))
    return sim 
```


### 3.基于用户的协同过滤(UserCF)

UserCF：用户协同过滤（User-based Collaborative Filtering，简称 UserCF）是一种基于用户相似性的推荐算法。它的核心思想是：如果两个用户喜欢相似的物品，那么目标用户可能也会喜欢与相似用户喜欢的其他物品。

$$
I = A \cap B 
(A为用户u喜欢的集合，B为用户v喜欢的集合)
$$

$$
\text{sim}(u, v) = \frac{|I|}{\sqrt{|A|\cdot|B|}}
$$

*举例子*

|       | ItemX | ItemY | ItemZ |
|-------|-------|-------|-------|
| UserA | 1     | 0     | 1     |
| UserB | 1     | 1     | 0     |
| UserC | 0     | 1     | 0     |

解释：有三位用户 A, B, C，有三个物品 X, Y, Z (1 表示喜欢，0 表示不喜欢)

$$
sin(UserA, UserB) = \frac{1}{\sqrt{2\cdot2}} = \frac{1}{2}
$$

$$
sin(UserA, UserC) = \frac{0}{\sqrt{2\cdot1}} = 0 
$$

$$
sin(UserB, UserC) = \frac{1}{\sqrt{2\cdot1}} = \frac{1}{\sqrt{2}}
$$


python程序

```python
import numpy as np
def calc_similarity(x, y):
    lenx = x.shape[0]
    leny = y.shape[0]
    I = np.sum(x == y)

    if lenx == 0 or leny == 0 :
        return 0 

    return I / (lenx * leny) ** 0.5
```

为了降低热门物品的权重，需要需改分子的大小，让越多人喜欢的物品的分子值越小，平衡热门度

$$
\text sin(u,v) = \frac{\sum_{l \in I}\frac{1}{\log(1+n_l)}}{\sqrt{|A|\cdot|B|}}
$$


预估目标用户对于物品x的喜爱程度

$$
\text{like}(\text{user},\text{item}) = \sum_j \text{like}(\text{user}_j, \text{item}) \cdot \text{sim}(\text{user}_j, \text{user})
$$


#### Q:为什么没有归一化的操作？
* A:因为在召回的结果中，注重的是分数的排序，而不是真实的预估值，所以基于排序只需要知道大小，而不需要归一化。但是为了标准，可自行选择归一化方式。


### 4.最近邻查找系统
给出一个大量的数据集（假设为二维数据集），给定一个目标target，找到与目标距离最近的点。给定计算点A、B距离的公式：

$$
\text{Distance} = 1 - \cos(\theta) = 1 - \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \cdot \|\mathbf{B}\|}
$$

```python
import matplotlib.pyplot as plt 
import pandas as pd 
import time 
import numpy as np 

def calc_dis(x, y):
    x = np.array(x)
    y = np.array(y)
    if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
        return float('inf')  
    return 1 - (np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
```

使用优先队列来维护距离最近的三个点。

```python
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
```


如果使用传统的枚举法，计算复杂度O(n)，当n非常大的时候，计算时间很长。
```python
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
```

所以采用最近邻查找系统(NearestNeibourSearchSystem),手动地将数据点分块，比如，以余弦距离为标准的系统，将二维数据点按照角度分区，在分一个分区中，求出平均点，计算平均点与目标点的距离，得到最近的平均点。在这个平均点的分区中，枚举所有点和目标点的的距离，找到最近点。

```python
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
```



运行两种查找策略
```python
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
```

运行结果

```markdown
第 3507517 个点，余弦距离为 1.5543122344752192e-15，点的坐标为 (9.600203437728684, -4.114372247528461)
第 3523773 个点，余弦距离为 3.206324095117452e-13，点的坐标为 (2.709546778492818, -1.161231765444164)
第 2501700 个点，余弦距离为 8.26116952623579e-13，点的坐标为 (3.7898366285952783, -1.624209932014855)
NNSS用时 7.723 秒

第 3507517 个点，余弦距离为 1.5543122344752192e-15，点的坐标为 (9.600203437728684, -4.114372247528461)
第 3523773 个点，余弦距离为 3.206324095117452e-13，点的坐标为 (2.709546778492818, -1.161231765444164)
第 2501700 个点，余弦距离为 8.26116952623579e-13，点的坐标为 (3.7898366285952783, -1.624209932014855)
Enumeration用时 32.573 秒
```

发现了提高了76%的运算效率


### 5.双塔模型
