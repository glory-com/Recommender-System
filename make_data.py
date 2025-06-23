import pandas as pd
import numpy as np


num_rows = 1000000

data = {
    'x': np.random.uniform(-10, 10, num_rows),
    'y': np.random.uniform(-10, 10, num_rows)
}

df = pd.DataFrame(data)

# 保存为 CSV 文件
df.to_csv("data.csv", index=False)