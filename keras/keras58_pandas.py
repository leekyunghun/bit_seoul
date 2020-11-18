import numpy as np
import pandas as pd

# pandas는 numpy로 불러올때보다 느리지만 numpy는 pandas만한 기능들이 없다.

datasets = pd.read_csv("./data/csv/iris_ys.csv", header = 0, index_col = 0, sep = ',')
print(datasets)

print(datasets.shape)

# index_col = None, 0, 1   / header = None, 0, 1

#              None           0              1

# None       (151, 6)     (151, 5)        (151, 5)


#  0         (150, 6)     (150, 5)        (150, 5)


#  1         (149, 6)     (149, 5)        (149, 5)

print(datasets.head())       # .head는 위에서 5개 보여줌
print(datasets.tail())       # .tail은 밑에서 5개 보여줌
print(type(datasets))

aaa = datasets.to_numpy()    # pandas를 numpy로 바꿔줌
print(type(aaa))
print(aaa.shape)

bbb = datasets.values
print(type(bbb))
print(bbb.shape)

np.save('./data/iris_ys_pd.npy', arr = aaa)