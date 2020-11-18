from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
print(iris)
print(type(iris))       # <class 'sklearn.utils.Bunch'>

x_data = iris.data
y_data = iris.target

print(type(x_data))
print(type(y_data))

np.save("./data/iris_x_npy", arr = x_data)          # 경로설정 후 원하는 numpy 배열을 저장
np.save("./data/iris_y_npy", arr = y_data)

