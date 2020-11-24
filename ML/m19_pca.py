import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x = datasets.data       # (442, 10)
y = datasets.target     # (442, )

pca = PCA(n_components=8)                 # 차원을 압축!!
x2d = pca.fit_transform(x)
print(x2d.shape)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR))

