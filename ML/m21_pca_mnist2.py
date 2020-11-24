# 실습 pca로 축소해서 모델을 완성
# 1. 축소한 차원의 가치가 0.95 이상    -> 154
# 2. 축소한 차원의 가치가 1.0 이상     -> 713

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical   
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Dense

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, x_test.shape)  # (60000, 28, 28), (10000, 28, 28)

# pca2 = PCA()                                               # pca 사용하여 원하는 가치 이상 값 찾기
# pca2.fit(x)
# cumsum = np.cumsum(pca2.explained_variance_ratio_)  

# d = np.argmax(cumsum >= 0.95) + 1
# d2 = np.argmax(cumsum >= 1.0) + 1
# print(d, d2)    

x = np.append(x_train, x_test, axis = 0)
x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

pca1 = PCA(n_components=154)
pca1 = pca1.fit_transform(x)

pca2 = PCA(n_components=713)
pca2 = pca2.fit_transform(x)
           
x1_train = pca1[:60000, :]
x1_test = pca1[60000:, :]
x2_train = pca2[:60000, :]
x2_test = pca2[60000:, :]
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)

# 2. 모델 구성
model1 = Sequential()
model1.add(Dense(100, input_shape = (154, )))
model1.add(Dense(70, activation = 'relu'))
model1.add(Dropout(0.2))
model1.add(Dense(30, activation = 'relu'))
model1.add(Dense(10, activation = 'softmax'))

model2 = Sequential()
model2.add(Dense(100, input_shape = (713, )))
model2.add(Dense(70, activation = 'relu'))
model2.add(Dropout(0.2))
model2.add(Dense(30, activation = 'relu'))
model2.add(Dense(10, activation = 'softmax'))

# 3. 컴파일, 훈련
model1.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.fit(x1_train, y_train, epochs=50, batch_size = 32, verbose = 1)

model2.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.fit(x2_train, y_train, epochs=50, batch_size = 32, verbose = 1)

# 4. 평가, 예측
loss1, accuracy1 = model1.evaluate(x1_test, y_test)
loss2, accuracy2 = model2.evaluate(x2_test, y_test)

print("loss1 : ", loss1)
print("accuracy1 : ", accuracy1)

print("loss2 : ", loss2)
print("accuracy2 : ", accuracy2)

# loss1 :  0.16694000363349915
# accuracy1 :  0.9624000191688538
# loss2 :  0.23068654537200928
# accuracy2 :  0.95660001039505