# 실습 pca로 축소해서 모델을 완성
# 1. 축소한 차원의 가치가 0.95 이상    -> 202
# 2. 축소한 차원의 가치가 1.0 이상     -> 3072

import numpy as np
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Dropout, Dense

# 1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x = np.append(x_train, x_test, axis = 0)
x = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

pca1 = PCA(n_components=202)
pca1 = pca1.fit_transform(x)

pca2 = PCA(n_components=3072)
pca2 = pca2.fit_transform(x)

pca1_x_train = pca1[:50000, :]
pca1_x_test = pca1[50000:, :]

pca2_x_train = pca2[:50000, :]
pca2_x_test = pca2[50000:, :]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2. 모델 구성
model1 = Sequential()
model1.add(Dense(3000, activation = 'relu', input_shape = (202, )))
model1.add(Dense(2000, activation = 'relu'))
model1.add(Dropout(0.3))
model1.add(Dense(1000, activation = 'relu'))
model1.add(Dense(500, activation = 'relu'))
model1.add(Dense(100, activation = 'softmax'))

model2 = Sequential()
model2.add(Dense(3000, activation = 'relu', input_shape = (3072, )))
model2.add(Dense(2000, activation = 'relu'))
model1.add(Dropout(0.3))
model2.add(Dense(1000, activation = 'relu'))
model2.add(Dense(500, activation = 'relu'))
model2.add(Dense(100, activation = 'softmax'))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'min') 

model1.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model1.fit(pca1_x_train, y_train, epochs = 50, batch_size = 100, validation_split = 0.2, verbose = 1, callbacks = [early_stopping])

model2.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model2.fit(pca2_x_train, y_train, epochs = 50, batch_size = 100, validation_split = 0.2, verbose = 1, callbacks = [early_stopping])

# 4. 평가, 예측
loss1, accuracy1 = model1.evaluate(pca1_x_test, y_test, batch_size = 1)
loss2, accuracy2 = model2.evaluate(pca2_x_test, y_test, batch_size = 1)

print("loss1 : ", loss1)
print("accuracy1 : ", accuracy1)
print("loss2 : ", loss2)
print("accuracy2 : ", accuracy2)

# loss1 :  1.1920930376163597e-07
# accuracy1 :  0.009999999776482582
# loss2 :  8.739947319030762
# accuracy2 :  0.1589999943971634