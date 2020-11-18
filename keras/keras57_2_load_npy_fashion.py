from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

# 1.데이터
x_train = np.load('./data/fashion_x_train.npy')
y_train = np.load('./data/fashion_y_train.npy')
x_test = np.load('./data/fashion_x_test.npy')
y_test = np.load('./data/fashion_y_test.npy')

# 데이터 전처리
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] , x_train.shape[2], 1) / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] , x_test.shape[2], 1) / 255.0

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(40, (2,2), input_shape = (28, 28, 1), padding = 'valid'))          # (27, 27, 40)     
model.add(Conv2D(60, (2,2), padding = 'valid'))                                     # (26, 26, 60)
model.add(Conv2D(30, (3,3)))                                                        # (24, 24, 30)
model.add(Conv2D(10, (2,2), strides = 1))                                           # (23, 23, 10)
model.add(MaxPooling2D(pool_size = 2))                                              # (11, 11, 20)
model.add(Flatten())                                                                # (2420, )
model.add(Dense(100, activation = 'relu'))                                          
model.add(Dense(10, activation = 'softmax'))

# 3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model1 = load_model('./model/fashion/03- 0.399306-second.hdf5')
model2 = load_model("./save/fashion/fashion_model_2.h5")
model.load_weights("./save/fashion/fashion_weights.h5")

# 4. 평가, 예측
result1 = model1.evaluate(x_test, y_test, batch_size = 1)
result2 = model2.evaluate(x_test, y_test, batch_size = 1)
result3 = model.evaluate(x_test, y_test, batch_size = 1)

print("CheckPoint")
print("loss : ", result1[0])
print("accuracy : ", result1[1])

print("\nLoad_model")
print("loss : ", result2[0])
print("accuracy : ", result2[1])

print("\nLoad_weight")
print("loss : ", result3[0])
print("accuracy : ", result3[1])

# CheckPoint
# loss :  0.28670573234558105
# accuracy :  0.8985999822616577

# Load_model
# loss :  0.28670573234558105
# accuracy :  0.8985999822616577

# Load_weight
# loss :  0.28670573234558105
# accuracy :  0.8985999822616577