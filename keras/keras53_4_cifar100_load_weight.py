from tensorflow.keras.datasets import cifar10, fashion_mnist, cifar100
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar100.load_data()        

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical               # 분류모델에서는 onehotencoding 필수
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(50000, 32 , 32 , 3).astype("float32") / 255.        # .astype("type") => 형 변환
x_test = x_test.reshape(10000, 32 , 32 , 3).astype("float32") / 255.

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(100, (2,2), activation = 'relu', input_shape = (32, 32, 3), padding = 'valid'))           # (31, 31, 20)     
model.add(Conv2D(600, (2,2), activation = 'relu', padding = 'valid'))                                     # (30, 30, 40)
model.add(Conv2D(300, (3,3), activation = 'relu'))                                                        # (28, 28, 10)
model.add(Conv2D(100, (2,2), activation = 'relu', strides = 1))                                           # (27, 27, 20)
model.add(MaxPooling2D(pool_size = 2))                                              # (13, 13, 20)
model.add(Flatten())                                                                # (3380, )
model.add(Dense(1000, activation = 'relu'))                                          
model.add(Dense(500, activation = 'relu'))                                          
model.add(Dense(100, activation = 'softmax')) 

# 3. 컴파일, 훈련
from tensorflow.keras.models import load_model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model1 = load_model("./model/cifar100/06- 3.554917.hdf5")
model2 = load_model("./save/cifar100/cifar100_model_2.h5")
model.load_weights("./save/cifar100/cifar100_weights.h5")

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
# loss :  2.9012770652770996
# accuracy :  0.2840000092983246

# Load_model
# loss :  2.9012770652770996
# accuracy :  0.2840000092983246

# Load_weight
# loss :  2.9012770652770996
# accuracy :  0.2840000092983246