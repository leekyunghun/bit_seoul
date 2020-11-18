# 다중분류
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LSTM, Input, MaxPooling2D, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 1.데이터
x = np.load('./data/iris_x_npy.npy')
y = np.load('./data/iris_y_npy.npy')

# 데이터 전처리
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler  # 데이터 전처리 기능 (최대값, 최소값 이용), 총 4가지
scaler = MinMaxScaler()                                                                                 
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = StandardScaler()

scaler.fit(x)                                                   
x = scaler.transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7)     # sklearn 사용할때 데이터 나누기 아주좋음

x_train = x_train.reshape(x_train.shape[0], 2, 2, 1)
x_test = x_test.reshape(x_test.shape[0], 2, 2, 1)

from tensorflow.keras.utils import to_categorical               # 분류모델에서는 onehotencoding 필수
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 2.모델 구성
model = Sequential()
model.add(Conv2D(20, (2, 2), input_shape = (2, 2, 1)))
model.add(Dropout(0.2))
model.add(Conv2D(30, (1, 1), input_shape = (2, 2, 1)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3, activation = 'softmax'))

# 3.컴파일, 훈련
from tensorflow.keras.models import load_model
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model1 = load_model("./model/iris/99- 0.078107-second.hdf5")
model2 = load_model("./save/iris/iris_model_2.h5")
model.load_weights("./save/iris/iris_weights.h5")

# 4.예측, 평가
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
# loss :  0.1106095090508461
# accuracy :  0.9555555582046509

# Load_model
# loss :  0.1106095090508461
# accuracy :  0.9555555582046509

# Load_weight
# loss :  0.1106095090508461
# accuracy :  0.9555555582046509

