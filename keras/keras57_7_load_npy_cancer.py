import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, LSTM, Input, MaxPooling2D, Flatten
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# 1.데이터
x = np.load('./data/cancer_x.npy')
y = np.load('./data/cancer_y.npy')

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

# 2.모델 구성
model = Sequential()
model.add(Dense(64, activation = 'relu', input_shape = (30, )))
model.add(Dropout(0.2))
model.add(Dense(30))
model.add(Dense(10, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))

# 3.컴파일, 훈련
from tensorflow.keras.models import load_model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy']) 

model1 = load_model("./model/cancer/74- 0.008421.hdf5")
model2 = load_model("./save/cancer/cancer_model_2.h5")
model.load_weights("./save/cancer/cancer_weights.h5")

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
# loss :  0.04791942983865738
# accuracy :  0.9707602262496948

# Load_model
# loss :  0.04791942983865738
# accuracy :  0.9707602262496948

# Load_weight
# loss :  0.04791942983865738
# accuracy :  0.9707602262496948