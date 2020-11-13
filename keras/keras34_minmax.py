from numpy import array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM

# 1.데이터
x = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
          [5,6,7],[6,7,8],[7,8,9],[8,9,10],
          [9,10,11],[10,11,12],
          [2000,3000,4000],[3000,4000,5000],[4000,5000,6000],   # 데이터 전처리를 할때 입력 데이터만 전처리를 한다. (Y 데이터는 전처리 X)
          [100,200,300]])                                        # (14, 3)

y = array([4,5,6,7,8,9,10,11,12,13,5000,6000,7000,400])

x_predict = array([55,65,75])                                   # (3, )
x_predict2 = array([6600, 6700, 6800])                          # x의 최대값인 6000보다 큰값이지만 그냥 x의 최대값으로 scaler.transform(x_predict2) 사용 
                                                                # 훈련한 x의 값들보다 큰 값들이여도 x의 최대값으로 연산은 가능하다.
x_predict = x_predict.reshape(1, 3)
x_predict2 = x_predict2.reshape(1, 3)

from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler       # 데이터 전처리 기능 (최대값, 최소값 이용), 총 4가지
scaler = MinMaxScaler()                                                                                 
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
# scaler = StandardScaler()

scaler.fit(x)                                                   # x의 max, min값이 scaler에 저장
x = scaler.transform(x)                                         # x의 max, min값으로 전처리
x_predict = scaler.transform(x_predict)                         # fit(x)로 저장되어있는 max, min값으로 x_predict값들을 전처리
x_predict2 = scaler.transform(x_predict2)

x = x.reshape(x.shape[0], x.shape[1], 1)
x_predict = x_predict.reshape(x_predict.shape[0], x_predict.shape[1], 1)
x_predict2 = x_predict2.reshape(x_predict2.shape[0], x_predict2.shape[1], 1)

# 2.모델 구성
input = Input(shape=(3,1))
lstm = LSTM(100, activation = 'relu')(input) 
dense1 = Dense(256)(lstm)
dense2 = Dense(100, activation = 'relu')(dense1)
dense3 = Dense(50)(dense2)
dense4 = Dense(10, activation = 'relu')(dense3)
output = Dense(1)(dense4)

model = Model(inputs = input, outputs = output)

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor = 'loss', patience = 20, mode = 'min')
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])
history = model.fit(x, y, epochs = 300, batch_size = 1, validation_split = 0.25, verbose = 1)

# 4.평가, 예측
loss, mae = model.evaluate(x, y, batch_size = 1)
print("loss : ", loss)
print("mae : ", mae)

y_predict = model.predict(x_predict)
print("y_predict : ", y_predict)

y_predict2 = model.predict(x_predict2)
print("y_predict2 : ", y_predict2)
