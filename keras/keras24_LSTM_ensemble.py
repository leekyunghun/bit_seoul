from numpy import array

# 1.데이터
x1 = array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
           [5,6,7],[6,7,8],[7,8,9],[8,9,10],
           [9,10,11],[10,11,12],
           [20,30,40],[30,40,50],[40,50,60]])               # (13,3)        # 시계열 데이터는 데이터중 예측할 구간과 가까울수록 학습에 영향을 많이 줌
           
x2 = array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
            [50,60,70],[60,70,80],[70,80,90],[80,90,100],
            [90,100,110],[100,110,120],
            [2,3,4],[3,4,5],[4,5,6]])                       # (13,3)

y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])               # (13, )

x1_predict = array([55,65,75])                              # (3, )
x2_predict = array([65,75,85])                              # (3, )

x1 = x1.reshape(13,3,1)
x2 = x2.reshape(13,3,1)

x1_predict = x1_predict.reshape(1,3,1)
x2_predict = x2_predict.reshape(1,3,1)

# 2.모델 구성
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input

# 모델 1
input1 = Input(shape=(3,1))
lstm = LSTM(50, activation = 'relu')(input1) 
dense1 = Dense(256, activation = 'relu')(lstm)
dense2 = Dense(128, activation = 'relu')(dense1)
dense3 = Dense(64, activation = 'relu')(dense2)
output1 = Dense(8)(dense3)

# 모델 2
input2 = Input(shape=(3,1))
lstm2 = LSTM(40, activation = 'relu')(input2) 
dense2_1 = Dense(128, activation = 'relu')(lstm2)
dense2_2 = Dense(64, activation = 'relu')(dense2_1)
dense2_3 = Dense(16, activation = 'relu')(dense2_2)
dense2_4 = Dense(4, activation = 'relu')(dense2_3)
output2 = Dense(8)(dense2_4)

# 모델 병합, concatenate
from tensorflow.keras.layers import concatenate, Concatenate
merge = Concatenate()([output1, output2])

middle1 = Dense(64, activation = 'relu')(merge)
middle1 = Dense(80, activation = 'relu')(middle1)
middle1 = Dense(40, activation = 'relu')(middle1)

######### output 모델 구성 (분기)
output = Dense(20, activation = 'relu')(middle1)
output = Dense(10, activation = 'relu')(output)
output = Dense(1)(output)

model = Model(inputs = [input1, input2], outputs = output)
model.summary()

# 3.컴파일, 훈련
# from tensorflow.keras.callbacks import EarlyStopping        # 조기종료 기능
# early_stopping = EarlyStopping(monitor = 'loss', patience = 10, mode = 'min')

model.compile(loss = 'mse', optimizer = 'adam')
model.fit([x1, x2], y, epochs = 300, batch_size = 1, verbose = 1)

# 4.평가, 예측
predict = model.predict([x1_predict, x2_predict])
print("predict : ", predict)

