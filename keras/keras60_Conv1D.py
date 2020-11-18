import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, LSTM, Dense, Flatten, MaxPooling1D, Dropout

# 1.데이터
a = np.array((range(1,101)))
size = 5

def split_x(seq, size):
    aaa = []
    for i in range(len(seq) - size + 1):
        subset = seq[i : (i+size)]
        aaa.append(subset)

    return np.array(aaa)

dataset = split_x(a, size)      # (96, 5)

x = dataset[:, :-1]
y = dataset[:, -1]

x_pred = np.array([97,98,99,100])

# # 데이터 전처리
# from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler  # 데이터 전처리 기능 (최대값, 최소값 이용), 총 4가지
# scaler = MinMaxScaler()                                                                                 
# # scaler = MaxAbsScaler()
# # scaler = RobustScaler()
# # scaler = StandardScaler()

# scaler.fit(x)                                                   
# x = scaler.transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.8)

print(x_train.shape, x_test.shape)

x_train = x_train.reshape(x_train.shape[0], 4, 1) / 100.
x_test = x_test.reshape(x_test.shape[0], 4, 1) / 100.
x_pred = x_pred.reshape(1, 4, 1) / 100.

# 2.모델 구성
model = Sequential()
model.add(Conv1D(64, 2, input_shape = (4, 1)))
model.add(Dropout(0.2))
model.add(Conv1D(30, 2, activation = 'relu'))
model.add(Flatten())
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1))

model.summary()

# 3.컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'min') 

model.compile(loss = 'mse', optimizer = 'adam', metrics=['mae'])
model.fit(x_train, y_train, epochs = 200, batch_size = 10, verbose = 1, validation_split = 0.2, callbacks = [early_stopping])

# 4.평가, 예측
loss, mae = model.evaluate(x_test, y_test, batch_size = 1)
print("loss : ", loss)
print("mae : ", mae)

predict = model.predict(x_pred)
print("predict : ", predict)

# Epoch 109/200
# 6/6 [==============================] - 0s 4ms/step - loss: 4.7619 - mae: 1.6007 - val_loss: 0.0633 - val_mae: 0.2485
# 20/20 [==============================] - 0s 2ms/step - loss: 0.0624 - mae: 0.2471
# loss :  0.06240389868617058
# mae :  0.24708525836467743
# predict : [[101.17455]]