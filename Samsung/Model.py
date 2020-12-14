import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import concatenate, Concatenate, BatchNormalization, Activation
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from keras import regularizers

# 1. 데이터

# 데이터 split
def split_x_2(seq, size):                           
    bbb = []
    for i in range(len(seq) - size + 1):
        for j in range(i, i+1):
            bbb.append(seq[j:j+size, :])
    bbb = np.array(bbb)
    return bbb

samsung_x = np.load("./Samsung/samsung_x.npy", allow_pickle=True).astype('float32')
samsung_y = np.load("./Samsung/samsung_y.npy", allow_pickle=True).astype('float32')
samsung_predict = np.load("./Samsung/samsung_predict.npy", allow_pickle=True).astype('float32')
samsung_y = samsung_y.reshape(659, 1)
samsung_predict = samsung_predict.reshape(1, samsung_predict.shape[0], 1)

bit_computer_x = np.load("./Samsung/bit_computer_x.npy", allow_pickle=True).astype('float32')
bit_computer_predict = np.load("./Samsung/bit_computer_predict.npy", allow_pickle=True).astype('float32')
bit_computer_predict = bit_computer_predict.reshape(1, bit_computer_predict.shape[0], 1)

gold_x = np.load("./Samsung/gold_x.npy", allow_pickle=True).astype('float32')
gold_predict = np.load("./Samsung/gold_predict.npy", allow_pickle=True).astype('float32')
gold_predict = gold_predict.reshape(1, gold_predict.shape[0], 1)

kosdaq_x = np.load("./Samsung/kosdaq_x.npy", allow_pickle=True).astype('float32')
kosdaq_predict = np.load("./Samsung/kosdaq_predict.npy", allow_pickle=True).astype('float32')
kosdaq_predict = kosdaq_predict.reshape(1, kosdaq_predict.shape[0], 1)

# print(samsung_x.shape, samsung_y.shape,samsung_predict.shape)   (659, 3) (659, 1) (3,)
# print(bit_computer_x.shape, bit_computer_predict.shape)         (659, 6) (6,)
# print(gold_x.shape, gold_predict.shape)                         (659, 5) (5,)
# print(kosdaq_x.shape, kosdaq_predict.shape)                     (659, 4) (4,)

# scaler
scaler1 = StandardScaler()
scaler2 = StandardScaler()
scaler3 = StandardScaler()
scaler4 = StandardScaler()

scaler1.fit(samsung_x)                                                   
samsung_x = scaler1.transform(samsung_x)                                       

scaler2.fit(bit_computer_x)  
bit_computer_x = scaler2.transform(bit_computer_x)  

scaler3.fit(gold_x)                                                   
gold_x = scaler3.transform(gold_x)    

scaler4.fit(kosdaq_x)                                                   
kosdaq_x = scaler4.transform(kosdaq_x)  

samsung_x_train, samsung_x_test, samsung_y_train, samsung_y_test = train_test_split(samsung_x, samsung_y, train_size = 0.7, shuffle = True, random_state = 66)
bit_computer_x_train, bit_computer_x_test = train_test_split(bit_computer_x, train_size = 0.7, shuffle = True, random_state = 66)
gold_x_train, gold_x_test = train_test_split(gold_x, train_size = 0.7, shuffle = True, random_state = 66)
kosdaq_x_train, kosdaq_x_test = train_test_split(kosdaq_x, train_size = 0.7, shuffle = True, random_state = 66)

# print(samsung_x_train.shape)                (461, 3)
# print(bit_computer_x_train.shape)           (461, 6)
# print(gold_x_train.shape)                   (461, 5)
# print(kosdaq_x_train.shape)                 (461, 4)
# print(samsung_x_test.shape)                 (198, 3)
# print(bit_computer_x_test.shape)            (198, 6)
# print(gold_x_test.shape)                    (198, 5)
# print(kosdaq_x_test.shape)                  (198, 4)

# samsung_x_train = split_x_2(samsung_x_train, 10)
# samsung_x_test = split_x_2(samsung_x_test, 10)
# samsung_y_train = split_x_2(samsung_y_train, 10)
# samsung_y_test = split_x_2(samsung_y_test, 10)

# bit_computer_x_train = split_x_2(bit_computer_x_train, 10)
# bit_computer_x_test = split_x_2(bit_computer_x_test, 10)

# gold_x_train = split_x_2(gold_x_train, 10)
# gold_x_test = split_x_2(gold_x_test, 10)

# kosdaq_x_train = split_x_2(kosdaq_x_train, 10)
# kosdaq_x_test = split_x_2(kosdaq_x_test, 10)

# # print(samsung_x_train.shape)                (452, 10, 3)
# # print(bit_computer_x_train.shape)           (452, 10, 6)      
# # print(gold_x_train.shape)                   (452, 10, 5)
# # print(kosdaq_x_train.shape)                 (452, 10, 4) 
# # print(samsung_x_test.shape)                 (189, 10, 3)
# # print(bit_computer_x_test.shape)            (189, 10, 6)
# # print(gold_x_test.shape)                    (189, 10, 5)
# # print(kosdaq_x_test.shape)                  (189, 10, 4)
# # print(samsung_y_train.shape)                (452, 10, 1)
# # print(samsung_y_test.shape)                 (189, 10, 1)

samsung_x_train = samsung_x_train.reshape(461, 3, 1)
bit_computer_x_train = bit_computer_x_train.reshape(461, 6, 1)
gold_x_train = gold_x_train.reshape(461, 5, 1)
kosdaq_x_train = kosdaq_x_train.reshape(461, 4, 1)
samsung_x_test = samsung_x_test.reshape(198, 3, 1)
bit_computer_x_test = bit_computer_x_test.reshape(198, 6, 1)
gold_x_test = gold_x_test.reshape(198, 5, 1)
kosdaq_x_test = kosdaq_x_test.reshape(198, 4, 1)

# 2. 모델 구성
# samsung
input1 = Input(shape = (3, 1))
lstm1 = LSTM(1000)(input1)

dense1_1 = Dense(700, kernel_initializer='glorot_uniform', kernel_regularizer = regularizers.l2(0.001))(lstm1)
dense1_1 = BatchNormalization()(dense1_1)
dense1_1 = Activation('relu')(dense1_1)

dense1_2 = Dense(500, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.001))(dense1_1)
dense1_2 = BatchNormalization()(dense1_2)
dense1_2 = Activation('relu')(dense1_2)

dense1_3 = Dense(300, kernel_initializer='glorot_uniform', activation='relu')(dense1_2)
output1 = Dense(200, activation = 'relu')(dense1_3)

# bit_computer
input2 = Input(shape = (6, 1))
lstm2 = LSTM(1000)(input2)

dense2_1 = Dense(600, kernel_initializer='glorot_uniform', kernel_regularizer = regularizers.l2(0.001))(lstm2)
dense2_1 = BatchNormalization()(dense2_1)
dense2_1 = Activation('relu')(dense2_1)

dense2_2 = Dense(400, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.001))(dense2_1)
dense2_2 = BatchNormalization()(dense2_2)
dense2_2 = Activation('relu')(dense2_2)

dense2_3 = Dense(200, kernel_initializer='glorot_uniform', activation='relu')(dense2_2)
output2 = Dense(200, activation = 'relu')(dense2_3)

# gold
input3 = Input(shape = (5, 1))
lstm3 = LSTM(1000)(input3)

dense3_1 = Dense(500, kernel_initializer='glorot_uniform', kernel_regularizer = regularizers.l2(0.001))(lstm3)
dense3_1 = BatchNormalization()(dense3_1)
dense3_1 = Activation('relu')(dense3_1)

dense3_2 = Dense(400, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.001))(dense3_1)
dense3_2 = BatchNormalization()(dense3_2)
dense3_2 = Activation('relu')(dense3_2)

dense3_3 = Dense(300, kernel_initializer='glorot_uniform', activation='relu')(dense3_2)
output3 = Dense(200, activation = 'relu')(dense3_3)

# kosdaq
input4 = Input(shape = (4, 1))
lstm4 = LSTM(1000)(input4)

dense4_1 = Dense(200, kernel_initializer='glorot_uniform', kernel_regularizer = regularizers.l2(0.001))(lstm4)
dense4_1 = BatchNormalization()(dense4_1)
dense4_1 = Activation('relu')(dense4_1)

dense4_2 = Dense(300, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.001))(dense4_1)
dense4_2 = BatchNormalization()(dense4_2)
dense4_2 = Activation('relu')(dense4_2)

dense4_3 = Dense(100, kernel_initializer='glorot_uniform', activation='relu')(dense4_2)
output4 = Dense(200, activation = 'relu')(dense4_3)

merge = Concatenate()([output1, output2, output3, output4])

middle1 = Dense(1000, activation = 'relu')(merge)
middle1 = Dense(500, activation = 'relu')(middle1)

output = Dense(100, activation = 'relu')(middle1)
output = Dense(30, activation = 'relu')(output)
output = Dense(1)(output)

model = Model(inputs = [input1, input2, input3, input4], outputs = output)
model.summary()

# 3.컴파일, 훈련
modelpath = "./model/homework3/{epoch:02d}-{val_loss: 4f}-05.hdf5"
model = load_model('./model/homework3/186- 103177800.000000-05.hdf5')
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# early_stopping = EarlyStopping(monitor = 'val_loss', patience = 300, mode = 'min')
# cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only = True, mode = 'auto')

# model.compile(loss = 'mse', optimizer = 'nadam', metrics = ['mae'])
# model.fit([samsung_x_train, bit_computer_x_train, gold_x_train, kosdaq_x_train], samsung_y_train, epochs = 1000, batch_size = 100, validation_split = 0.25, verbose = 1, callbacks = [early_stopping, cp])

# 4.평가, 예측
loss, mse = model.evaluate([samsung_x_test, bit_computer_x_test, gold_x_test, kosdaq_x_test], samsung_y_test)
print("loss : ", loss)
print("mse : ", mse)

predict = model.predict([samsung_predict, bit_computer_predict, gold_predict, kosdaq_predict])
print("2020년 11월 23일 삼성전자 시가 : ", predict)

# 7/7 [==============================] - 0s 5ms/step - loss: 663556160.0000 - mean_absolute_error: 8186.3877
# loss :  663556160.0
# mse :  8186.3876953125
# predict :  [[69507.914]]