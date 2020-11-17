from tensorflow.keras.datasets import cifar10, fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, LSTM
from tensorflow.keras.layers import Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import numpy as np

# 1.데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()        # (28, 28, 1)

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

model.summary()

# 3. 컴파일, 훈련
modelpath = "./model/fashion/{epoch:02d}-{val_loss: 4f}.hdf5"                               # Checkpoint가 저장될 경로 설정

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint               # 조기종료 기능
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30)
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only = True, mode = 'auto')      # Model Checkpoint monitor로 지정한 값이 좋을때마다 저장 

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
hist = model.fit(x_test, y_test, epochs = 100, batch_size = 32, validation_split=0.2, verbose=1, callbacks=[early_stopping, cp])

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

model.save("./save/fashion/fashion_model_2.h5")
model.save_weights("./save/fashion/fashion_weights.h5")

# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test, batch_size = 32)
print("loss : ", loss)
print("accuracy : ", accuracy)

# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize = (10, 6))       # 단위가 무엇인지 찾아보기
plt.subplot(2, 1, 1)                # 2행 1열 중 첫번째
plt.plot(hist.history['loss'], marker = '.', c = 'red', label = 'loss')
plt.plot(hist.history['val_loss'], marker = '.', c = 'blue', label = 'val_loss')
plt.grid()

plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc = 'upper right')

plt.subplot(2, 1, 2)                # 2행 1열 중 첫번째
plt.plot(hist.history['accuracy'], marker = '.', c = 'red')
plt.plot(hist.history['val_accuracy'], marker = '.', c = 'blue')
plt.grid()

plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'val_accuracy'])

plt.show()

