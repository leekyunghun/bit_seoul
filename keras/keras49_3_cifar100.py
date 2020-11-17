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

model.summary()

# 3. 컴파일, 훈련
modelpath = "./model/cifar100/{epoch:02d}-{val_loss: 4f}.hdf5"                               # Checkpoint가 저장될 경로 설정

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint               # 조기종료 기능
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30)
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only = True, mode = 'auto')      # Model Checkpoint monitor로 지정한 값이 좋을때마다 저장 

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
hist = model.fit(x_test, y_test, epochs = 100, batch_size=32, validation_split=0.2, verbose=1, callbacks=[early_stopping, cp])

loss = hist.history['loss']
val_loss = hist.history['val_loss']
acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']

model.save("./save/cifar100/cifar100_model_2.h5")
model.save_weights("./save/cifar100/cifar100_weights.h5")

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