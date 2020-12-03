# 배운걸 써먹어서 cifar에 사용
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.applications import VGG16, VGG19, Xception, ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
import numpy as np

# 1. 데이터 
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

from tensorflow.keras.utils import to_categorical              
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(50000, 32, 32, 3).astype("float32") / 255.     
x_test = x_test.reshape(10000, 32, 32, 3).astype("float32") / 255.

# 2. 모델 구성
resnet101 = ResNet101(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
resnet101.trainable = False
model = Sequential()
model.add(Flatten())                                                             
model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('relu'))                                        
model.add(Dense(10, activation = 'softmax'))

# 3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping  
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30, mode = 'min') 

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 100, batch_size = 100, validation_split = 0.2, verbose = 1, callbacks = [early_stopping])

# 4. 평가, 예측
loss, accuracy = model.evaluate(x_test, y_test)
print("loss : ", loss)
print("accuracy : ", accuracy)

# 313/313 [==============================] - 0s 2ms/step - loss: 2.3483 - accuracy: 0.4100
# loss :  2.3483383655548096
# accuracy :  0.4099999964237213