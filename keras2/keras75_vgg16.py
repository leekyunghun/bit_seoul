from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Sequential
# model = VGG16()
# model = VGG16(weights='imagenet')       # 이대로 사용하면 input_shape가 고정되어있다. 
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))     # include_top을 False로 두면 input_shape 변경 가능
vgg16.trainable = False

vgg16.summary()

print("동결하기 전 훈련되는 가중치의 수 : ", len(vgg16.trainable_weights))
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256))
# model.add(BatchNormalization())       # 가중치에 연산을 해주어서 weight값들이 생긴다.
# model.add(Dropout(0.2))               # 안해줌
model.add(Activation('relu'))           # 안해줌
model.add(Dense(256))
model.add(Dense(10, activation = 'softmax'))

model.summary()

print(len(model.trainable_weights))

import pandas as pd
pd.set_option("max_colwidth", -1)
layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
aaa = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
print(aaa.loc[:])