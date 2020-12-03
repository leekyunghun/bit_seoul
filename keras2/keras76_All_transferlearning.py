from tensorflow.keras.applications import VGG16, VGG19, Xception, ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.layers import Dense, Flatten, BatchNormalization, Dropout, Activation
from tensorflow.keras.models import Sequential

#VGG16
vgg16 = VGG16()
# model = VGG16(weights='imagenet')       # 이대로 사용하면 input_shape가 고정되어있다. 
# vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))     # include_top을 False로 두면 input_shape 변경 가능
vgg16.trainable = True
vgg16.summary()
print("동결하기 전 훈련되는 가중치의 수 : ", len(vgg16.trainable_weights), "\n")

# VGG19
vgg19 = VGG19()
vgg19.trainable = True
vgg19.summary()
print("동결하기 전 훈련되는 가중치의 수 : ", len(vgg19.trainable_weights), "\n")

# Xception
xception = Xception()
xception.trainable = True
xception.summary()
print("동결하기 전 훈련되는 가중치의 수 : ", len(xception.trainable_weights), "\n")

# ResNet50
resnet50 = ResNet50()
resnet50.trainable = True
resnet50.summary()
print("동결하기 전 훈련되는 가중치의 수 : ", len(resnet50.trainable_weights), "\n")

# ResNet50V2
resnet50v2 = ResNet50V2()
resnet50v2.trainable = True
resnet50v2.summary()
print("동결하기 전 훈련되는 가중치의 수 : ", len(resnet50v2.trainable_weights), "\n")

# ResNet101
resnet101 = ResNet101()
resnet101.trainable = True
resnet101.summary()
print("동결하기 전 훈련되는 가중치의 수 : ", len(resnet101.trainable_weights), "\n")

# ResNet101V2
resnet101v2 = ResNet101V2()
resnet101v2.trainable = True
resnet101v2.summary()
print("동결하기 전 훈련되는 가중치의 수 : ", len(resnet101v2.trainable_weights), "\n")

# ResNet152
resnet152 = ResNet152()
resnet152.trainable = True
resnet152.summary()
print("동결하기 전 훈련되는 가중치의 수 : ", len(resnet152.trainable_weights), "\n")

# ResNet152V2
resnet152v2 = ResNet152V2()
resnet152v2.trainable = True
resnet152v2.summary()
print("동결하기 전 훈련되는 가중치의 수 : ", len(resnet152v2.trainable_weights), "\n")

# InceptionV3
inceptionv3 = InceptionV3()
inceptionv3.trainable = True
inceptionv3.summary()
print("동결하기 전 훈련되는 가중치의 수 : ", len(inceptionv3.trainable_weights), "\n")

# InceptionResNetV2
inceptionResnetv2 = InceptionResNetV2()
inceptionResnetv2.trainable = True
inceptionResnetv2.summary()
print("동결하기 전 훈련되는 가중치의 수 : ", len(inceptionResnetv2.trainable_weights), "\n")

# MobileNet
mobilenet = MobileNet()
mobilenet.trainable = True
mobilenet.summary()
print("동결하기 전 훈련되는 가중치의 수 : ", len(mobilenet.trainable_weights), "\n")

# MobileNetV2
mobilenetv2 = MobileNetV2()
mobilenetv2.trainable = True
mobilenetv2.summary()
print("동결하기 전 훈련되는 가중치의 수 : ", len(mobilenetv2.trainable_weights), "\n")

# DenseNet121
densenet121 = DenseNet121()
densenet121.trainable = True
densenet121.summary()
print("동결하기 전 훈련되는 가중치의 수 : ", len(densenet121.trainable_weights), "\n")

# DenseNet169
densenet169 = DenseNet169()
densenet169.trainable = True
densenet169.summary()
print("동결하기 전 훈련되는 가중치의 수 : ", len(densenet169.trainable_weights), "\n")

# DenseNet201
densenet201 = DenseNet201()
densenet201.trainable = True
densenet201.summary()
print("동결하기 전 훈련되는 가중치의 수 : ", len(densenet201.trainable_weights), "\n")

# NASNetLarge
nasnetlarge = NASNetLarge()
nasnetlarge.trainable = True
nasnetlarge.summary()
print("동결하기 전 훈련되는 가중치의 수 : ", len(nasnetlarge.trainable_weights), "\n")

# NASNetMobile
nasnetmobile = NASNetMobile()
nasnetmobile.trainable = True
nasnetmobile.summary()
print("동결하기 전 훈련되는 가중치의 수 : ", len(nasnetmobile.trainable_weights), "\n")

# model = Sequential()
# model.add(vgg16)
# model.add(Flatten())
# model.add(Dense(256))
# # model.add(BatchNormalization())       # 가중치에 연산을 해주어서 weight값들이 생긴다.
# # model.add(Dropout(0.2))               # 안해줌
# model.add(Activation('relu'))           # 안해줌
# model.add(Dense(256))
# model.add(Dense(10, activation = 'softmax'))

# model.summary()

# print("동결하기 전 훈련되는 가중치의 수 : ", len(model.trainable_weights))

# import pandas as pd
# pd.set_option("max_colwidth", -1)
# layers = [(layer, layer.name, layer.trainable) for layer in model.layers]
# aaa = pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])
# print(aaa.loc[:])

# VGG16
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# 동결하기 전 훈련되는 가중치의 수 :  32

# VGG19
# Total params: 143,667,240
# Trainable params: 143,667,240
# Non-trainable params: 0
# 동결하기 전 훈련되는 가중치의 수 :  38

# Xception
# Total params: 22,910,480
# Trainable params: 22,855,952
# Non-trainable params: 54,528
# 동결하기 전 훈련되는 가중치의 수 :  156

# ResNet50
# Total params: 25,636,712
# Trainable params: 25,583,592
# Non-trainable params: 53,120
# 동결하기 전 훈련되는 가중치의 수 :  214

# ResNet50V2
# Total params: 25,613,800
# Trainable params: 25,568,360
# Non-trainable params: 45,440
# 동결하기 전 훈련되는 가중치의 수 :  174

# ResNet101
# Total params: 44,707,176
# Trainable params: 44,601,832
# Non-trainable params: 105,344
# 동결하기 전 훈련되는 가중치의 수 :  418

# ResNet101V2
# Total params: 44,675,560
# Trainable params: 44,577,896
# Non-trainable params: 97,664
# 동결하기 전 훈련되는 가중치의 수 :  344

# ResNet152
# Total params: 60,419,944
# Trainable params: 60,268,520
# Non-trainable params: 151,424
# 동결하기 전 훈련되는 가중치의 수 :  622

# # ResNet152V2
# Total params: 60,380,648
# Trainable params: 60,236,904
# Non-trainable params: 143,744
# 동결하기 전 훈련되는 가중치의 수 :  514

# # InceptionV3
# Total params: 23,851,784
# Trainable params: 23,817,352
# Non-trainable params: 34,432
# 동결하기 전 훈련되는 가중치의 수 :  190

# # InceptionResNetV2
# Total params: 55,873,736
# Trainable params: 55,813,192
# Non-trainable params: 60,544
# 동결하기 전 훈련되는 가중치의 수 :  490

# # MobileNet
# Total params: 4,253,864
# Trainable params: 4,231,976
# Non-trainable params: 21,888
# 동결하기 전 훈련되는 가중치의 수 :  83

# # MobileNetV2
# Total params: 3,538,984
# Trainable params: 3,504,872
# Non-trainable params: 34,112
# 동결하기 전 훈련되는 가중치의 수 :  158 

# # DenseNet121
# Total params: 8,062,504
# Trainable params: 7,978,856
# Non-trainable params: 83,648
# 동결하기 전 훈련되는 가중치의 수 :  364

# # DenseNet169
# Total params: 14,307,880
# Trainable params: 14,149,480
# Non-trainable params: 158,400
# 동결하기 전 훈련되는 가중치의 수 :  508 

# # DenseNet201
# Total params: 20,242,984
# Trainable params: 20,013,928
# Non-trainable params: 229,056
# 동결하기 전 훈련되는 가중치의 수 :  604

# # NASNetLarge
# Total params: 88,949,818
# Trainable params: 88,753,150
# Non-trainable params: 196,668
# 동결하기 전 훈련되는 가중치의 수 :  1018 

# # NASNetMobile
# Total params: 5,326,716
# Trainable params: 5,289,978
# Non-trainable params: 36,738
# 동결하기 전 훈련되는 가중치의 수 :  742
