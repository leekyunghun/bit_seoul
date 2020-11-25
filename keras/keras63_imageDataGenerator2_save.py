from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten

np.random.seed(33)

# 이미지 생성 옵션 정하기
train_datagen = ImageDataGenerator(rescale = 1/255, horizontal_flip = True, vertical_flip = True, 
                                   width_shift_range = 0.1, height_shift_range = 0.1, rotation_range = 5, 
                                   zoom_range = 1.2, shear_range = 0.7, fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale = 1/255)

# 1. 데이터
# flow 또는 flow_from_directory
# 실제 데이터가 있는 곳을 알려주고, 이미지를 불러오는 작업
xy_train = train_datagen.flow_from_directory('./data/data1/train', target_size = (150, 150), 
                                                    batch_size = 5, class_mode = 'binary')# , save_to_dir = './data/data1_2/train')
xy_test = test_datagen.flow_from_directory('./data/data1/test', target_size = (150, 150),
                                                 batch_size = 5, class_mode = 'binary')

# print("========================================================================")
# print(type(xy_train))
# print(xy_train[0])
# print(xy_train[0].shape)      Error
# print(xy_train[0][0].shape)     # (5, 150, 150, 3)      xy_train[a][b] a: 전체 데이터 갯수를 batch_size로 나누었을때의 index, b:x_data, y_data = ([0], [1])
# print(xy_train[0][1].shape)     # (5, )
# print(len(xy_train))

# print("========================================================================")
# print(xy_train[0][0][0])

# np.save('./data/keras63_train_x.npy', arr = xy_train[0][0])
# np.save('./data/keras63_train_y.npy', arr = xy_train[0][1])
# np.save('./data/keras63_test_x.npy', arr = xy_test[0][0])
# np.save('./data/keras63_test_y.npy', arr = xy_test[0][1])

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(64, (2, 2), activation = 'relu', strides = (2,2), input_shape = (150, 150, 3))) # 150
model.add(Conv2D(128, (2, 2), activation = 'relu', strides = (2,2)))  # 74
model.add(Dropout(0.3))
model.add(Conv2D(256, (2, 2), activation = 'relu', strides = (2,2)))  # 36
model.add(MaxPooling2D())                                            # 18
model.add(Conv2D(64, (2, 2), activation = 'relu', strides = (2,2)))  # 8
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

# 3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
hist = model.fit_generator(xy_train, steps_per_epoch = 32, epochs = 100, validation_data = xy_test, validation_steps = 24)

# 4. 평가, 예측
loss, accuracy = model.evaluate(xy_test)
print("loss : ", loss)
print("accuracy : ", accuracy)

# 그래프
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])

plt.title("loss & accuracy")
plt.ylabel("loss, accuracy")
plt.xlabel("epoch")

plt.legend(["train loss", "val loss", "train accuracy", "val accuracy"])  # 그래프가 어떤거에 해당하는지 알려줌
plt.show() 