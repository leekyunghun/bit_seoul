from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten, BatchNormalization
import PIL.Image as pilimg
from keras import regularizers

np.random.seed(33)

# 이미지 생성 옵션 정하기
train_datagen = ImageDataGenerator(rescale = 1/255., horizontal_flip = True, vertical_flip=True,
                                   width_shift_range = 0.2, height_shift_range = 0.2, 
                                   zoom_range = 0.2, fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale = 1/255.)
pred_datagen = ImageDataGenerator(rescale = 1/255.)

# 1. 데이터
xy_train = train_datagen.flow_from_directory('D:/intel image classification/train', target_size = (150, 150), batch_size = 20, class_mode = 'sparse') 
                                                    
xy_test = test_datagen.flow_from_directory('D:/intel image classification/test', target_size = (150, 150), batch_size = 20, class_mode = 'sparse')

predict = pred_datagen.flow_from_directory('D:/intel image classification/pred', target_size = (150, 150), batch_size = 100, class_mode = None)

# np.save('./project/project_train_x.npy', arr = xy_train[0][0])
# np.save('./project/project_train_y.npy', arr = xy_train[0][1])
# np.save('./project/project_test_x.npy', arr = xy_test[0][0])
# np.save('./project/project_test_y.npy', arr = xy_test[0][1])

# 2. 모델 구성
model = Sequential()
model.add(Conv2D(512, (2,2), activation='relu', kernel_regularizer= regularizers.l1(0.001), input_shape=(150,150,3)))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(MaxPooling2D(5,5))

model.add(Conv2D(256, (2,2), kernel_regularizer= regularizers.l1(0.001), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(512, (2,2), kernel_regularizer= regularizers.l1(0.001), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(128, (2,2), kernel_regularizer= regularizers.l1(0.001), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(256, (2,2), kernel_regularizer= regularizers.l1(0.001), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(MaxPooling2D(5,5))

model.add(Flatten())

model.add(Dense(256, kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())

model.add(Dense(128, kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(64, kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())

model.add(Dense(32, kernel_initializer='he_normal', activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(7,activation='softmax'))

model.summary()

# 3. 컴파일, 훈련
# model = load_model('./project/CheckPoint/CheckPoint-12- 1.497516.hdf5') 
modelpath = "./project/CheckPoint/CheckPoint-{epoch:02d}-{val_loss: 4f}.hdf5"  

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint          
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 30)
cp = ModelCheckpoint(filepath = modelpath, monitor = 'val_accuracy', save_best_only = True, mode = 'auto')

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
hist = model.fit_generator(xy_train, steps_per_epoch = 800, epochs = 50, validation_data = xy_test, callbacks=[early_stopping, cp])

# 4. 평가, 예측
loss, accuracy = model.evaluate(xy_test)
print("loss : ", loss)
print("accuracy : ", accuracy)

y_pred = model.predict_generator(predict, steps=1, verbose = 1)
print(y_pred.shape)
y_pred = np.round(y_pred)
y_pred = np.argmax(y_pred, axis=1).reshape(-1,1)

# 0 ~ 6 사이로 이루어진 라벨들을 문자열 이름으로 출력하기 위해 바꿔주는 함수
a = ['buildings', 'forest', 'glacier', 'human', 'mountain', 'sea', 'street']

def printIndex(array, i):
    if array[i][0] == 0:
        return a[0]
    elif array[i][0] == 1:
        return a[1]
    elif array[i][0] == 2:
        return a[2]
    elif array[i][0] == 3:
        return a[3]
    elif array[i][0] == 4:
        return a[4]
    elif array[i][0] == 5:
        return a[5]
    elif array[i][0] == 6:
        return a[6]

# Matplotlib을 활용한 예측값과 실제값 시각화
fig = plt.figure()
rows = 5
cols = 10

for i in range(50):
    ax = fig.add_subplot(rows, cols, i+1)
    ax.imshow(predict[0][i])
    label = printIndex(y_pred, i)
    ax.set_xlabel(label)
    ax.set_xticks([]), ax.set_yticks([])
plt.show()
