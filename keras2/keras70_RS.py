import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint    

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()        # mnist 예제의 좋은기능 train set 과 test set을 나눠주는 기능이 있음

print(x_train.shape, x_test.shape)                              # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)                              # (60000, )       (10000, )

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical               # 분류모델에서는 onehotencoding 필수
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)
print(y_test.shape)

x_train = x_train.reshape(60000, 28 * 28).astype("float32") / 255.        # .astype("type") => 형 변환
x_test = x_test.reshape(10000, 28 * 28).astype("float32") / 255.

# 2. 모델

from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
# keras model을 함수형태로 만들었음
def build_model(drop = 0.5, optimizer = Adam, learning_rate = 0.001, node = 100):
    inputs = Input(shape = (28*28, ), name = 'input')
    x = Dense(256, activation='relu', name = 'hidden1')(inputs)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name = 'hidden2')(x)
    x = Dropout(drop)(x)
    x = Dense(64, activation='relu', name = 'hidden3')(x)
    x = Dropout(drop)(x)
    outputs = Dense(10, activation='softmax', name = 'outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)
    # early_stopping = EarlyStopping(monitor = 'val_loss', patience = stop)
    model.compile(optimizer = optimizer(learning_rate = learning_rate), metrics = ['accuracy'], loss = 'categorical_crossentropy')
    print(optimizer)
    print(learning_rate)
    print(node)
    return model

# GridSearch parameter 함수
def create_hyperparameter():                # 파라미터를 지정해줄때의 이름과 모델 함수에 들어있는 파라미터 변수 이름을 동일시켜야함
    batches = [30]
    # learning_rate = [0.1, 0.05, 0.001]
    learning_rate = [0.001]
    # optimizers = [Adam, RMSprop, Adadelta]
    optimizers = [RMSprop]
    # drop = np.linspace(0.1, 0.5, 5)       # numpy가 아닌 튜플로 넣어주면 에러가 안생김
    dropout = [0.3]
    epochs = [3]
    node = [128, 256, 512] 
    return{"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout, "epochs" : epochs, "learning_rate" : learning_rate, "node" : node}

hyperparameters = create_hyperparameter()

# keras 모델을 sklearn에서 사용할수있게 바꿔주는 기능
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose = 1)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
search = RandomizedSearchCV(model, hyperparameters, cv = 3)
search.fit(x_train, y_train)

print(search.best_params_)

score = search.score(x_test, y_test)
print("최종 스코어 : ", score)

# 결과
# {'optimizer': 'adam', 'epochs': 50, 'drop': 0.3, 'batch_size': 30}
# 334/334 [==============================] - 0s 1ms/step - loss: 0.1437 - accuracy: 0.9842
# 최종 스코어 :  0.9842000007629395