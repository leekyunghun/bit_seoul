import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()        # mnist 예제의 좋은기능 train set 과 test set을 나눠주는 기능이 있음

print(x_train.shape, x_test.shape)                              # (60000, 28, 28) (10000, 28, 28)
print(y_train.shape, y_test.shape)                              # (60000, )       (10000, )

# 데이터 전처리 1.OneHotEncoding
from tensorflow.keras.utils import to_categorical               # 분류모델에서는 onehotencoding 필수
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_train.shape)
print(y_test.shape)

x_train = x_train.reshape(60000, 28 * 28, 1).astype("float32") / 255.        # .astype("type") => 형 변환
x_test = x_test.reshape(10000, 28 * 28, 1).astype("float32") / 255.

# 2.모델
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint    
from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta
from tensorflow.keras.activations import selu, elu, relu
from sklearn.metrics import r2_score

# keras model을 함수형태로 만들었음
def build_model(activation, drop = 0.5, optimizer = Adam, learning_rate = 0.001, node = 100):
    inputs = Input(shape = (28*28, 1), name = 'input')
    x = LSTM(50, name = 'hidden1')(inputs)
    x = Activation(activation)(x)
    x = Dropout(drop)(x)

    x = Dense(node, name = 'hidden2')(x)
    x = Activation(activation)(x)
    x = Dropout(drop)(x)

    x = Dense(node, name = 'hidden3')(x)
    x = Activation(activation)(x)
    x = Dropout(drop)(x)

    x = Dense(node, name = 'hidden4')(x)
    x = Activation(activation)(x)
    x = Dropout(drop)(x)
    
    outputs = Dense(10, activation='softmax', name = 'outputs')(x)
    model = Model(inputs = inputs, outputs = outputs)

    # early_stopping = EarlyStopping(monitor = 'val_loss', patience = stop)
    model.compile(optimizer = optimizer, metrics = ['accuracy'], loss = 'categorical_crossentropy')
    return model

    # GridSearch parameter 함수
def create_hyperparameter():                # 파라미터를 지정해줄때의 이름과 모델 함수에 들어있는 파라미터 변수 이름을 동일시켜야함
    batches = [30, 40]
    optimizers = ['rmsprop', 'adam', 'adadelta']
    # drop = np.linspace(0.1, 0.5, 5)       # numpy가 아닌 튜플로 넣어주면 에러가 안생김
    dropout = [0.2, 0.3]
    epochs = [30, 50]
    learning_rate = [0.01, 0.05, 0.001]
    node = [64, 128]
    activation = ['relu', 'selu', 'elu']
    return{"batch_size" : batches, "optimizer" : optimizers, "drop" : dropout, "epochs" : epochs, "node" : node, "activation" : activation, "learning_rate" : learning_rate}

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

# {'optimizer': 'rmsprop', 'node': 128, 'learning_rate': 0.01, 'epochs': 30, 'drop': 0.2, 'batch_size': 30, 'activation': 'selu'}
# 334/334 [==============================] - 4s 11ms/step - loss: 0.2470 - accuracy: 0.9329
# 최종 스코어 :  0.9329000115394592