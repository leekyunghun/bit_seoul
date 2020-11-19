from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
# 1.데이터
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 1, 1, 0]

# 2.모델
# model = LinearSVC()
# model = SVC()
model = Sequential()
model.add(Dense(1, input_dim = 2, activation = 'sigmoid'))

# 3.훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['acc'])
model.fit(x_data, y_data, batch_size = 1, epochs = 100)
model.fit(x_data, y_data)

# 4.평가, 예측
y_predict = model.predict(x_data)
print(y_data, "의 예측결과\n", y_predict)

acc1 = model.evaluate(x_data, y_data)              # 준비되어있는 데이터 사용
print("model.evaluate : ", acc1)

# acc2 = accuracy_score(y_data, y_predict)        # 실제값과 predict한 값 사용
# print("accuracy_score : ", acc2)
