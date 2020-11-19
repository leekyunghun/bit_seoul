from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# 1.데이터
x_data = [[0, 0], [1, 0], [0, 1], [1, 1]]
y_data = [0, 1, 1, 0]

# 2.모델
model = LinearSVC()

# 3.훈련
model.fit(x_data, y_data)

# 4.평가, 예측
y_predict = model.predict(x_data)
print(y_data, "의 예측결과 ", y_predict)

acc1 = model.score(x_data, y_predict)
print("model.score : ", acc1)

acc2 = accuracy_score(y_data, y_predict)
print("accuracy_score : ", acc2)
