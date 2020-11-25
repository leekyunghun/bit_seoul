# 과적합 방지
# 1. 훈련데이터를 늘린다
# 2. feature 수를 줄인다
# 3. regularization

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBClassifier, XGBRegressor, plot_importance
import matplotlib.pyplot as plt

max_depth = 10
learning_rate = 0.1
n_estimators = 500
n_jobs = -1
colsample_bylevel = 1
colsample_bytree = 0.7

boston = load_boston()

x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, train_size = 0.8, shuffle = True, random_state = 66)

model = XGBRegressor(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators, n_jobs=n_jobs,
                     colsample_bylevel=colsample_bylevel, colsample_bytree=colsample_bytree)
                    
                    # max_depth = 나무의 최대 깊이, 값을 늘리면 모델이 더 복잡해지고 과적합 될 가능성이 높아진다
                    # learning_rate = 학습률 디폴트 값 : 0.3, 사용되는 일반적인 최종 값 : 0.01-0.2
                    # n_estimators = 생성할 tree 수
                    # n_jobs = 병렬로 실행할 수,  -1은 전부 다 사용
                    # colsample_bylevel = 각 트리를 훈련하기 위해 각 노드에서 사용되는 기능  default = 1
                    # colsample_bytree = 각 트리를 훈련하는 데 사용될 기능 (무작위로 선택됨)의 비율, 보통 0.5 ~ 1 사용됨, default = 1

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

print("최종정답률 : ", r2_score(y_test, y_predict))

plot_importance(model)          # matplotlib으로 이쁘게 볼수있다.
plt.show()

# 1. 주어진 파라미터
# max_depth = 5
# learning_rate = 1
# n_estimators = 300
# n_jobs = -1
# colsample_bylevel = 1
# colsample_bytree = 1

# 최종정답률 :  0.8454028763099724

# 2. 파라미터 수정 후
# max_depth = 10
# learning_rate = 0.1
# n_estimators = 500
# n_jobs = -1
# colsample_bylevel = 1
# colsample_bytree = 0.7

# 최종정답률 :  0.9361737288707355