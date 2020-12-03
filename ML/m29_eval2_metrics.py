from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

iris = load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 66)

# 2. 모델
# model = XGBClassifier(n_estimators=1000, learning_rate=0.01)
model = XGBClassifier(learning_rate=0.1)

# 3. 훈련
model.fit(x_train, y_train, verbose = 1, eval_metric = ["mlogloss", "merror"], eval_set = [(x_train, y_train), (x_test, y_test)])

# eval metric의 대표 param : rmse, mae, logloss, error, auc

results = model.evals_result()
# print("eval's result : ", results)

predict = model.predict(x_test)

accuracy = accuracy_score(y_test, predict)
print("accuracy : ", accuracy)

# [99]    validation_0-mlogloss:0.02383   validation_0-merror:0.00000     validation_1-mlogloss:0.14753   validation_1-merror:0.10000
# accuracy :  0.9