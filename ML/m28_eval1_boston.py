from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, accuracy_score

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, shuffle = True, random_state = 66)

# 2. 모델
# model = XGBRegressor(n_estimators=1000, learning_rate=0.01)
model = XGBRegressor(learning_rate=0.1)

# 3. 훈련
model.fit(x_train, y_train, verbose = 1, eval_metric = "rmse", eval_set = [(x_train, y_train), (x_test, y_test)])

# eval metric의 대표 param : rmse, mae, logloss, error, auc

results = model.evals_result()
# print("eval's result : ", results)

predict = model.predict(x_test)

r2 =r2_score(predict, y_test)
print("R2 : ", r2)

# [99]    validation_0-rmse:0.42866       validation_1-rmse:2.37740
# R2 :  0.9319043870366732