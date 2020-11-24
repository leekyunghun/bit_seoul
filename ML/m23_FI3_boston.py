# 기준 xgboost
# 1. feature importances가 0인 값들을 제거 or 하위 30% 제거
# 디폴트와 성능 비교
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier, XGBRegressor                           # xgboost -> cmd에서 pip install xgboost로 다운

boston = load_boston()
boston.data = np.delete(boston.data, [1, 3, 6, 11], axis=1)
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, train_size = 0.8, shuffle = True, random_state = 66)

model = XGBRegressor(max_depth=4)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

print("최종정답률 : ", r2_score(y_test, y_predict))

print(model.feature_importances_)

# import matplotlib.pyplot as plt
# import numpy as np
# def plot_feature_importances_boston(model):
#     n_features = boston.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), boston.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)

# plot_feature_importances_boston(model)
# plt.show()

# Default
# 최종정답률 :  0.9328109815565079
# [0.01669537 0.00150525 0.02149532 0.0007204  0.05927434 0.29080436
#  0.01197547 0.05330402 0.0360474  0.02261044 0.07038534 0.01352609

# FI
# 최종정답률 :  0.9310264719671016
# [0.01555075 0.01334106 0.05806042 0.28824472 0.06061063 0.02149809
#  0.02820203 0.08540782 0.0185576  0.41052684]