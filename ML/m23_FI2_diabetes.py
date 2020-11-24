# 기준 xgboost
# 1. feature importances가 0인 값들을 제거 or 하위 30% 제거
# 디폴트와 성능 비교
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier, XGBRegressor                          # xgboost -> cmd에서 pip install xgboost로 다운

diabetes = load_diabetes()
diabetes.data = np.delete(diabetes.data, [0, 4, 5], axis=1)
x_train, x_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, train_size = 0.8, shuffle = True, random_state = 66)

model = XGBRegressor(max_depth=4)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

print("최종정답률 : ", r2_score(y_test, y_predict))

print(model.feature_importances_)

# import matplotlib.pyplot as plt
# import numpy as np
# def plot_feature_importances_diabetes(model):
#     n_features = diabetes.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), diabetes.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)

# plot_feature_importances_diabetes(model)
# plt.show()

# Default
# 최종정답률 :  0.31163770597265394
# [0.03951401 0.08722725 0.18159387 0.08551976 0.04845208 0.06130722
#  0.05748899 0.0561045  0.32311246 0.05967987]

# FI
# 최종정답률 :  0.33352805041288736
# [0.06622774 0.2319132  0.10845745 0.07807689 0.14214802 0.27237186
#  0.10080484]