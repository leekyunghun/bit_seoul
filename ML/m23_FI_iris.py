# 기준 xgboost
# 1. feature importances가 0인 값들을 제거 or 하위 30% 제거
# 디폴트와 성능 비교

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier                           # xgboost -> cmd에서 pip install xgboost로 다운

iris = load_iris()

iris.data = iris.data[:, 1:]
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, train_size = 0.8, shuffle = True, random_state = 66)

model = XGBClassifier(max_depth=4)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)

print("acc : ", acc)
print(model.feature_importances_)

# import matplotlib.pyplot as plt
# import numpy as np
# def plot_feature_importances_iris(model):
#     n_features = iris.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), iris.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)

# plot_feature_importances_iris(model)
# plt.show()

# default
# acc :  0.9
# [0.01759811 0.02607087 0.6192673  0.33706376]

# FI
# acc :  0.8666666666666667
# [0.02887404 0.04256101 0.9285649 ]