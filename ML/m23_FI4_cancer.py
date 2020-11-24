# 기준 xgboost
# 1. feature importances가 0인 값들을 제거 or 하위 30% 제거
# 디폴트와 성능 비교
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBClassifier                           # xgboost -> cmd에서 pip install xgboost로 다운

cancer = load_breast_cancer()
cancer.data = np.delete(cancer.data, [0, 9, 25], axis = 1)
x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size = 0.8, shuffle = True, random_state = 66)

model = XGBClassifier(max_depth=4)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)

print("acc : ", acc)
print(model.feature_importances_)

# import matplotlib.pyplot as plt
# import numpy as np
# def plot_feature_importances_cancer(model):
#     n_features = cancer.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_, align='center')
#     plt.yticks(np.arange(n_features), cancer.feature_names)
#     plt.xlabel('Feature Importances')
#     plt.ylabel('Features')
#     plt.ylim(-1, n_features)

# plot_feature_importances_cancer(model)
# plt.show()

# Default
# acc :  0.9736842105263158
# [0.         0.03518598 0.00053468 0.02371635 0.00661651 0.02328466
#  0.00405836 0.09933352 0.00236719 0.         0.01060954 0.00473884
#  0.01074011 0.01426315 0.0022232  0.00573987 0.00049415 0.00060479
#  0.00522006 0.00680739 0.01785728 0.0190929  0.3432317  0.24493258
#  0.00278067 0.         0.01099805 0.09473949 0.00262496 0.00720399]

# FI
# acc :  0.9736842105263158
# [0.03518598 0.00053468 0.02371635 0.00661651 0.02328466 0.00405836
#  0.09933352 0.00236719 0.01060954 0.00473884 0.01074011 0.01426315
#  0.0022232  0.00573987 0.00049415 0.00060479 0.00522006 0.00680739
#  0.01785728 0.0190929  0.3432317  0.24493258 0.00278067 0.01099805
#  0.09473949 0.00262496 0.00720399]