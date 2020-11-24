from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size = 0.8, shuffle = True, random_state = 66)

# model = DecisionTreeClassifier(max_depth=4)
# model = RandomForestClassifier(max_depth=4)
model = GradientBoostingClassifier(max_depth=4)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)

print("acc : ", acc)
print(model.feature_importances_)

# acc :  0.9649122807017544
# [1.27320854e-04 5.60780869e-02 7.10405260e-04 1.84993145e-03
#  3.16910409e-03 1.75472015e-03 1.54015901e-03 8.61798028e-02
#  1.68790852e-03 1.69100029e-04 4.78669723e-03 1.33534529e-05
#  1.26588346e-03 1.38133210e-02 3.31145197e-03 3.50101573e-03
#  4.68699442e-03 1.32623661e-04 1.27424182e-03 1.09908194e-03
#  2.44034546e-01 3.40218603e-02 4.18466915e-03 4.12886768e-01
#  4.20846367e-03 1.46431100e-04 6.07237611e-03 1.07227682e-01
#  3.16351925e-05 3.43642256e-05]