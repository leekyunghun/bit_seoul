from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size = 0.8, shuffle = True, random_state = 66)

# model = DecisionTreeClassifier(max_depth=4)
model = RandomForestClassifier(max_depth=4)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)

print("acc : ", acc)
print(model.feature_importances_)

# acc :  0.9649122807017544
# [0.02751064 0.01534157 0.02617932 0.05182397 0.00397574 0.01686472
#  0.0277883  0.09058234 0.00250049 0.00357469 0.00604151 0.00525667
#  0.01054341 0.05916275 0.00372855 0.003547   0.00497302 0.00253921
#  0.0045064  0.00280893 0.11776537 0.01484487 0.17029915 0.10552116
#  0.00974419 0.02023137 0.03880582 0.13935245 0.00883117 0.00535521]