from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

cancer = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(cancer.data, cancer.target, train_size = 0.8, shuffle = True, random_state = 66)

model = DecisionTreeClassifier(max_depth=4)

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

acc = accuracy_score(y_test, y_predict)

print("acc : ", acc)
print(model.feature_importances_)

# acc :  0.9385964912280702
# [0.         0.0624678  0.         0.         0.         0.
#  0.02364429 0.         0.         0.         0.         0.
#  0.         0.01297421 0.         0.         0.         0.
#  0.         0.         0.         0.02187676 0.         0.75156772
#  0.00738884 0.         0.         0.12008039 0.         0.        ]