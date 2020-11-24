import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

iris = pd.read_csv('./data/csv/iris_ys.csv', header=0, index_col=0)

x = iris.iloc[:, 0:4]
y = iris.iloc[:, 4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

allAIgorithms = all_estimators(type_filter = 'classifier')
kfold = KFold(n_splits = 5, shuffle=True)

for (name, algorithm) in allAIgorithms:
    try:
        model = algorithm()
        scores = cross_val_score(model, x_train, y_train, cv = kfold) 
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 :', accuracy_score(y_test, y_pred))
        print(name, '의 CV_score : ', scores, '\n')

    except:
        pass

import sklearn
print(sklearn.__version__)