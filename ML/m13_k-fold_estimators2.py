import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.testing import all_estimators
import warnings

warnings.filterwarnings('ignore')

boston_house_prices = pd.read_csv('./data/csv/boston_house_prices.csv', header=1, index_col=None)
print(boston_house_prices)

x = boston_house_prices.iloc[:, :13]
y = boston_house_prices.iloc[:, 13]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 66)

allAIgorithms = all_estimators(type_filter = 'regressor')
kfold = KFold(n_splits = 5, shuffle=True)

for (name, algorithm) in allAIgorithms:
    try:                                                                    # sklearn의 버전이 올라간것에 의해 안되는 model들이 있어서 예외처리로 안되는건 pass
        model = algorithm()
        scores = cross_val_score(model, x_train, y_train, cv = kfold) 
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답률 :', r2_score(y_test, y_pred))
        print(name, '의 CV_score : ', scores, '\n')

    except:
        pass

import sklearn
print(sklearn.__version__)
