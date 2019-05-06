from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *


train_path = ["./data/titanic_train.csv", "./data/titanic_test.csv"]


reader = Reader(sep=",", header=0)
data = reader.train_test_split(train_path, 'Survived')

data = Drift_thresholder().fit_transform(data)

Optimiser().evaluate(None, data)




Predictor().fit_predict(None, data)