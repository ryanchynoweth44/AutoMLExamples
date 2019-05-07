from mlbox.preprocessing import *
from mlbox.optimisation import *
from mlbox.prediction import *


train_path = ["./data/titanic_train.csv", "./data/titanic_test.csv"]


reader = Reader(sep=",", header=0)
data = reader.train_test_split(train_path, 'Survived')

reader.train_test_split()
data = Drift_thresholder().fit_transform(data)


opt = Optimiser(scoring='accuracy', n_folds=3)
opt.evaluate(None, data)

space = {
    'ne__numerical_strategy': {"search":"choice", "space":[0]},
    'ce__strategy': {"search":"choice", "space":["label_encoding", "random_projection", "entity_embedding"]},
    'fs__threshold': {"search":"uniform", "space":[0.01, 0.3]},
    'est__max_depth': {"search":"choice", "space":[3,4,5,6,7]}
}

best_params = opt.optimise(space, data, 10)

model = Predictor().fit_predict(best_params, data)

