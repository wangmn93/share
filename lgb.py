import numpy as np
from lightgbm import train, Dataset
from sklearn.datasets import load_boston
from scipy.stats import pearsonr
from numpy.testing import assert_array_equal

params = dict(boosting_type='gbdt', class_weight=None, colsample_bytree=1.,
               device='cpu',importance_type='split', learning_rate=0.1,
              max_depth=-1, min_child_samples=20, min_child_weight=0.001,
              min_split_gain=0.0, n_estimators=5, n_jobs=1, num_leaves=31,
              objective='mse', random_state=0, reg_alpha=0.0, reg_lambda=0.0,
              silent=True, subsample=1., subsample_for_bin=200000,
              subsample_freq=0)

X,y = load_boston(return_X_y=True)
dataset = Dataset(data=X, label=y, free_raw_data=False)

train_index = np.arange(0,100)
test_index = np.arange(100,200)

train_set0 = Dataset(X[train_index],y[train_index], free_raw_data=False).construct()
train_set1 = dataset.subset(used_indices=train_index).construct()

assert_array_equal(train_set0.data, train_set1.data)
assert_array_equal(train_set0.label, train_set1.label)

booster0 = train(params=params, train_set=train_set0, num_boost_round=5)
booster1 = train(params=params, train_set=train_set1, num_boost_round=5)

pred0 = booster0.predict(X[test_index])
pred1 = booster1.predict(X[test_index])

print(pearsonr(pred0,pred1))
