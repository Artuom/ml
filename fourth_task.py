import pandas
import numpy as np
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor

dset = load_boston()

values = dset['data']
classes = dset['target']

opti_values = scale(values)
cross_valid = KFold(n_splits=5, shuffle=True, random_state=42)

metriks = np.linspace(1, 10, 200)
results = []
for p in metriks:
    neighbor = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p, metric='minkowski')
    results.append(cross_val_score(neighbor, opti_values, classes, cv=cross_valid, scoring='neg_mean_squared_error'))

kachestvo = pandas.DataFrame(results, metriks).mean(axis=1).sort_values(ascending=False)
print kachestvo
