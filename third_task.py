import pandas
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

df = np.loadtxt('wine.data.txt', delimiter=',')
classes = df[:, 0]
values = df[:, 1:]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
for k in xrange(1, 51):
    neighbors = KNeighborsClassifier(n_neighbors=k)
    results.append(cross_val_score(neighbors, values, classes, cv=kf, scoring='accuracy'))

kachestvo = pandas.DataFrame(results, xrange(1, 51)).mean(axis=1).sort_values(ascending=False)

top_accuracy = kachestvo.head(1)
print(1, top_accuracy.index[0])
print(2, top_accuracy.values[0])
# https://github.com/tyz910/hse-shad-ml/blob/master/03-statement-neighbours/main.py