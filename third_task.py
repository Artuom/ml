import pandas
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

df = np.loadtxt('wine.data.txt', delimiter=',')
classes = df[:, 0]
values = df[:, 1:]
kf = KFold(n_splits=5, shuffle=True, random_state=42)


def assesment(values):
    results = []
    for k in xrange(1, 51):
        neighbors = KNeighborsClassifier(n_neighbors=k)
        results.append(cross_val_score(neighbors, values, classes, cv=kf, scoring='accuracy'))
    kachestvo = pandas.DataFrame(results, xrange(1, 51)).mean(axis=1).sort_values(ascending=False)
    return kachestvo


top_kachestvo = assesment(values).head(1)
# print(1, top_kachestvo.index[0])
# print(2, top_kachestvo.values[0])

new_values = scale(values)
new_top_kachestvo = assesment(new_values).head(1)
print new_top_kachestvo.index[0]
print '{0:4.2f}'.format(new_top_kachestvo.values[0])

