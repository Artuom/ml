import pandas
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

df = np.loadtxt('wine.data.txt', delimiter=',')
classes = df[:, 0]
values = df[:, 1:]
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print kf
