import pandas
import numpy as np
from sklearn.svm import SVC

df = pandas.read_csv('svm-data.csv', header=None)
aim_var = df[0]
values = df.loc[:, 1:]

svc_met = SVC(C=100000, kernel='linear', random_state=241)

result = svc_met.fit(values, aim_var)
print result.support_
