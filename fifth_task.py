import pandas
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df_train = pandas.read_csv('perceptron-train.csv', header=None)
df_test = pandas.read_csv('perceptron-test.csv', header=None)

train_aim_vars = df_train[0]
train_vals = df_train.loc[:, 1:]
test_aim_vars = df_test[0]
test_vals = df_test.loc[:, 1:]
clf = Perceptron(random_state=241)
clf.fit(train_vals, train_aim_vars)

accuracy_first = accuracy_score(test_aim_vars, clf.predict(test_vals))
print "before => ", accuracy_first # 0.655

scaler = StandardScaler()
train_vals_scaled = scaler.fit_transform(train_vals)
test_vals_scaled = scaler.transform(test_vals)

clf_new = Perceptron(random_state=241)
clf_new.fit(train_vals_scaled, train_aim_vars)
accuracy_scaled = accuracy_score(test_aim_vars, clf_new.predict(test_vals_scaled))
print "after => ", accuracy_scaled

print "diff => ", accuracy_scaled - accuracy_first