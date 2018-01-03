import pandas
import numpy as np
from sklearn.tree import DecisionTreeClassifier
df = pandas.read_csv('titanic.csv')[['Pclass', 'Fare', 'Age', 'Sex']]
df = df[df['Age'].notnull()]
df2 = pandas.read_csv('titanic.csv')[['Survived', 'Age']]
df2 = df2[df2['Age'].notnull()]
aim_var = df2['Survived']
to_change = {'male':1, 'female':0}
df['Sex'] = df['Sex'].map(to_change)
# print df
clf = DecisionTreeClassifier(random_state=241)
clf.fit(df, aim_var)
importances = clf.feature_importances_
print importances
# Fare Sex