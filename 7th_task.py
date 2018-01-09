from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
import numpy as np
from sklearn.svm import SVC

newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
X = newsgroups.data
y = newsgroups.target
new_y = []
#for i in y:
#    new_y.append(str(i))

cv = KFold(n_splits=5, shuffle=True, random_state=241)
grid = {'C': np.power(10.0, np.arange(-5, 6))}
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)

count_vector = TfidfVectorizer()
X_train = count_vector.fit_transform(X)
gs.fit(X_train, y)
print gs.coef_