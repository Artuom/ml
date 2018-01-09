from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV
import numpy as np
from sklearn.svm import SVC
import pandas

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

score = 0
C = 0
for a in gs.grid_scores_:
    if a.mean_validation_score > score:
        score = a.mean_validation_score
        C = a.parameters['C']

clf.fit(X_train, y)
words = count_vector.get_feature_names()
coef = pandas.DataFrame(clf.coef_.data, clf.coef_.indices)
top_words = coef[0].map(lambda w: abs(w)).sort_values(ascending=False).head(10).index.map(lambda i: words[i])
top_words.sort_values()
print top_words
