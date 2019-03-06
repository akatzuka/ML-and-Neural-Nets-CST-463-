%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata
from sklearn import datasets
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#mnist = fetch_mldata('MNIST original')
#mnist
#X, y = mnist["data"], mnist["target"]
#X.shape
#y.shape

digits = datasets.load_digits()
X, y = digits["data"], digits["target"]
X.shape
y.shape

#some_digit = X[36000]
#some_digit_image = some_digit.reshape(28, 28)
#plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
#interpolation="nearest")
#plt.axis("off")
#plt.show()

plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,
y, test_size=0.30)  

log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
voting_clf = VotingClassifier(
estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
voting='hard')
voting_clf.fit(X_train, y_train)

for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    
X_val_predictions = np.empty((len(X), len(estimators)), dtype=np.float32)

for index, estimator in enumerate(estimators):
    X_val_predictions[:, index] = estimator.predict(X)

X_val_predictions



class Stacker(BaseEstimator, TransformerMixin):
    def __init__ (self, attribute_names):
        self.attribute_names = attribute_names
    def fit (self, X, y = None):
        return self
    def predict (self, X):
        return X[self.attribute_names]