import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


input_file = "https://raw.githubusercontent.com/akatzuka/CST-463-Project-1/master/default_cc_train.csv"
df = pd.read_csv(input_file)

df.info()
df.isna().sum()

plt.hist(df["AGE"])
ax = sns.countplot(df["EDUCATION"])
ax = sns.countplot(df["SEX"])
sns.countplot(df["default.payment.next.month"])

sns.distplot(df['LIMIT_BAL'], hist=True, kde=True, 
             bins=int(180/5), color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})

mean_male = int(np.mean(df[df["SEX"] == 1]["LIMIT_BAL"]))
mean_female = int(np.mean(df[df["SEX"] == 2]["LIMIT_BAL"]))
sns.countplot(df[df["SEX"]==1]["default.payment.next.month"])
sns.countplot(df[df["SEX"]==2]["default.payment.next.month"])

male_default = len(df[df["SEX"]==1][df["default.payment.next.month"]==1])/len(df[df["SEX"]==1])
female_default = len(df[df["SEX"]==2][df["default.payment.next.month"]==1])/len(df[df["SEX"]==2])

X = df.iloc[:,1:24]
y = np.asarray(df["default.payment.next.month"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled,
y, test_size=0.25)

rnd_clf = RandomForestClassifier(n_estimators = 60, n_jobs = -1)
rnd_clf_fitted = rnd_clf.fit(X_train, y_train)
RFC_score = rnd_clf.score(X_test, y_test)

etclf = ExtraTreesClassifier(n_estimators = 50, n_jobs = -1)
etclf_fitted = etclf.fit(X_train, y_train)
ETC_score = etclf.score(X_test, y_test)

svm_clf = SVC()
voting_clf = VotingClassifier(
estimators=[('etf', etclf), ('rf', rnd_clf), ('svc', svm_clf)],
voting='hard')
voting_clf.fit(X_train, y_train)

for clf in (etclf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))
    