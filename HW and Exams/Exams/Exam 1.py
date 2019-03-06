import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

input_file = "https://raw.githubusercontent.com/grbruns/cst495/master/imports-abridged.csv"
df = pd.read_csv(input_file)

arr = np.asarray(list(df.columns.values))
df.info()

for i in range(df.shape[0]):
    if (df.price[i] == "?"):
        df = df.drop([i])
      
sns.distplot(df.price.astype(int))
plt.title('Histogram of Price')

sns.scatterplot(x = df.price.astype(int),y = df.engine_size.astype(int))
plt.title('Scatterplot of Price vs Engine Size')

for i in range(df.shape[0]):
    if (df.num_of_doors[i] == "two"):
        df.num_of_doors[i]=2
    if (df.num_of_doors[i] == "four"):
        df.num_of_doors[i]=4

df_door = pd.get_dummies(df.num_of_doors)
df = df.join(df_door)

X = df
X = X.drop('price', 1)
X = X.drop('num_of_doors', 1)
y = np.asarray(df.price)

df = df.replace("?", "nan")

X_mean = Imputer(missing_values=np.nan, strategy='mean')
X = X_mean.fit(X)



from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import tree

breast_cancer = load_breast_cancer()
features = breast_cancer['feature_names']
X = breast_cancer['data']
y = breast_cancer['target']

np.isnan(X).sum()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,
y, test_size=0.25)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)