# Importing the libraries
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# reading the data
df=pd.read_csv("./emails.csv")
#print(df.head(10))

# print(df.isnull().sum())

# print(df.describe())

# print(df.corr())

X=df.iloc[:, 1:3001]
y=df.iloc[:, -1].values

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.25)

mnb=MultinomialNB(alpha=1.9)
mnb.fit(train_x, train_y)
mnb_pred=mnb.predict(test_x)
print("Accuracy Score for Naive Bayes: ", accuracy_score(mnb_pred, test_y))

svc=SVC(C=1.0, kernel="rbf", gamma="auto")
svc.fit(train_x, train_y)
svc_pred=svc.predict(test_x)
print("Accuracy Score for SVC: ", accuracy_score(svc_pred, test_y))

rfc=RandomForestClassifier(n_estimators=10, criterion="gini")
rfc.fit(train_x, train_y)
rfc_pred=rfc.predict(test_x)
print("Accuracy Score for Random Forest Classification: ", accuracy_score(rfc_pred, test_y))
