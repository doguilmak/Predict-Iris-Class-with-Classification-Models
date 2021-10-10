# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 12:17:56 2021

@author: doguilmak

dataset: https://www.kaggle.com/uciml/iris

"""
#%%
# 1. Libraries

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

#%%
# Data Preprocessing

start = time.time()
df = pd.read_excel('iris.xls')
print(df.info()) # Looking for the missing values
print(df)

print("{} duplicated data.".format(df.duplicated().sum()))
dp = df[df.duplicated(keep=False)]
dp.head(5)
df.drop_duplicates(inplace= True)
print("{} duplicated data.".format(df.duplicated().sum()))
print("\n", df.describe().T)

x = df.iloc[:, 0:4].values
y = df.iloc[:, 4:].values

# Train - Test Split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.33, random_state=0) 


# Scaling Datas
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#%%
# System Success Libraries

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

#%%
# Logistic Regression

from sklearn.linear_model import LogisticRegression
logr = LogisticRegression(random_state=0)
logr.fit(X_train,y_train) 
y_pred = logr.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print('\nLogistic Regression Confusion Matrix')
print(cm)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")

# Plotting Confusion Matrix
plot_confusion_matrix(logr, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Logistic Regression Classifier')  
plt.show()

predict = np.array([6.7, 3.3, 5.7, 2.5]).reshape(1, 4)
print(logr.predict(predict))

#%%
# K-NN

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, metric='minkowski') 
knn.fit(X_train,y_train) 
y_pred = knn.predict(X_test) 

cm = confusion_matrix(y_test,y_pred)

print('\nK-NN Confusion Matrix')
print(cm)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")

# Plotting Confusion Matrix
plot_confusion_matrix(knn, X_test, y_test)
plt.title('K-NN Classifier')  
plt.show()

predict = np.array([6.7, 3.3, 5.7, 2.5]).reshape(1, 4)
print(knn.predict(predict))

from sklearn import neighbors
from mlxtend.plotting import plot_decision_regions

def knn_comparison(data, k, m="euclidean"):
  x = df.iloc[:,1:3].values 
  y = df['iris'].astype('category').cat.codes.to_numpy()
  clf = neighbors.KNeighborsClassifier(n_neighbors=k, metric = m)
  clf.fit(x, y)
  plot_decision_regions(x, y, clf=clf, legend=2)
  plt.xlabel('Sepal Width')
  plt.ylabel('Setal Length')
  plt.title('K-NN with K='+ str(k))
  plt.show()
  
for i in [1, 2, 3, 5, 10, 15]:
    knn_comparison(df, i)
    
for i in [1, 2, 3, 5, 10, 15]:
    knn_comparison(df, i, "manhattan")
    
print(f"\nK-NN Score: {knn.score(X_test, y_test)}")

#%%
# SVC

from sklearn.svm import SVC
svc = SVC(kernel='sigmoid') # Types: linear-rbf-sigmoid-precomputed
svc.fit(X_train,y_train)
y_pred = svc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print('\nSVC Confusion Matrix')
print(cm)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")

# Plotting Confusion Matrix
plot_confusion_matrix(svc, X_test, y_test, cmap=plt.cm.Blues)
plt.title('SVC Classifier')  
plt.show()

predict = np.array([6.7, 3.3, 5.7, 2.5]).reshape(1, 4)
print(svc.predict(predict))

#%% 
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
# Naif Bayes:
# -Gaussian Naive Bayes(GaussianNB)
# -Multinomial Naive Bayes(MultinomialNB)
# -Bernoulli Navie Bayes(BernoulliNB)

gnb = GaussianNB()
gnb.fit(X_train, y_train) 
y_pred = gnb.predict(X_test) 

cm = confusion_matrix(y_test,y_pred)

print('\nGaussian Naive Bayes(GaussianNB)')
print(cm)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")

# Plotting Confusion Matrix
plot_confusion_matrix(gnb, X_test, y_test)
plt.title('Gaussian Naive Bayes')
plt.show()

predict = np.array([6.7, 3.3, 5.7, 2.5]).reshape(1, 4)
print(gnb.predict(predict))

#%%
# Desicion Tree Classifier

from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion = 'entropy') 
dtc.fit(X_train, y_train) 
y_pred = dtc.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print('\nDecision Tree Classifier Confusion Matrix')
print(cm)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")

# Plotting Confusion Matrix
plot_confusion_matrix(dtc, X_test, y_test)
plt.title('Decision Tree Classifier')
plt.show()

predict = np.array([6.7, 3.3, 5.7, 2.5]).reshape(1, 4)
print(dtc.predict(predict))

from sklearn import tree
tree.plot_tree(dtc)
plt.show()

#%%
# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10, criterion = 'entropy')
rfc.fit(X_train,y_train)
y_pred = rfc.predict(X_test)

cm = confusion_matrix(y_test,y_pred)

print('\nRandom Forest Classifier')
print(cm)
print(f"Accuracy score: {accuracy_score(y_test, y_pred)}")

# Plotting Confusion Matrix
plot_confusion_matrix(dtc, X_test, y_test, cmap=plt.cm.Blues)
plt.title('Random Forest Classifier')
plt.show()

predict = np.array([6.7, 3.3, 5.7, 2.5]).reshape(1, 4)
print(rfc.predict(predict))

#%%    
# ROC , TPR, FPR

print("\nPredict Probability")
y_proba = rfc.predict_proba(X_test)
print("Real values:\n", y_test)
print("Predict probability:\n", y_proba)

from sklearn import metrics
fpr , tpr , thold = metrics.roc_curve(y_test, y_proba[:,0], pos_label='Iris-setosa')

print("False Positive Rate:\n",fpr)
print("True Positive Rate:\n",tpr)

# Plotting Receiver Operating Characteristic
import seaborn as sns
font1 = {'family': 'serif',
        'color': 'black',
        'weight': 'normal',
        'size': 10,
        }
font2 = {'family': 'serif',
         'color': 'black',
         'weight': 'normal',
         'size': 15,
         }

lw = 1

sns.set_style("whitegrid")
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.plot(fpr, tpr, color='red', linestyle='-', marker='o', markerfacecolor='black', markersize=5)
plt.title("ROC", fontdict=font2)
plt.xlabel("False Positive Rate", fontdict=font1)
plt.ylabel("True Positive Rate", fontdict=font1)
plt.show()


#%%
# K-Fold Cross Validation

from sklearn.model_selection import cross_val_score

success = cross_val_score(estimator = svc, X=X_train, y=y_train , cv = 4)
print("\nK-Fold Cross Validation:")
print("Success Mean:\n", success.mean())
print("Success Standard Deviation:\n", success.std())

#%%
# Grid Search

from sklearn.model_selection import GridSearchCV
p = [{'C':[1,2,3,4,5], 'kernel':['linear'], 'gamma':[1,0.5,0.1,0.01,0.001]},
     {'C':[1,2,3,4,5], 'kernel':['rbf'], 'gamma':[1,0.5,0.1,0.01,0.001]},
     {'C':[1,2,3,4,5], 'kernel':['sigmoid'], 'gamma':[1,0.5,0.1,0.01,0.001]},
     {'C':[1,2,3,4,5], 'kernel':['callable'], 'gamma':[1,0.5,0.1,0.01,0.001]}]


gs = GridSearchCV(estimator= svc,
                  param_grid=p,
                  scoring='accuracy',
                  cv=5,
                  n_jobs=-1)

grid_search = gs.fit(X_train, y_train)
best_result = grid_search.best_score_
best_parameters = grid_search.best_params_
print("\nGrid Search:")
print("Best result:\n", best_result)
print("Best parameters:\n", best_parameters)

#%%
# Pickle
"""
import pickle
file = "logr.save"
pickle.dump(logr, open(file, 'wb'))

downloaded_data = pickle.load(open(file, 'rb'))
print(downloaded_data.predict(X_test))
"""

#%%

end = time.time()
cal_time = end - start
print("\nProcess took {} seconds.".format(cal_time))
