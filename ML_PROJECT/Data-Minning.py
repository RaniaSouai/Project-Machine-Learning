import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from joblib import dump
from pandas.io.common import infer_compression
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, RandomForestClassifier

from sklearn.model_selection import KFold, cross_val_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import OneHotEncoder

from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from sklearn import metrics
from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("marketing_campaign1.csv")
train = pd.read_csv("marketing_campaign1.csv")

test= pd.read_csv("marketing_campaign1.csv")

data[['ID','Year_Birth','Education','Marital_Status','Income','Kidhome','Teenhome','Dt_Customer','Recency','MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts','MntGoldProds','NumDealsPurchases','NumWebPurchases','NumCatalogPurchases','NumStorePurchases','NumWebVisitsMonth','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','Complain','Z_CostContact','Z_Revenue','Response']]=data["ID;Year_Birth;Education;Marital_Status;Income;Kidhome;Teenhome;Dt_Customer;Recency;MntWines;MntFruits;MntMeatProducts;MntFishProducts;MntSweetProducts;MntGoldProds;NumDealsPurchases;NumWebPurchases;NumCatalogPurchases;NumStorePurchases;NumWebVisitsMonth;AcceptedCmp3;AcceptedCmp4;AcceptedCmp5;AcceptedCmp1;AcceptedCmp2;Complain;Z_CostContact;Z_Revenue;Response"].str.split(";", expand=True)

data=data.drop(["ID;Year_Birth;Education;Marital_Status;Income;Kidhome;Teenhome;Dt_Customer;Recency;MntWines;MntFruits;MntMeatProducts;MntFishProducts;MntSweetProducts;MntGoldProds;NumDealsPurchases;NumWebPurchases;NumCatalogPurchases;NumStorePurchases;NumWebVisitsMonth;AcceptedCmp3;AcceptedCmp4;AcceptedCmp5;AcceptedCmp1;AcceptedCmp2;Complain;Z_CostContact;Z_Revenue;Response"], axis=1)

train= data.copy()
test =data.copy()

#NORMALIZATION

train = train.drop('Z_CostContact',axis=1)
test = test.drop('Z_CostContact',axis=1)
train = train.drop('Z_Revenue',axis=1)
test = test.drop('Z_Revenue',axis=1)

train["Year_Birth"].astype(int)

test["Year_Birth"].astype(int)

train['Age'] = 2021- train["Year_Birth"].astype(int)
train = train.drop('Year_Birth',axis=1)

train = train.drop('Dt_Customer',axis=1)


test = test.drop('Dt_Customer',axis=1)

train["Income"] = pd.to_numeric(data["Income"], downcast="float")

test["Income"] = pd.to_numeric(data["Income"], downcast="float")

# Fillna fel income khatrou ne9es valeurs ma ywarihech 

avg_income = np.mean(train.Income)
train['Income'] = train['Income'].fillna(avg_income, axis=0)

# Fillna fel income khatrou ne9es valeurs ma ywarihech 

avg_income = np.mean(test.Income)
test['Income'] = test['Income'].fillna(avg_income, axis=0)

train.Marital_Status = train['Marital_Status'].map({'Single':1, 'Together':2, 'Married':3, 'Divorced':0, 'Widow':4, 'Alone':5,
       'Absurd':6, 'YOLO':7}).astype(int)

test.Marital_Status =test['Marital_Status'].map({'Single':1, 'Together':2, 'Married':3, 'Divorced':0, 'Widow':4, 'Alone':5,
       'Absurd':6, 'YOLO':7}).astype(int)

train.Education= train['Education'].map({'Graduation':2, 'PhD':4, 'Master':3, 'Basic':1, '2n Cycle':0}).astype(int)

test.Education= test['Education'].map({'Graduation':2, 'PhD':4, 'Master':3, 'Basic':1, '2n Cycle':0}).astype(int)

train["MntWines"] = pd.to_numeric(data["MntWines"], downcast="float")
train["MntFruits"] = pd.to_numeric(train["MntFruits"], downcast="float")
train["MntMeatProducts"] = pd.to_numeric(train["MntMeatProducts"], downcast="float")
train["MntFishProducts"] = pd.to_numeric(train["MntFishProducts"], downcast="float")
train["MntSweetProducts"] = pd.to_numeric(train["MntSweetProducts"], downcast="float")
train["MntGoldProds"] = pd.to_numeric(train["MntGoldProds"], downcast="float")

test["MntWines"] = pd.to_numeric(data["MntWines"], downcast="float")
test["MntFruits"] = pd.to_numeric(test["MntFruits"], downcast="float")
test["MntMeatProducts"] = pd.to_numeric(test["MntMeatProducts"], downcast="float")
test["MntFishProducts"] = pd.to_numeric(test["MntFishProducts"], downcast="float")
test["MntSweetProducts"] = pd.to_numeric(test["MntSweetProducts"], downcast="float")
test["MntGoldProds"] = pd.to_numeric(test["MntGoldProds"], downcast="float")

train["total_Mnt"] = train["MntWines"] + train["MntFruits"] + train["MntMeatProducts"]+ train['MntFishProducts'] + train["MntSweetProducts"] + train["MntGoldProds"]

train['MntWines_tot'] = train['MntWines']/train['total_Mnt']
train['MntFruits_tot'] = train["MntFruits"]/train['total_Mnt']
train["MntMeatProducts_tot"] = train["MntMeatProducts"]/train['total_Mnt']
train["MntFishProducts_tot"] = train["MntFishProducts"]/train['total_Mnt']
train["MntSweetProducts_tot"] = train["MntSweetProducts"]/train['total_Mnt']
train["MntGoldProds_tot"] = train["MntGoldProds"]/train['total_Mnt']

test["total_Mnt"] = test["MntWines"] + test["MntFruits"] + test["MntMeatProducts"]+ test['MntFishProducts'] + test["MntSweetProducts"] + test["MntGoldProds"]

test['MntWines_tot'] = test['MntWines']/test['total_Mnt']
test['MntFruits_tot'] = test["MntFruits"]/test['total_Mnt']
test["MntMeatProducts_tot"] = test["MntMeatProducts"]/test['total_Mnt']
test["MntFishProducts_tot"] = test["MntFishProducts"]/test['total_Mnt']
test["MntSweetProducts_tot"] = test["MntSweetProducts"]/test['total_Mnt']
test["MntGoldProds_tot"] = test["MntGoldProds"]/test['total_Mnt']

train["AcceptedCmp1"] = pd.to_numeric(train["AcceptedCmp1"], downcast="float")
train["AcceptedCmp2"] = pd.to_numeric(train["AcceptedCmp2"], downcast="float")
train["AcceptedCmp3"] = pd.to_numeric(train["AcceptedCmp3"], downcast="float")
train["AcceptedCmp4"] = pd.to_numeric(train["AcceptedCmp4"], downcast="float")
train["AcceptedCmp5"] = pd.to_numeric(train["AcceptedCmp5"], downcast="float")

test["AcceptedCmp1"] = pd.to_numeric(test["AcceptedCmp1"], downcast="float")
test["AcceptedCmp2"] = pd.to_numeric(test["AcceptedCmp2"], downcast="float")
test["AcceptedCmp3"] = pd.to_numeric(test["AcceptedCmp3"], downcast="float")
test["AcceptedCmp4"] = pd.to_numeric(test["AcceptedCmp4"], downcast="float")
test["AcceptedCmp5"] = pd.to_numeric(test["AcceptedCmp5"], downcast="float")

train["AcceptedCmps"] = train[["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4", "AcceptedCmp5"]].sum(axis = 1)

test["AcceptedCmps"] = test[["AcceptedCmp1","AcceptedCmp2","AcceptedCmp3","AcceptedCmp4", "AcceptedCmp5"]].sum(axis = 1)

train["NumCatalogPurchases"] = pd.to_numeric(train["NumCatalogPurchases"], downcast="float")
train["NumStorePurchases"] = pd.to_numeric(train["NumStorePurchases"], downcast="float")
train["NumWebPurchases"] = pd.to_numeric(train["NumWebPurchases"], downcast="float")
train["NumWebVisitsMonth"] = pd.to_numeric(train["NumWebVisitsMonth"], downcast="float")
train["NumDealsPurchases"] = pd.to_numeric(train["NumDealsPurchases"], downcast="float")

test["NumCatalogPurchases"] = pd.to_numeric(test["NumCatalogPurchases"], downcast="float")
test["NumStorePurchases"] = pd.to_numeric(test["NumStorePurchases"], downcast="float")
test["NumWebPurchases"] = pd.to_numeric(test["NumWebPurchases"], downcast="float")
test["NumWebVisitsMonth"] = pd.to_numeric(test["NumWebVisitsMonth"], downcast="float")
test["NumDealsPurchases"] = pd.to_numeric(test["NumDealsPurchases"], downcast="float")

train["Total_Purchases"] = train[["NumCatalogPurchases","NumStorePurchases","NumWebPurchases","NumWebVisitsMonth","NumDealsPurchases"]].sum(axis=1) 

test["Total_Purchases"] = test[["NumCatalogPurchases","NumStorePurchases","NumWebPurchases","NumWebVisitsMonth","NumDealsPurchases"]].sum(axis=1) 

train_clean = train.copy()

test_clean = test.copy()

train_clean = train_clean.drop('ID',axis=1)

test_clean = test_clean.drop('ID',axis=1)

train_clean = train_clean.drop('Recency',axis=1)

test_clean = test_clean.drop('Recency',axis=1)

train_clean = train_clean.drop('NumWebVisitsMonth',axis=1)

test_clean = test_clean.drop('NumWebVisitsMonth',axis=1)

train_clean = train_clean.drop('Complain',axis=1)

test_clean = test_clean.drop('Complain',axis=1)

train_clean['Income'] = train_clean['Income'].fillna(avg_income, axis=0)

test_clean['Income'] = test_clean['Income'].fillna(avg_income, axis=0)

train_clean['Income'] = train_clean['Income'].replace('',avg_income)

test_clean['Income'] = test_clean['Income'].replace('',avg_income)
 
train_clean = train_clean.drop('NumCatalogPurchases',axis=1)
train_clean = train_clean.drop('NumStorePurchases',axis=1)
train_clean = train_clean.drop('NumWebPurchases',axis=1)
train_clean = train_clean.drop('NumDealsPurchases',axis=1)

test_clean = test_clean.drop('NumCatalogPurchases',axis=1)
test_clean = test_clean.drop('NumStorePurchases',axis=1)
test_clean = test_clean.drop('NumWebPurchases',axis=1)
test_clean = test_clean.drop('NumDealsPurchases',axis=1)

train_clean.drop(['MntGoldProds','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts'],axis=1)

test_clean.drop(['MntGoldProds','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','AcceptedCmp2','MntWines','MntFruits','MntMeatProducts','MntFishProducts','MntSweetProducts'],axis=1)

feature = ['Age','Education','Marital_Status','Income','Kidhome','Teenhome','total_Mnt','AcceptedCmps','Total_Purchases']

X=train_clean[feature]

y = train_clean.Response

train_clean['Income'] = train_clean['Income'].ffill(axis=0) 

test_clean['Income'] = test_clean['Income'].ffill(axis=0)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=10)

k_fold = KFold(n_splits = 10, shuffle = True, random_state  =0)

nan_values = train_clean[train_clean.isna().any(axis=1)]

#Model ya3ml el prediction
rfc = RandomForestClassifier(n_estimators=30,
                             random_state=1)
"""**GridSearchCV**"""

max_depth_range = range(1,16)
param_grid = dict(max_depth=max_depth_range)

grid = GridSearchCV(rfc,
                    param_grid,
                    cv = 10,
                    scoring = 'accuracy')

grid.fit(X_train, y_train)


best_rfc = RandomForestClassifier(n_estimators=50,
                                  random_state=1,
                                  max_depth = 12)
best_rfc.fit(X_train, y_train)

rfc_pred = best_rfc.predict(X_test)
accuracy_train = metrics.accuracy_score(y_train, best_rfc.predict(X_train))
accuracy_test = metrics.accuracy_score(y_test, rfc_pred)

forest = RandomForestClassifier()
scoring = 'accuracy'
score = cross_val_score(forest, X, y, cv= k_fold, scoring=scoring)

tree = tree.DecisionTreeClassifier()
scoring = 'accuracy'
score = cross_val_score(tree, X, y, cv= k_fold, n_jobs=1, scoring=scoring)

forest = RandomForestClassifier()
scoring = 'accuracy'
score = cross_val_score(forest, X, y, cv= k_fold, scoring=scoring)
rfc_cv = GridSearchCV(forest, param_grid, cv=10,scoring = 'accuracy')
rfc_cv.fit(X, y)

knn = KNeighborsClassifier(n_neighbors = 5)
scoring = 'accuracy'
score = cross_val_score(knn, X, y, cv= k_fold, n_jobs=1, scoring=scoring)

k_scores = []
for i in range(10,40) :   
    knn = KNeighborsClassifier(n_neighbors = i)
    scoring = 'accuracy'
    score = cross_val_score(knn, X, y, cv = k_fold, scoring = scoring)  
    k_scores.append(np.mean(score))


param_grid = {'n_neighbors': np.arange(30, 40)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=10,scoring = 'accuracy')
knn_cv.fit(X, y)

train_res = pd.DataFrame.from_dict(rfc_cv.cv_results_) #We save the results into a dataframe

"""#Submission"""

submission=data['ID']
submission = pd.DataFrame({"Revenue": train_clean["Income"] - train_clean["Total_Purchases"], "ID":data['ID']})


submission.to_csv("submission.csv", index = False)