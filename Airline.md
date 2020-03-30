# Intro
Have you ever wondered what airplane features contributed the most to airplane satisfaction?
This was a big question i had. After searching through kaggle i have found a dataset that contained over 10000k observations and 25 rows. Since my main goal was to find out if people liked their flights or not, i decided to use multiple classification models to try and answer my solution.

#Importing libraries and Data
```import pandas as pd
import numpy as np

#plotting
import seaborn as sns
import matplotlib.pyplot as plt
#preprocessing
from sklearn.preprocessing import StandardScaler
#modeling
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier

#metircs
from sklearn import metrics
from sklearn.metrics import confusion_matrix

```
My models i am going to run are knn, decision trees, and logestic regression. I also decided to use random forrest and grid search as my ensamble methods

# Data Cleaning
For my data cleaning i decided to standardize my data using a standard scalar method.
```ss = StandardScaler()
column_names = numerical_cols.columns
numerical_cols = pd.DataFrame(ss.fit_transform(numerical_cols))
numerical_cols.columns = column_names```
I also dropped unwanted columns and dropped na values i did not need.

# Analysis
All my models i ran followed the same work flow. First i ran basic models that i did not tune at all. 
```knn = KNeighborsClassifier()
knn.fit(X_train, y_train)```
After running my basic models i tried to improve them with some ensemble methods like gridsearch and random forest. After running all my models, random forest turned out the best with an accuracy of 90%
```plt.figure(figsize=(10,10))
plt.bar(height=sorted(dictrf.values()),x=dictrf.keys())
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel('Features',fontsize=25)
plt.ylabel('Satisfaction',fontsize=25)
plt.title('Feature Importance with Satisfaction',fontsize=20)``` After plotting my feature importances i learned that plane class and also type of travel were some of the most important features.

# Conclusion
After my analysis, if airline companies want to improve satisfaction, they need to work on their service in economy plus. They should add better inflight services and also make the boarding process more simple.