# EXNO:4-DS
## AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

## ALGORITHM:
STEP 1:Read the given Data.<br>
STEP 2:Clean the Data Set using Data Cleaning Process.<br>
STEP 3:Apply Feature Scaling for the feature in the data set.<br>
STEP 4:Apply Feature Selection for the feature in the data set.<br>
STEP 5:Save the data to the file.

## FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

## FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:<br>
1.Filter Method<br>
2.Wrapper Method<br>
3.Embedded Method<br>

## CODING AND OUTPUT:
<h3>Developed By: Sethukkarasi C</h3>
<h3>Register Number: 212223230201</h3>

```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("income.csv",na_values=[ " ?"])
data
```

![output1](/o1.png)

```
data.isnull().sum()
```

![output2](/o2.png)

```
missing=data[data.isnull().any(axis=1)]
missing
```

![output3](/o3.png)

```
data2=data.dropna(axis=0)
data2
```

![output4](/o4.png)

```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

![output5](/o5.png)

```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```

![output6](/o6.png)

```
data2
```

![output7](/o7.png)

```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

![output8](/o8.png)

```
columns_list=list(new_data.columns)
print(columns_list)
```

![output9](/o9.png)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

![output10](/o10.png)

```
y=new_data['SalStat'].values
print(y)
```

![output11](/o11.png)

```
x=new_data[features].values
print(x)
```

![output12](/o12.png)

```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```

![output13](/o13.png)

```
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

![output14](/o14.png)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

![output15](/o15.png)

```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

![output16](/o16.png)

```
data.shape
```

![output17](/o17.png)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

![output18](/o18.png)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

![output19](/o19.png)

```
tips.time.unique()
```

![output20](/o20.png)

```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

![output21](/o21.png)

```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

![output22](/o22.png)

## RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed.
