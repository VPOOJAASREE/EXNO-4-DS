# EXNO:4-DS

```
NAME: V. POOJAA SREE
REG.: 212223040147

```

# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:

```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")

df.head()
df.dropna()

```

# Output:

![1](https://github.com/user-attachments/assets/e527de5a-3a32-4a4e-8b64-1a6a9c9f4042)


```
max_vals=np.max(np.abs(df[['Height']]))
max_vals
max_vals1=np.max(np.abs(df[['Weight']]))
max_vals1
print("Height =",max_vals)
print("Weight =",max_vals1)

```

# Output:

![2](https://github.com/user-attachments/assets/1911ac73-e97f-43c6-897e-7ee2c39318ed)


```
df1=pd.read_csv("/content/bmi.csv")

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)

```

# Output:

![3](https://github.com/user-attachments/assets/24643e56-59c7-46dc-8875-4c025a40b6f5)


```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)

```

# Output:

![4](https://github.com/user-attachments/assets/4b0ad2b8-cd5a-4628-a2a0-484e46ef4791)


```
from sklearn.preprocessing import Normalizer
df2=pd.read_csv("/content/bmi.csv")
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df

```

# Output:

![5](https://github.com/user-attachments/assets/2148f19d-e768-40b9-b790-f49ba90129a3)


```
df3=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3

```

# Output:

![6](https://github.com/user-attachments/assets/e6135651-8ddd-4167-839a-0311e47f15c2)


```
df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()

```

# Output:

![7](https://github.com/user-attachments/assets/a1edc76f-abc7-48ed-b3c6-c647ca515e11)


```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data

```

# Output:

![8](https://github.com/user-attachments/assets/7400ce5b-7939-4da9-92bb-419c1abcca97)


```
data.isnull().sum()

```

# Output:

![9](https://github.com/user-attachments/assets/5ab2984e-293f-45e4-a267-ec5698b56a62)


```
missing=data[data.isnull().any(axis=1)]
missing

```

# Output:

![10](https://github.com/user-attachments/assets/d17071d5-2f8b-4798-a60f-8bcc3bcd49aa)


```
data2=data.dropna(axis=0)
data2

```

# Output:

![11](https://github.com/user-attachments/assets/caa643c5-1f0b-4fb8-ae7e-704ba0d4b6a4)


```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])

```

# Output:

![12](https://github.com/user-attachments/assets/b6bfcc76-6167-42ca-b147-f88ee2b1db7e)


```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs

```

# Output:

![13](https://github.com/user-attachments/assets/a2c13dc5-3435-4673-9517-120000bd1936)


```
data2

```

# Output:

![14](https://github.com/user-attachments/assets/5d487f22-1bfa-49a7-82e1-521623e414b8)


```
new_data=pd.get_dummies(data2, drop_first=True)
new_data

```

# Output:

![15](https://github.com/user-attachments/assets/d894b1df-d313-42be-aa7a-4c4145a6fb74)


```
columns_list=list(new_data.columns)
print(columns_list)

```

# Output:

![16](https://github.com/user-attachments/assets/76ba70d5-78a1-48e5-9d61-0cb2c032d10a)


```
features=list(set(columns_list)-set(['SalStat']))
print(features)

```

# Output:

![17](https://github.com/user-attachments/assets/dff44c8f-4112-45b3-8bdb-6966d764eda1)


```
y=new_data['SalStat']
print(y)

```

# Output:

![18](https://github.com/user-attachments/assets/810a1ea9-7491-4ed4-9c2b-11a663c32345)


```
x=new_data[features].values
print(x)

```

# Output:

![19](https://github.com/user-attachments/assets/55b32787-07fd-480b-9f46-94ed46b3cc8a)


```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)

```

# Output:

![20](https://github.com/user-attachments/assets/ac6f3f3e-208e-426a-a56d-139957f62b3b)


```
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)

```

# Output:

![21](https://github.com/user-attachments/assets/65c34fd6-fa54-40b7-b239-90b6b7fb995d)


```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data= {
    'Feature1' : [1,2,3,4,5],
    'Feature2' : ['A','B','C','A','B'],
    'Feature3' : [0,1,1,0,1],
    'Target' : [0,1,1,0,1]
}
df=pd.DataFrame(data)

X=df[['Feature1','Feature3']]
y=df['Target']

selector = SelectKBest(score_func= mutual_info_classif, k=1)
X_new=selector.fit_transform(X,y)

selected_feature_indices = selector.get_support(indices=True)
selected_features = X.columns[selected_feature_indices]

print("Selected Features:", selected_features)

```

# Output:

![22](https://github.com/user-attachments/assets/93607372-c638-47d7-9f9d-54f2c4192d4f)




# RESULT:
       # INCLUDE YOUR RESULT HERE
