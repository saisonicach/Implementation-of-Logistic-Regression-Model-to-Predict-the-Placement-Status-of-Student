# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values.
 

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SAI SONICA .CH
RegisterNumber: 212219040130

import pandas as pd
df=pd.read_csv("/content/drive/MyDrive/Colab Notebooks/Semster 2/Intro to ML/Placement_Data.csv")
df.head()
df.tail()
df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()
df1.isnull().sum()
#to check any empty values are there
df1.duplicated().sum()
#to check if there are any repeted values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

df1["gender"] = le.fit_transform(df1["gender"])
df1["ssc_b"] = le.fit_transform(df1["ssc_b"])
df1["hsc_b"] = le.fit_transform(df1["hsc_b"])
df1["hsc_s"] = le.fit_transform(df1["hsc_s"])
df1["degree_t"] = le.fit_transform(df1["degree_t"])
df1["workex"] = le.fit_transform(df1["workex"])
df1["specialisation"] = le.fit_transform(df1["specialisation"])
df1["status"] = le.fit_transform(df1["status"])
df1
x=df1.iloc[:,:-1]
x
y = df1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.09,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
#liblinear is library for large linear classification
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))
*/
```

## Output:
## Original data(first five columns):
![image](https://user-images.githubusercontent.com/79306169/174434211-71a933c8-cbaf-4c4d-ab4d-623e6dde5e60.png)
## Data after dropping unwanted columns(first five):
![image](https://user-images.githubusercontent.com/79306169/174434214-d566c42c-7c03-4bb9-9d1c-af4c4ce1d517.png)
## Checking the presence of null values:
![image](https://user-images.githubusercontent.com/79306169/174434222-f78aad4e-97e3-48c9-be18-5153951a5d1e.png)
## Checking the presence of duplicated values:
![image](https://user-images.githubusercontent.com/79306169/174434236-d9ee5b5d-b96e-4fd2-b20d-0dba64a41d1e.png)
## Data after Encoding:
![image](https://user-images.githubusercontent.com/79306169/174434242-6588f677-e9cd-4b8b-8b9f-f1d34606f17a.png)
## X Data:
![image](https://user-images.githubusercontent.com/79306169/174434247-d38bb3ed-d324-4722-9973-36da67702d33.png)
## Y Data:
![image](https://user-images.githubusercontent.com/79306169/174434260-0039a1a5-6d3c-49a8-a7ee-b2f2306dc421.png)
## Predicted Values:
![image](https://user-images.githubusercontent.com/79306169/174434281-2b920b8f-321b-4088-923f-5e454114bbbd.png)
## Accuracy Score:
![image](https://user-images.githubusercontent.com/79306169/174434286-aaecf677-8471-4dc2-a51b-692c221ded5e.png)
## Confusion Matrix:
![image](https://user-images.githubusercontent.com/79306169/174434290-fb058691-deec-4b1e-aae1-93507eabb0fc.png)
## Classification Report:
![image](https://user-images.githubusercontent.com/79306169/174434297-4b637843-96d9-49ca-af7f-35d3c90319f8.png)
## Predicting output from Regression Model:
![image](https://user-images.githubusercontent.com/79306169/174434303-f709de31-d832-481f-9b9c-8590db2ed07a.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
