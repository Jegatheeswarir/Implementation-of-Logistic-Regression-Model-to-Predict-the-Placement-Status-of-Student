# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1. Start

Step 2. Load the California Housing dataset and select the first 3 features as input (X) and target variables (Y) (including the target price and another feature).

Step 3. Split the data into training and testing sets, then scale (standardize) both the input features and target variables.

Step 4. Train a multi-output regression model using Stochastic Gradient Descent (SGD) on the training data.

Step 5. Make predictions on the test data, inverse transform the predictions, calculate the Mean Squared Error, and print the results.

Step 6. Stop

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: JEGATHEESWARI R
RegisterNumber:  212223230092
*/
import pandas as pd
df=pd.read_csv('Placement_Data.csv')
df.head()
```
![Screenshot 2025-03-25 110420](https://github.com/user-attachments/assets/15e5a205-9d9c-4f74-a089-e49227463fd2)
```
data1 = df.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
```

![Screenshot 2025-03-25 110501](https://github.com/user-attachments/assets/582429b6-d7a5-4758-8e48-3cea2b76b43f)

```
data1.isnull().sum()
```

![Screenshot 2025-03-25 110544](https://github.com/user-attachments/assets/b5525864-5bda-45ec-8fde-b66c2c7685f5)

```
data1.duplicated().sum()
```

![Screenshot 2025-03-25 110622](https://github.com/user-attachments/assets/4fd3a531-a912-4279-97dc-7912a55652c5)

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_p"] = le.fit_transform(data1["degree_p"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["etest_p"] = le.fit_transform(data1["etest_p"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
```

![Screenshot 2025-03-25 110659](https://github.com/user-attachments/assets/d85a9837-7034-4e30-b01c-b500db849dbc)

```
x = data1.iloc[:,:-1]
x
```

![Screenshot 2025-03-25 110739](https://github.com/user-attachments/assets/78249ac7-184f-4c0d-90ca-35b49f146503)

```
y = data1["status"]
y
```

![Screenshot 2025-03-25 110821](https://github.com/user-attachments/assets/d18264b3-c73d-41ef-ad08-026867719768)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
```

![Screenshot 2025-03-25 110857](https://github.com/user-attachments/assets/9ebfa867-a5ad-4d06-bc76-4ee864c39324)

```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```

![Screenshot 2025-03-25 110938](https://github.com/user-attachments/assets/f6c3d64d-e5e4-425e-9727-f361756d4ce4)

```
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```

![Screenshot 2025-03-25 111012](https://github.com/user-attachments/assets/909c07a9-d48a-4683-9091-198c87dfe69b)

```
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
```

![Screenshot 2025-03-25 111051](https://github.com/user-attachments/assets/96f8137f-e7c8-43fc-acd9-e377d4d7e07f)

```
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

![Screenshot 2025-03-25 111130](https://github.com/user-attachments/assets/24763088-26e9-4c4a-a24a-3e9f09d138a6)

## Output:


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
