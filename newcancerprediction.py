import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
# print(breast_cancer_dataset)

data_frame=pd.DataFrame(breast_cancer_dataset.data,columns=breast_cancer_dataset.feature_names)
#convention of Dataframe is (rows required,columns required)
# print(data_frame.head())
#adding the 'target' column to the data frame.
data_frame['label']=breast_cancer_dataset.target
#convention is dataframe[key]=values respective for each key,like dictionary
#print last 5 rows
# print(data_frame.tail())
#number of rows and columns
# print(data_frame.shape)
#means we have the data for 569 different people
#getting for the info ablut data
# print(data_frame.info())
# print(data_frame.isnull().sum())
# print(data_frame.describe())
# print(data_frame['label'].value_counts())
#1-->benign
#2-->Malignant,these tumors are those which could spread to other parts of the body,more riskier than
#the benign which are not that much riskier, when compared to malignant cases.
# print(data_frame.groupby('label').mean())
x=data_frame.drop(columns='label',axis=1)
y=data_frame['label']
# print(x)
# print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
# print(x.shape,x_train.shape,x_test.shape)
#use either linear regression or logistic
model=LogisticRegression()
model.fit(x_train,y_train)
knn=KNeighborsClassifier()
knn.fit(x_train,y_train)
xk_train_prediction=knn.predict(x_train)
training_kdata_accuracy=knn.score(x_train,y_train)
print("Training accuracy: ",training_kdata_accuracy)
xk_test_prediction=knn.predict(x_test)
testing_kdata_accuracy=accuracy_score(y_test,xk_test_prediction)
print("Testing accuracy: ",testing_kdata_accuracy)

#accuracy on training
x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(y_train,x_train_prediction)

print('Accuracy on training data=',training_data_accuracy)

#accuracy on testing
x_test_prediction=model.predict(x_test)
testing_data_accuracy= accuracy_score(y_test,x_test_prediction)
print('Accuracy on testing data=',testing_data_accuracy)
#vvimpt
a=input().split()
b = []
for i in a:
    b.append(float(i))

# print(b)
# print(type(b))
input_data=tuple(b)
# print(h)

# input_data=
# input_data=(17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189)

input_data_as_numpy_array =np.asarray(input_data)
#reshape the numpy array for 1 data point
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshaped)
print(prediction)

if(prediction[0]==0):
    print("The Breast cancer is Malignant")
else:
    print("The Breast Cancer is Benign")

