import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Data Collection and Data Preprocessing
sonar_data=pd.read_csv('sonar data.csv',header=None)
print(sonar_data.head())
print(sonar_data.shape)
print(sonar_data.describe())
print(sonar_data.isnull().sum())
print(sonar_data[60].value_counts())

print(sonar_data.groupby(60).mean())

#separating data and labels
X=sonar_data.drop(columns=60,axis=1)
Y=sonar_data[60]

print(X)
print(Y)

#Training and Test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=1)
# stratify=Y ensures that the proportion of different classes is maintained in both the training and testing datasets, mirroring the distribution in the original dataset Y. 
print(X.shape,X_train.shape,X_test.shape)  
print(X_train)
print(Y_train)

# Model Training --> Logistic Regression 
model = LogisticRegression()
model.fit(X_train,Y_train)

# Model evalution
# accuracy on training data
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(X_train_prediction,Y_train)
print(training_data_accuracy)           #here the accuracy of our model is about 83% for X_train data. 

# accuracy on test data
X_test_prediction=model.predict(X_test)
test_data_accuracy=accuracy_score(X_test_prediction,Y_test)
print(test_data_accuracy)


##### Making a Predictive System #####

#this input_data is a data for Rock as we have copied it from sonar data 
input_data=(0.0286,0.0453,0.0277,0.0174,0.0384,0.0990,0.1201,0.1833,0.2105,0.3039,0.2988,0.4250,0.6343,0.8198,1.0000,0.9988,0.9508,0.9025,0.7234,0.5122,0.2074,0.3985,0.5890,0.2872,0.2043,0.5782,0.5389,0.3750,0.3411,0.5067,0.5580,0.4778,0.3299,0.2198,0.1407,0.2856,0.3807,0.4158,0.4054,0.3296,0.2707,0.2650,0.0723,0.1238,0.1192,0.1089,0.0623,0.0494,0.0264,0.0081,0.0104,0.0045,0.0014,0.0038,0.0013,0.0089,0.0057,0.0027,0.0051,0.0062)
# changing the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data) 
print(input_data_as_numpy_array.shape)              #shape of this one is (60,)

#reshape the np array as we are predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
# this line of code takes the NumPy array input_data_as_numpy_array and transforms it into a 2-dimensional array (a matrix) with exactly one row. The number of columns will be determined by the total number of elements in the original array because of use of -1. 
print(input_data_reshaped.shape)                #shape of this one is (1,60)

prediction=model.predict(input_data_reshaped)
print(type(prediction))         # type of prediction is numpy array
print(prediction)           #it will print ['R'] 

if (prediction[0] =='R'):
    print('The object is a Rock')
else:
    print('The object is a mine')    


