import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

dataset = pd.read_csv('../data/diabetes.csv')
# print ( len(dataset) )

#we don't want zeros in some specific columns, terefore replacing with average
zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']

for column in zeros:
	dataset[column] = dataset[column].replace(0, np.NaN)
	mean = int(dataset[column].mean(skipna = True))
	dataset[column] = dataset[column].replace(np.NaN, mean)

#splitting
x = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state = 0, test_size = 0.2)


#scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.tranform(X_test)

#defining K
import math
k = int(math.sqrt(len(Y_test)))

if (k % 2 == 0):
	k -= 1

#definig the model
classifier = KNeighborsClassifier(n_neighbors = k, p = 2, metric = 'euclidean')

classifier.fit(X_train, Y_train)

#predicting
Y_pred = classifier.predict(X_test)

#Evaluating the model
cm = confusion_matrix(Y_test, Y_pred)
#print(cm)
print(f1_score(Y_test, Y_pred))

print(accuracy_score(Y_test, Y_pred))