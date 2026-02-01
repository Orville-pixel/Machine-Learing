import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

#Question-A1
A = np.array([1,2,3,4,5,6,7,8,9])
B = np.array([2,3,4,5,6,7,8,9,0])

def dot_product_by_me(X,Y):
    sum=0
    if len(X) == len(Y):
      for i in range(len(X)):          
        sum += X[i]*Y[i] 
    return sum

def euclidean_norm_by_me(X):
   sum=0
   for i in X:
      sum += i*i
   return math.sqrt(sum)
print("Dot product of A and B by python package: ",np.dot(A,B))
print("Dot product of A and B by my function:",dot_product_by_me(A,B))
print("The Euclidean norm of A and B by python function is: ",np.linalg.norm(A),np.linalg.norm(B))
print("Euclidean norm of A and B by my function: ",euclidean_norm_by_me(A),euclidean_norm_by_me(B))

#Question-A2
path = r"C:\Users\rvija\Desktop\amrita\Semester-4\Machine Learning\Lab Session Data.xlsx"
data = pd.read_excel(path, "Purchase data")
X = data.iloc[:, 1:4].values
payment = data.iloc[:,4].values
print("Features: ",X)
print("Expenditure of customer: ",payment)
class_1 = X[payment < 250]
class_2 = X[payment >= 250]
centroid_1 = class_1.mean(axis=0)
centroid_2 = class_2.mean(axis=0)
spread_1 = class_1.std(axis=0)
spread_2 = class_2.std(axis=0)
distance = np.linalg.norm(centroid_1 - centroid_2)
print("Mean of class 1: ",centroid_1)
print("Standard deviation of class 1: ",spread_1)
print("Mean of class 2: ",centroid_2)
print("Standard deviation of class 2: ",spread_2)
print("The inter-class distance between class 1 and 2: ",distance)


#Question-A3
candies_feature = data.iloc[:,1].values
print("Candies feature: ",candies_feature)
histogram_candies_data = np.histogram(candies_feature)
mean_candies_feature = np.mean(candies_feature)
variance_candies_feature = np.var(candies_feature)
plt.hist(candies_feature)
plt.xlabel("Candies Purchased")
plt.ylabel("Frequency")
plt.show()
print("Histogram counts:", histogram_candies_data)
print("Mean of candies feature:", mean_candies_feature)
print("Variance of candies feature:", variance_candies_feature)

#Question-A4
def minkwosi_distance(feature_1,feature_2,p):
    diff=0
    result=0
    for i in range(len(feature_1)):
        if feature_1[i] > feature_2[i]:
           diff = feature_1[i] - feature_2[i]
        else:
           diff = feature_2[i] - feature_1[i]
        result += diff ** p
    return result ** (1/p)

p_vector = [1,2,3,4,5,6,7,8,9,10]
print(p_vector)

mangoes_feature_vector = data.iloc[:,2].values
milk_packets_feature_vector = data.iloc[:,3].values

minkwosi_dist_vector = []
for i in range(1,11):
   minkwosi_dist = minkwosi_distance(mangoes_feature_vector,milk_packets_feature_vector,i)
   minkwosi_dist_vector.append(minkwosi_dist)

print(minkwosi_dist_vector)
plt.plot(p_vector, minkwosi_dist_vector)
plt.xlabel("p(1-10)")
plt.ylabel("distance")
plt.show()


#Question-5
print("Minkwosi distance through my function: ",minkwosi_distance(mangoes_feature_vector,milk_packets_feature_vector,2))
print("Minkwosi distance through python function: ",minkowski(mangoes_feature_vector, milk_packets_feature_vector, 2))

#Question-6
X = data.iloc[:, 1:4].values
y = np.where(data.iloc[:, 4].values < 250, 0, 1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
print("Training features: ", X_train)
print("Testing features: ", X_test)
print("Training labels: ", y_train)
print("Testing labels: ", y_test)



#7
"""neigh3 = KNeighborsClassifier(n_neighbors=3)
neigh3.fit(X_train, y_train)"""

#8
"""accuracy3 = neigh3.score(X_test, y_test)
print("Accuracy(k=3):", accuracy3)"""


#9
"""X_pred=neigh3.predict(X_test)
print("Predicted labels:", X_pred)"""

#11
"""neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, y_train)
accuracy = neigh.score(X_test, y_test)
print("Accuracy(k=1):", accuracy)
k=range(1,8)
accuracies=[]
for i in k:
    n=KNeighborsClassifier(n_neighbors=i)
    n.fit(X_train, y_train)
    acc=n.score(X_test, y_test)
    accuracies.append(acc)

plt.plot(k,accuracies,marker='o')
plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.title("kNN Accuracy vs k")
plt.grid(True)
plt.show()"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


X = data.iloc[:, 1:4].values
y = np.where(data.iloc[:, 4].values < 250, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

print("Confusion Matrix (Training Data):")
print(cm_train)

print("\nConfusion Matrix (Testing Data):")
print(cm_test)

print("\nTraining Performance Metrics:")
print("Accuracy:", accuracy_score(y_train, y_train_pred))
print("Precision:", precision_score(y_train, y_train_pred))
print("Recall:", recall_score(y_train, y_train_pred))
print("F1-Score:", f1_score(y_train, y_train_pred))

print("\nTesting Performance Metrics:")
print("Accuracy:", accuracy_score(y_test, y_test_pred))
print("Precision:", precision_score(y_test, y_test_pred))
print("Recall:", recall_score(y_test, y_test_pred))
print("F1-Score:", f1_score(y_test, y_test_pred))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


X = data.iloc[:, 1:4].values
y = np.where(data.iloc[:, 4].values < 250, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)

def confusion_matrix_custom(y_true, y_pred):
    TP = FP = TN = FN = 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
    return TP, FP, TN, FN

def accuracy_custom(TP, FP, TN, FN):
    return (TP + TN) / (TP + TN + FP + FN)

def precision_custom(TP, FP):
    return TP / (TP + FP) if (TP + FP) != 0 else 0

def recall_custom(TP, FN):
    return TP / (TP + FN) if (TP + FN) != 0 else 0

def fbeta_score_custom(precision, recall, beta):
    if precision + recall == 0:
        return 0
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)

TP, FP, TN, FN = confusion_matrix_custom(y_test, y_test_pred)

print("Custom Confusion Matrix Values:")
print("TP:", TP)
print("FP:", FP)
print("TN:", TN)
print("FN:", FN)

accuracy = accuracy_custom(TP, FP, TN, FN)
precision = precision_custom(TP, FP)
recall = recall_custom(TP, FN)
f1_score_custom = fbeta_score_custom(precision, recall, beta=1)

print("\nCustom Performance Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1_score_custom)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from math import sqrt


X = data.iloc[:, 1:4].values
y = np.where(data.iloc[:, 4].values < 250, 0, 1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

package_predictions = neigh.predict(X_test)
package_accuracy = neigh.score(X_test, y_test)

print("Package kNN Predictions:", package_predictions)
print("Package kNN Accuracy:", package_accuracy)

def euclidean_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i]) ** 2
    return sqrt(distance)

def knn_custom(X_train, y_train, test_vector, k):
    distances = []
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], test_vector)
        distances.append((dist, y_train[i]))
    distances.sort(key=lambda x: x[0])
    k_nearest_labels = [label for (_, label) in distances[:k]]
    predicted_class = Counter(k_nearest_labels).most_common(1)[0][0]
    return predicted_class

custom_predictions = []
for test_vector in X_test:
    pred = knn_custom(X_train, y_train, test_vector, k=3)
    custom_predictions.append(pred)

custom_predictions = np.array(custom_predictions)

correct = np.sum(custom_predictions == y_test)
custom_accuracy = correct / len(y_test)

print("\nCustom kNN Predictions:", custom_predictions)
print("Custom kNN Accuracy:", custom_accuracy)

print("\nComparison of kNN Classifiers:")
print("Package kNN Accuracy:", package_accuracy)
print("Custom kNN Accuracy:", custom_accuracy)



   


        
           
       
       
       
        
   
   







