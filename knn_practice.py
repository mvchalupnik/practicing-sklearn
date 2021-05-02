from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


plt.style.use('ggplot')
iris = datasets.load_iris()
# Iris built-in dataset

print(type(iris))
#a bunch is similar to a dictionary, contains key value pairs
print(iris.keys())
print(iris.DESCR)

print(iris.data.shape) #print the dimensions of the keys

########### exploratory data analysis
x = iris.data
y = iris.target
#in this case x is already nicely available, but in general to extract x from a dataframe, can use: 
# df.drop('party', axis=1) for some dataframe df where 'party' is e.g. the y variable you want to take out
df = pd.DataFrame(x, columns=iris.feature_names)
_ = pd.plotting.scatter_matrix(df, c=y, figsize=[8,8], s=150, marker = 'D')
#can use countplot for binary data sets
plt.show()



########### KNN classification
knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(iris['data'], iris['target'])


#pass two arguments: features as numpy array, and labels as numpy array
#requires features must take on continuous values
X_new = np.array([[5.6, 2.8, 3.9, 1.1], [5.7, 2.6, 3.8, 1.3], [4.7, 3.2, 1.3, 0.2]])
prediction = knn.predict(X_new)

print(X_new.shape)
#3 observations, 4 features

print('Prediction: {}'.format(prediction))
#this will print out your model's prediction

############# Testing accuracy
#Test_size specifies what proportion of the original data is used for the test set
#Random_state sets a seed for the random number generator to split the data
#Stratify will make sure the proportion of X labels in the training and test set matches the original set. 
X_train, X_test, y_train, y_test = \
train_test_split(x, y, test_size=0.3, random_state=21, stratify=y)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Test set predictions:\\n {}".format(y_pred))
print("knn score {}".format(knn.score(X_test, y_test)))

#Model complexity: 
#Larger k = smoother decision boundary = less complex model, too large k will cause "underfitting"
#(generally, complex models run the risk of noise in the specific data that you have) 
#Smaller k = more complex model = Too few nearest neighbors will cause "overfitting"

##Test and plot for different values of number of neighbors
plt.figure() #create a new figure
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    #accuracy: the number of correct predictions divided by total number of data points
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

