from sklearn.metrics import classification_report
#Alternatively can indiividually import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn.utils import resample
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.preprocessing import scale

cancer = load_breast_cancer()
print(cancer.DESCR)
print(cancer.keys())
print(cancer.target_names)
print(cancer.feature_names)

X = cancer.data
y = cancer.target

############## Scaling/normalization using scikit's "scale"

# Scale the features: X_scaled
X_scaled = scale(X)

# Print the mean and standard deviation of the unscaled features
print("Mean of Unscaled Features: {}".format(np.mean(X))) 
print("Standard Deviation of Unscaled Features: {}".format(np.std(X)))

# Print the mean and standard deviation of the scaled features
print("Mean of Scaled Features: {}".format(np.mean(X_scaled))) 
#mean is close to 0
print("Standard Deviation of Scaled Features: {}".format(np.std(X_scaled)))
#standard deviation is close to 1


###########Confusion matrix and classification report for KNN classification

knn = KNeighborsClassifier(n_neighbors=8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(type(y_pred))

#first argument in confusion_matrix is always true

##               | predicted cancer| predicted benign
##  ----------------------------------------------
## actual cancer | True positive   | False negative
## actual benign | False positive  | True negative
print(confusion_matrix(y_test, y_pred))


#Accuracy = (tp + tn)/(tp + fn + fp + tn) 
#Precision = tp/(tp + fp) # Out of all predicted cancer cases, which are really cancer 
#Recall or Sensitivity = tp/(tp + fn) : Out of all actual cancer cases, which were correctly predicted as cancer
#F1score = 2 * (precision*recall)/(precision + recall) 
#Support = The number of samples of True response in that class
print(classification_report(y_test, y_pred))


######### Resampling to help with class imbalance 
#Warning: when resampling, always make sure to use train/test sets!!
print('Number of target variables equal to 1: {}'.format(len(y[y==1])))
print('Number of target variables equal to 0: {}'.format(len(y[y==0])))
print('Classes arent terribly imbalanced, but well practice resampling on this anyway')


#Create dataframe
df = pd.DataFrame(data=X_train, columns=cancer['feature_names'])
df['tumor'] = y_train


yestumor = df[df['tumor'] ==1] #357 elements
notumor = df[df['tumor'] ==0] #212 elements

# Upsample minority and combine with majority
notumors_upsampled = resample(notumor, replace=True, n_samples=len(yestumor), random_state=123)
upsampled = pd.concat([yestumor, notumors_upsampled])

# Downsample majority and combine with minority
tumors_downsampled = resample(yestumor, replace = False,  n_samples = len(notumor), random_state = 123)
downsampled = pd.concat([tumors_downsampled, notumor])

# Upsampled feature matrix and target array
X_train_up = upsampled.drop('tumor', axis=1)
y_train_up = upsampled['tumor']

# Instantiate, fit, predict
loan_lr_up = LogisticRegression(solver='liblinear')
loan_lr_up.fit(X_train_up, y_train_up)
upsampled_y_pred = loan_lr_up.predict(X_test)

# Print evaluation metrics
print('\n\nUpsampled metrics: ')
print("Confusion matrix:\n {}".format(confusion_matrix(y_test, upsampled_y_pred)))
print("Classification report: {}".format(classification_report(y_test, upsampled_y_pred)))


#Can repeat for downsampled tumors
# Downsampled feature matrix and target array
X_train_down = downsampled.drop('tumor', axis=1)
y_train_down = downsampled['tumor']

# Instantiate, fit, predict
loan_lr_down = LogisticRegression(solver='liblinear')
loan_lr_down.fit(X_train_down, y_train_down)
downsampled_y_pred = loan_lr_down.predict(X_test)
# Print evaluation metrics
print('\n\nDownscaled metrics: ')
print("Confusion matrix:\n {}".format(confusion_matrix(y_test, downsampled_y_pred)))
print("Classification report: {}".format(classification_report(y_test, downsampled_y_pred)))


# USEFUL FOR DEBUGGING: 
#quit(0)

########### Logistic regression (binary)
#Recall principles of logistic regression: https://youtu.be/hjrYrynGWGA
logreg = LogisticRegression() #by default, logistic regression threshold is 0.5 (knn also has threshold)
#The threshold just sets where to draw the line on the sigmoid function between classifying as 0 or 1 (since in 
#general your function will return some number on the continuum between 0 and 1)
logreg.fit(X_train, y_train)
ypred = logreg.predict(X_test)

#ROC "Recieving-Operator Characteristic" Curve: Set of points you get when varying threshold and considering
#False positive and True positive rates
y_pred_prob = logreg.predict_proba(X_test)[:,1] #predict_proba returns an array with probabilities; here second column is 
												#probabilities of predicted labels being 1
#fpr: false positive rate, tpr: true positive rate
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
#True positive rate is also known as recall: TPR = tp/(tp + fn)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label="Logistic Regression")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.show()


################## Compute area under ROC curve: another metric for classification models: 
logreg = LogisticRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
logreg.fit(X_train, y_train)
y_pred_prob = logreg.predict_proba(X_test)[:,1]
roc_auc_score = roc_auc_score(y_test, y_pred_prob)
print("ROC AUC score is {}".format(roc_auc_score))

#Alternatively, can compute using cross-validation
cv_scores = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')
print("Alternative calculation of ROC AUC score is {}".format(cv_scores))

################# Hyperparameter Grid search
param_grid = {'n_neighbors': np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, param_grid, cv=5) #shouldn't number of folds be determined by the size of param_grid
knn_cv.fit(X,y)
print("Tuned Logistic Regression Parameters: {}".format(knn_cv.best_params_)) #n_neighbors for KNN 
print("Best score is {}".format(knn_cv.best_score_)) #Accuracy for KNN? 

#Note that LogisticRegression has a hyperparameter called C, where C controls the inverse of regularization strength:
# a large C can lead to an overfit model, while a small C leads to an underfit model 
# C stands for Controlled variable decay, where
# Logistic Regression Loss function = sum_i ( - (y_i log(a_i) - (y_i-1) log(a_i-1)) + (1/C) sum(theta_j))
# (where theta_j are all of your logistic regression coefficients)


############### Hold-out set
#Good practice to split data into "test" (or "holdout") and "training" sets at the beginning
#Use training set for k-fold cross-validation

# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=42)

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit it to the training data
logreg_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))

# Predict on the test set and compute metrics
#I guess it automatically takes the best hyperparameters from the training set
#and uses those when you call predict? 
y_pred = logreg_cv.predict(X_test)
ac = logreg_cv.score(X_test, y_test)
#Default "score" for logistic regression is accuracy, see: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

##mse = mean_squared_error(y_test, y_pred)

print("Tuned C and penalty: {}".format(logreg_cv.best_params_))
print("Tuned Accuracy: {}".format(ac))

##print("Tuned MSE: {}".format(mse))

