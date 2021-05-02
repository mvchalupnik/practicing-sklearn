from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

# Import necessary modules
from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV 

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
##from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier



cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

X_train, X_test, y_train, y_test = \
train_test_split(X, y, test_size=0.3, random_state=21, stratify=y)

############### Randomized Search Cross-Validation
#Optimize hyperparameters via a randomized search cross-validation
# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()


# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X,y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
#Can also use GridSearchCV


############### Practicing Bagged Trees
#Bagged models tend to decrease model variance (but increase bias)
# Instantiate bootstrap aggregation model
bagged_model = BaggingClassifier(n_estimators=50, random_state=123) #default is Tree, I believe
#I wonder how this is different from sklearn.ensemble.RandomForestClassifier? 

bagged_model.fit(X_train, y_train)
bagged_pred = bagged_model.predict(X_test)
print("Bagged Tree accuracy: {}".format(accuracy_score(y_test, bagged_pred)))

############### Practice boosted models
#Boosted models tend to decrease model bias (but increase variance)
# Boosting model
#If not base_estimator specified, base_estimator is DecisionTreeClassifier with depth 1
boosted_model = AdaBoostClassifier(n_estimators=50, random_state=123)

# Fit
boosted_model_fit = boosted_model.fit(X_train, y_train)

# Predict
boosted_pred = boosted_model_fit.predict(X_test)

# Print model accuracy
print("Boosted Tree accuracy: " + str(accuracy_score(y_test, boosted_pred)))



############### Tune a Random Forest using GridSearchCV
#max_features lets your model decide how many features to use

# Create the hyperparameter grid
param_grid = {"criterion": ["gini"], "min_samples_split": [2, 10, 20], 
              "max_depth": [None, 2, 5, 10],"max_features": [10, 20, 30]}

# Instantiate classifier and GridSearchCV, fit
loans_rf = RandomForestClassifier()
rf_cv = GridSearchCV(loans_rf, param_grid, cv=5)
fit = rf_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned Random Forest Parameter: {}".format(rf_cv.best_params_))
print("Tuned Random Forest Accuracy: {}".format(rf_cv.best_score_))


