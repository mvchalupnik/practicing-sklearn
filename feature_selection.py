from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from scipy.stats import boxcox

from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LarsCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor



#Boston property value dataset
data1 = load_boston()
print(data1.keys())
print(data1.DESCR)

boston = pd.DataFrame(data=data1['data'], columns=data1['feature_names'])
boston['MEDV'] = data1['target']

y = boston['MEDV'].values
X = boston.drop('MEDV', axis=1).values


############### Look at variable distributions when train_test_split is used
# Create `loan_data` subset: loan_data_subset
boston_subset = boston[['MEDV','CRIM','CHAS']]

# Create train and test sets
trainingSet, testSet = train_test_split(boston_subset, test_size=0.2, random_state=123)

# Examine pairplots
plt.figure()
sns.pairplot(trainingSet, hue='CHAS', palette='RdBu')
plt.show()

plt.figure()
sns.pairplot(testSet, hue='CHAS', palette='RdBu')
plt.show()

############## Log and Power transformations
# Subset data
crim = boston['CRIM']

# Histogram and kernel density estimate
plt.figure()
sns.distplot(crim)
plt.show()

# Box-Cox transformation; lmbda=0 is log transform; lmbda=0.5 is square root 
crim_log = boxcox(crim, lmbda=0)

# Histogram and kernel density estimate
plt.figure()
sns.distplot(crim_log)
plt.show()


############# Plotting for outlier detection
# Univariate and multivariate boxplots
fig, ax =plt.subplots(1,2)
sns.boxplot(y=boston['CRIM'], ax=ax[0])
sns.boxplot(x='CRIM', y='MEDV', data=boston, ax=ax[1])
plt.show()



###
# Create correlation matrix and print it
cor = boston.corr()
print(cor)

# Correlation matrix heatmap
plt.figure()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

# Correlation with output variable
cor_target = abs(cor["MEDV"])

# Selecting highly correlated features
best_features = cor_target[cor_target > 0.5]
print(best_features)

#####################Use SVR and RFECV as a wrapper for feature selection
#SVR is like SVM except for y as a continuous rather than categorical variable. It doesn't seem like
#it is super commonly used, but here I guess we are using it for feature selection under I believe
#the wrapper method

#Also see example from here: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html

# Instantiate estimator and feature selector
X = boston.drop('MEDV', axis=1)

#why use an SVR here? 
svr_mod = SVR(kernel="linear")
#Recursive feature elimination cross-validation: selects the best subset of features for the supplied estimator
#by fitting the model multiple times, then at each step removing the weakest features. cef_ or feature_importances_ 
#can be used to determine feature strength
feat_selector = RFECV(svr_mod, cv=5)

# Fit
feat_selector = feat_selector.fit(X, y)

# Print support and ranking
print(feat_selector.support_) #False means feature can be eliminated?? (what is the algorithm process? still not sure) 
print(feat_selector.ranking_) #Ranking
print(X.columns)

################## Use LarsCV for hyperparameter optimization (wrapper)
#LARS works by starting with one variable, increasing its corresponding coefficient, and when the residual has 
#correlation with some other variable as much as it does the variable you started with, adding that in 
#and increasing in the joint least squares direction (find through fitting just those variables??), iterating. 
#This is a feature selection method because at the end you will find some coefficients are 0. 

# Instantiate
lars_mod = LarsCV(cv=5, normalize=False)

# Fit
feat_selector = lars_mod.fit(X, y)

# Print r-squared score and estimated alpha
print(lars_mod.score(X, y))
print(lars_mod.alpha_)

################# Using a RandomForestRegressor for feature selection (Tree-based methods)
#The way feature importance is calculated: Create trees. Then take one feature variable, 
#permute it randomly (shuffle), then rerun the observations through the trees. Calculate 
#the rate of misclassification. The % increase of misclassification rate gives the feature importance
#https://link.springer.com/article/10.1023/A:1010933404324

# Instantiate
rf_mod = RandomForestRegressor(max_depth=2, random_state=123, 
              n_estimators=100, oob_score=True)

# Fit
rf_mod.fit(X, y)

# Print
print(boston.columns)
print(rf_mod.feature_importances_)


################ Using Extra Trees method for feature selection
# Instantiate
xt_mod = ExtraTreesRegressor()

# Fit
xt_mod.fit(X, y)

# Print
print(boston.columns)
print(xt_mod.feature_importances_)