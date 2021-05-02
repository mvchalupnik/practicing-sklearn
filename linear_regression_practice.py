from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso, LassoCV


######## Import data and exploratory analysis

#Boston property value dataset
data1 = datasets.load_boston()
print(data1.keys())
print(data1.DESCR)

boston = pd.DataFrame(data=data1['data'], columns=data1['feature_names'])
boston['MEDV'] = data1['target']

print(boston.head())
print(list(boston.columns))

#drop target: MEDV: Median value of owner-occupied homes in $1000's
#Need ".values" in order to get rid of the pesky index that usually appears, and just make it a straight numpy array
y = boston['MEDV'].values
X = boston.drop('MEDV', axis=1).values

X_rooms = X[:,5] #fifth column
print(type(X_rooms), type(y)) 


##Need to reshape to a (n, 1) array instead of just (n) sized array in order to fit with python's LinearRegression
print("\n\n Shape of y: " + str(y.shape))
print("\n\n Shape of x: " + str(X_rooms.shape))
y = y.reshape(-1, 1)
X_rooms = X_rooms.reshape(-1,1)
print("\n\n Shape of y: " + str(y.shape))
print("\n\n Shape of x: " + str(X_rooms.shape))

#Plot a simple scatter plot for one data
plt.scatter(X_rooms, y)
plt.ylabel('Value of house / 1000 ($)')
plt.xlabel('Number of rooms')
plt.show()

#Plot a heatmap for all the data
plt.figure()
sns.heatmap(boston.corr(), square=True, cmap='RdYlGn')
plt.show()

##################### Fitting a simple linear regression model
reg = LinearRegression()
reg.fit(X_rooms, y)
prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1, 1)

plt.figure()
plt.scatter(X_rooms, y, color='blue')
plt.plot(prediction_space, reg.predict(prediction_space), color='black', linewidth=3)
plt.show()

################### use training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)


#r^2 : metric used to quantify linear regression performance. Equal to 1 - sum(error^2)/variance(y)
print("R^2 value: " + str(reg_all.score(X_test, y_test)))
#"Note you will generally never use linear regression out of the box like this, you will most likely use regularization"

#RMSE: Root mean squared error, another commonly used metric for linear regression
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

#################### Cross-validation
#split into k folds, hold one for testing, use the remaining to train the model. 
#then repeat by exchanging each of the k folds for testing
#Useful to prevent overfitting: fitting that is specific to the specific sample and not generalizable
reg = LinearRegression()
cv_scores = cross_val_score(reg, X_rooms, y, cv=5, scoring='r2') #for 5 fold cross-validation
#score reported is R^2 -> this is the default score for linear regression models
#more info: https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
#Note that the R^2 can be negative, if the fit is worse than just fitting a horizontal line

print(cv_scores)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

######################## Ridge regression or L2 regularization
#Regularization can help prevent overfitting by penalizing large coefficients
#change loss function to be sum(error^2) + alpha * sum((x_i)^2)
X_train, X_test, y_train, y_test = train_test_split(X_rooms, y, test_size = 0.3, random_state = 42)
ridge = Ridge(alpha = 0.1, normalize = True) #Setting normalize equal to True makes sure all coefficents are on the same scale
#Can (and should) use instead: RidgeCV for built-in cross-validation
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge.score(X_test, y_test)
#similar to LinearRegression above


####################### Lasso regression or L1 regularization
#change loss function to be sum(error^2) + alpha * sum(abs(x_i))
names = boston.drop('MEDV', axis=1).columns
lasso = Lasso(alpha=0.1)
lasso_coef = lasso.fit(X, y).coef_
#I guess a multivariable linear regression just works the same as a single variable linear regression? 
plt.figure()
_ = plt.plot(range(len(names)), lasso_coef)
_ = plt.xticks(range(len(names)), names, rotation=60)
_ = plt.ylabel('Coefficients')
plt.show()

###################### Combining Ridge regression with cross-validation
def display_plot(cv_scores, cv_scores_std):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(alpha_space, cv_scores)

    std_error = cv_scores_std / np.sqrt(10)

    ax.fill_between(alpha_space, cv_scores + std_error, cv_scores - std_error, alpha=0.2)
    ax.set_ylabel('CV Score +/- Std Error')
    ax.set_xlabel('Alpha')
    #we want our R^2 to be close to 1, so taking the maximum highlights the largest R^2
    ax.axhline(np.max(cv_scores), linestyle='--', color='.5')
    ax.set_xlim([alpha_space[0], alpha_space[-1]])
    ax.set_xscale('log')
    plt.show()


# Setup the array of alphas and lists to store scores
alpha_space = np.logspace(-4, 0, 50)
ridge_scores = []
ridge_scores_std = []

# Create a ridge regressor: ridge
ridge = Ridge(normalize=True)

# Compute scores over range of alphas
for alpha in alpha_space:

    # Specify the alpha value to use: ridge.alpha
    ridge.alpha = alpha
    
    # Perform 10-fold CV: ridge_cv_scores
    ridge_cv_scores = cross_val_score(ridge, X, y, cv=10)
    
    # Append the mean of ridge_cv_scores to ridge_scores
    ridge_scores.append(np.mean(ridge_cv_scores))
    
    # Append the std of ridge_cv_scores to ridge_scores_std
    ridge_scores_std.append(np.std(ridge_cv_scores))

# Display the plot
display_plot(ridge_scores, ridge_scores_std)

############## LassoCV regression
y = boston['MEDV'].values
X = boston.drop('MEDV', axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)

# Instantiate cross-validated lasso, fit
lasso_cv = LassoCV(alphas=None, cv=10, max_iter=10000) #maximum number of iterations run to solve for optimal alpha
lasso_cv.fit(X_train, y_train)

# # Instantiate lasso, fit, predict and print MSE
lasso = Lasso(alpha = lasso_cv.alpha_)
lasso.fit(X_train, y_train)
print(mean_squared_error(y_true=y_test, y_pred=lasso.predict(X_test)))
