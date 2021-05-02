# Import module
from sklearn.decomposition import TruncatedSVD
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression

cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
print(cancer.keys())
print(cancer.feature_names)
print("There are {} features".format(len(cancer.feature_names)))

X_scaled = scale(X)
y_scaled = scale(y)



############ Practicing principle component analysis
#Use eigenvalue decomposition to decompose the covariance matrix 
#Helps reduce multicollinearity and dimensionality

# PCA
pca = PCA(n_components=3)

# Fit and transform
principalComponents = pca.fit_transform(X_scaled)

# Print ratio of variance explained
#After transformation, we keep the three (as specified in n_components) transformed features with the highest 
#transformed covariance eigenvalues
print("PCA explained variance ratios: {}".format(pca.explained_variance_ratio_))
print("PCA components: {}".format(pca.components_))
print("The PCA component vector has size {}, because there are {} vectors with length {}".format(pca.components_.shape, \
	pca.components_.shape[0], pca.components_.shape[1]))



########## Practicing singular value decomposition

# SVD
svd = TruncatedSVD(n_components=2)

# Fit and transform
principalComponents = svd.fit_transform(X_scaled)

# Print ratio of variance explained
print("SVD explained variance ratios: {}".format(svd.explained_variance_ratio_))

################ Practicing creating a PCA plot for visualization
#first project data points onto the two PCA axes. Note that principle components coming out of svd are already normalized.
df_pca = pd.DataFrame(np.transpose(np.array([np.dot(X_scaled, svd.components_[0]), np.dot(X_scaled, svd.components_[1]),\
	cancer.target])), \
	columns=['Principal component 1', 'Principal component 2', 'Target value'])




targets = [0, 1]
colors = ['r', 'b'] #two colors for the two principle value axes? 

fig = plt.figure()
ax = plt.gca()

# For loop to create plot
print(list(zip(targets, colors)))
for target, color in zip(targets, colors):
    indicesToKeep = df_pca['Target value'] == target
    ax.scatter(df_pca.loc[indicesToKeep, 'Principal component 1']
               , df_pca.loc[indicesToKeep, 'Principal component 2']
               , c = color
               , s = 50)

# Legend    
ax.legend(targets)
ax.grid()
plt.title('Two Component PCA/SVD')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.show()

########### Practicing making a Scree plot for visualizing principal components
#A scree plot is just a line plot of the principal components (fairly simple) to show how much variance (or some other metric)
#is explained by each component
# List principal components names
# PCA
pca = PCA(n_components=10)

# Fit and transform
principalComponents = pca.fit_transform(X_scaled)

principal_components = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10']

# Create a DataFrame
pca_df = pd.DataFrame({'Variance Explained': pca.explained_variance_ratio_,
             'PC':principal_components})

plt.figure() 
# Plot DataFrame
sns.barplot(x='PC',y='Variance Explained', 
           data=pca_df, color="c")
plt.show()


# Plot cumulative variance
#Note: not setting n_components returns all the principal components from the trained model
pca = PCA()
principalComponents = pca.fit_transform(X_scaled)
cumulative_var = np.cumsum(pca.explained_variance_ratio_)*100
plt.figure()
plt.plot(cumulative_var,'k-o',markerfacecolor='None',markeredgecolor='k')
plt.title('Principal Component Analysis',fontsize=12)
plt.xlabel("Principal Component",fontsize=12)
plt.ylabel("Cumulative Proportion of Variance Explained",fontsize=12)
plt.show()

########## PCA with train and test, also visualize with heatmap 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state = 42)

pca = PCA()
pca.fit(X_train)

# Transform train and test
X_trainPCA = pca.transform(X_train)
X_testPCA = pca.transform(X_test)

# Import
from sklearn.linear_model import LinearRegression

# Instantiate, fit, predict
LinRegr = LinearRegression()
LinRegr.fit(X_trainPCA, y_train)
predictions = LinRegr.predict(X_testPCA)

# The coefficients
print('Coefficients: \n', LinRegr.coef_)

# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_test, predictions))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, predictions))

# Correlation matrix
X_trainPCA = pd.DataFrame(X_trainPCA)
diab_corrPCA = X_trainPCA.corr()

# Generate correlation heatmap
ax = sns.heatmap(diab_corrPCA, center=0, square=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
plt.show()

# Print correlations
print(diab_corrPCA)
#Will see that it removes all multicollinearity, though it may not improve r^2 or RMSE