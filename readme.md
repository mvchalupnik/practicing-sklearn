# Supervised and unsupervised learning practice
## Using sklearn in Python

These are a collection of short python scripts that use functions from the sklearn library for data processing, supervised, and unsupervised learning. Each of these scripts use built-in datasets from sklearn for easy access to data. The purpose of these scripts is pedagogical and detailed comments throughout the scripts explain important data science concepts. 

1. dataframe_practice.py  
	1. Practice loading CSV into dataframe. 
	2. Practice selecting sections of dataframes based on criterion.
	3. Practice creating dummy variables for categorical data
	4. Practice finding and replacing missing values/imputation
	5. Outlier detection  
3. feature_selection.py  
	1. Practice looking at the distribution of the train and test sets using sns.pairplot
	2. Practice log transformations for skewed data
	3. Plotting for outlier detection
	1. Practice SVR and RFECV for feature selection
	2. Use LarsCV for feature selection (wrapper method)
	3. Use RandomForest and ExtraTrees for feature selection (Tree method) 
2. linear_regression_practice.py  
	1. Make sklearn test dataset into a DataFrame
	1. Scatter plot and heatmap for exploratory data analysis
	2. Linear regression practice
	3. R^2 and RMSE calculation
	4. Cross-validation plot - CV score +/- standard error plotted against alpha
	5. Ridge and Lasso regression - plot for feature selection
	6. LassoCV practice
3. knn_practice.py  
	2. KNN practice
	3. Test and train split practice
4. metrics_practice.py  
	1. Mean, standard deviation
	2. Confusion matrix for KNN classifier
	3. ROC curve for Logistic regression as threshold varies
	4. GridSearchCV for hyperparameter tuning
	5. Hold-out set
5. decision_tree_practice.py  
	1. Decision Tree classifier
	2. Random search CV
	3. Bagging, Boosting ensemble methods
	4. Tune a random forest using GridSearchCV
6. pca_practice.py  
	1. Practice with PCA and SVD
	2. Practice PCA projection plot
	3. Practice with scree plot for PCA visualization
7. clustering_practice.py  
	1. Scaling/normalizing data
	2. KMeans clustering
	3. Hierarchical agglomerative clustering (the one with the dendrograms)
	4. Determining K by silhouette method and elbow method

