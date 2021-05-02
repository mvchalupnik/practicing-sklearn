import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.impute import SimpleImputer
# Explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer
# Now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer


################### Practice handling dataframes

#First, generate arrays to compose our dataframe
sizes = np.array([2, 3, 4, 2, 3, 3, 2])
fruit_names = np.array(["apple", "lemon", "apple", "pear", "pear", "apple", "apple"])
is_tasty = np.array([0, 1, 1, 0, 1, 1, 0])
id_num = np.array([1,2,3,4,5,6, 7])

#Practice creating a dataframe
df_1 = pd.DataFrame(data=np.c_[id_num, sizes, fruit_names, is_tasty], columns = np.array(["id_num", "sizes", "fruit_names", "is_tasty"]))
print("My first dataframe: \n")
print(df_1)

print("The types in df_1: {}".format(df_1.dtypes))

#Concatenate automatically changes the type of the column. To change it back:
df_1.sizes = df_1.sizes.astype(int)
df_1.id_num = df_1.id_num.astype(int)
df_1.is_tasty = df_1.is_tasty.astype(int)

print("The types in df_1: {}".format(df_1.dtypes))

#Practice merging dataframes
id_num_2 = np.array([2,1,3,4,5,6, 7])
weights = np.array([100, 200, 100, 100, 100, 100, np.NaN])

df_2 = pd.DataFrame(data=np.c_[id_num_2, weights], columns = np.array(["id_num", "weights"]))
print("\n\nMy second dataframe: \n")
print(df_2)

#Merge "on" will combine dataframes, keeping only the union of the two columns in the dataframe
#https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html
#If there are two of the same index value in one column, all permutations of other feature variables will be generated and kept (see documentation above)
#Can change the type of merge by changing "how", in this way can get intersection vs union, etc. (default is "inner")
df = df_1.merge(df_2, on="id_num")

print("\nMy merged dataframe")
print(df)

#Split into training and test sets
np.random.seed(10)
inds = np.random.choice(range(len(df)), size=3, replace=False)
train = df.loc[df.id_num.isin(inds)]
test = df.loc[~df.id_num.isin(inds)]


###Practicing selecting values from dataframe
#Find the null values:
print('Null values: {}'.format(df[df['weights'].isnull()]))
#Find the values where fruit_names is pear: 
print('Values where fruitnames is pear {}'.format(df[df.fruit_names=='pear']))
print('Values where fruitname is apple and weight is 100 {}'.format(df.loc[(df.weights == 100) & (df.fruit_names == 'apple')]))


#Leave-one-out mean encoding
#https://pkghosh.wordpress.com/2018/06/18/leave-one-out-encoding-for-categorical-feature-variables-on-spark/
#I won't do it for training and test separately, though that would be easy to implement with small modification
ws = [df.loc[fruit_names==i].dropna().sum().weights/(len(df.loc[fruit_names==i].dropna())) for i in np.unique(fruit_names)]
nms = [i for i in np.unique(fruit_names)]
meandict = dict(zip(nms, ws))
print("List of Leave-one-out means: {}".format(meandict))

#OR could use groupby and sum

#insert values into NaNs for each category in df
for i in np.unique(fruit_names):
	#df[row indexer, column indexer]
	df.loc[(df.weights.isnull()) & (df.fruit_names == i), 'weights'] = meandict[i]

print('df after replacement:')
print(df)




################## Model fitting

#Practice creating dummy variables for categorical data
df_dummy = pd.get_dummies(df).drop('fruit_names_pear', axis=1)
print(df.columns)
print(df)


################# Practice finding and replacing missing values
print("\n\nPractice finding and replacing missing values")
print(df)
df.loc[(df["id_num"] == 1) ,(df.index == 0)] = np.NaN
print(df)
print("Number of na values: ")
print(df.isna().sum())


# Drop rows with missing values
dropNArows = df.dropna(axis=0)
# Print percentage of rows remaining
print("Percentage of rows remaining after droping rows with na")
print(dropNArows.shape[0]/df.shape[0] * 100)

# Drop columns with missing values
print("Percentage of columns remaining after dropping columns with na")
# Drop columns with missing values
dropNAcols = df.dropna(axis=1)
# Print percentage of columns remaining
print(dropNAcols.shape[1]/df.shape[1] * 100)


##Impute 0 for missing data
df_filled = df.fillna(0)
print(df['sizes'].describe())
print(df_filled['sizes'].describe())

#Impute with mean
# Subset numeric features: numeric_cols
numeric_cols = df.select_dtypes(include=[np.number])
# Impute with mean
imp_mean = SimpleImputer(strategy='mean')
loans_imp_mean = imp_mean.fit_transform(numeric_cols)
# Convert returned array to DataFrame
loans_imp_meanDF = pd.DataFrame(loans_imp_mean, columns=numeric_cols.columns)
print("\n\nDataframe info after imputation of numeric columns with mean")
# Check the DataFrame's info
print(loans_imp_meanDF.info())

##Impute with IterativeImputer
#https://scikit-learn.org/stable/modules/impute.html#iterative-imputer
#at each step, a feature column is designated as output y and the other feature 
#columns are treated as inputs X. A regressor is fit on (X, y) for known y. 
#Then, the regressor is used to predict the missing values of y. This is done 
#for each feature in an iterative fashion, and then is repeated for max_iter imputation rounds. 
# Iteratively impute
imp_iter = IterativeImputer(max_iter=5, sample_posterior=True, random_state=123)
loans_imp_iter = imp_iter.fit_transform(numeric_cols)
# Convert returned array to DataFrame
loans_imp_iterDF = pd.DataFrame(loans_imp_iter, columns=numeric_cols.columns)
# Check the DataFrame's info
print("\n\nDataframe info after iterative imputation of numeric columns")
print(loans_imp_iterDF.info())


########## Replace outliers - Winsorization
# Print: before dropping
print("\n\nDetect and replace outliers")
df = df_filled
print(df)
numeric_cols = df.select_dtypes(include=[np.number])
print(numeric_cols.mean())
print(numeric_cols.median())
print(numeric_cols.max())

# Create index of rows to keep
print(numeric_cols)
idx = (np.abs(stats.zscore(numeric_cols)) < 3).all(axis=1)

# Concatenate numeric and categoric subsets
categoric_cols = df.select_dtypes(include=['object'])
ld_out_drop = pd.concat([numeric_cols.loc[idx], categoric_cols.loc[idx]], axis=1)

# Print: after dropping
print(ld_out_drop.mean())
print(ld_out_drop.median())
print(ld_out_drop.max())

###Another example of winsorization

# Winsorize numeric columns
debt_win = stats.mstats.winsorize(df['weights'], limits=[0.05, 0.05])

# Convert to DataFrame, reassign column name
debt_out = pd.DataFrame(debt_win, columns=['weights'])

# Print: after winsorize
print(debt_out.mean())
print(debt_out.median())
print(debt_out.max())



