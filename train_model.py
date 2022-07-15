from math import gamma
import pandas as pd 
import numpy as np
import re
import string
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE
from collections import Counter


def preprocess(input_name):
	"""Clean and split input_name into individual non-alphabetic chars"""
	# Remove non-alphabetic characters
	input_name = re.sub('[^A-Za-z]+', '', input_name.lower())

	return ' '.join(list(input_name))

def smote_data(X,y):
	"""Return balanced number data points per class"""
	# Apply SMOTE to handle imbalanced dataset (as seen from EDA)
	smote = SMOTE()
	# Fit predictor and target variable
	x_smote, y_smote = smote.fit_resample(X, y)
	print('Original dataset shape', Counter(y))
	print('Resample dataset shape', Counter(y_smote))
	return x_smote, y_smote

def hypertune(model, X_train, X_test, y_train, y_test, smote_ind = False):
	"""Return best model after hypertuning parameters"""
	# Apply SMOTE to Train set
	# X_train, y_train = smote_data(X_train, y_train)

	# Define the search space
	param_grid = { 
		# Shrinks  weights to make the boosting process more conservative
		"learning_rate": [0.0001,0.001, 0.01, 0.1, 1, 2] ,
		# The deeper the tree, the more complex the model becomes
		"max_depth": range(3,21,2),
		# Minimum loss reduction required to make a split
		"gamma": [i/10.0 for i in range(0,5)],
		# Percentage of columns to be randomly sampled for each tree
		"colsample_bytree": [i/10.0 for i in range(3,10)],
		# l1 regularization, the higher the values/weights the simpler the model
		"reg_alpha": [1e-5, 1e-2, 0.1, 1, 10],
		# l2 regularization, the higher the values/weights the simpler the model
		"reg_lambda": [1e-5, 1e-2, 0.1, 1, 10]}

	# Cross validation
	cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=143)

	# Using RandomizedSearchCV to randomly sample parameters from defined 
	# parameter space (tradeoff between accuracy and time/resources)
	gs_xg = RandomizedSearchCV(estimator=model, 
                           param_distributions=param_grid, 
                           n_iter=48,
                           scoring='accuracy', 
                           n_jobs=-1, 
                           cv=cv)

	gs_xg.fit(X_train, y_train)

	print(f"best parameters {gs_xg.best_params_} "
				"with train {gs_xg.score(X_train, y_train)} and test "
				"{gs_xg.score(X_test, y_test)}")
	# print(f'best score: {gs_xg.best_params_}')
	
	return gs_xg

def main():

	## Load data
	df = pd.read_csv("name_gender.csv")
	# print("Dataset shape", df.shape)
    
	# Apply pre-processing to features
	df['clean_name'] = df['name'].apply(preprocess)

	# Apply preprocessing to taget variable
	# Convert gender to binary (male or not)
	df['male'] = (df['gender'].map({'M':1,'F':0}))
	df.drop(columns=['gender'], inplace = True)

	# Define features and target variables
	X = df['clean_name']
	y = df['male']

	# Extract Feature With CountVectorizer
	# Change token pattern to consider 1 character as token
	cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None)
	X = cv.fit_transform(X) 
	
	# Split into training and set data
	X_train, X_test, y_train, y_test = (train_test_split(X, y, test_size=0.2, random_state=111))

	## Define Classifier
	clf = XGBClassifier()


	# Hypertune model and 
	final_model = hypertune(clf, X_train, X_test, y_train, y_test, True)

	
	# save vectorizer and model
	with open('model/final_model.pkl', 'wb') as f:
    	    pickle.dump((cv,final_model), f)


if __name__ == '__main__':
	main()
