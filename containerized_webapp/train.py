import pandas as pd 
import numpy as np
import re
import string
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def preprocess(input_name):
	"""
	clean and split name into individual non-alphabetic chars
	"""
	# Remove non-alphabetic characters
	input_name = re.sub('[^A-Za-z]+', '', input_name.lower())

	return ' '.join(list(input_name))


if __name__ == '__main__':


	## loading data
	df = pd.read_csv("name_gender.csv")
	print("Dataset shape", df.shape)
    
	# Clean input name
	df['clean_name'] = df['name'].apply(preprocess)

	# Convert gender to binary (male or not)
	df['male'] = (df['gender'].map({'M':1,'F':0}))
	df.drop(columns=['gender'], inplace = True)

	X = df['clean_name']
	y = df['male']

	# Extract Feature With CountVectorizer
	# Change token pattern to consider 1 letters
	cv = CountVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=None)
	X = cv.fit_transform(X) # Fit the Data

	# # Apply SMOTE to handle imbalanced dataset
	# from imblearn.over_sampling import SMOTE
	# from collections import Counter
	# smote = SMOTE()
	# # fit predictor and target variable
	# x_smote, y_smote = smote.fit_resample(X, y)
	# print('Original dataset shape', Counter(y))
	# print('Resample dataset shape', Counter(y_smote))
	# X = x_smote
	# y = y_smote

	# Split into training and set data
	X_train, X_test, y_train, y_test = (train_test_split(X, y, test_size=0.2, random_state=111))


	# Use apply stratified kfolds validation and 

	

	## Using Classifier
	# clf = LogisticRegression()
	clf = XGBClassifier(max_depth=5)
	clf.fit(X_train,y_train)

	print("Train acc:", clf.score(X_train,y_train)*100)
	print("Test acc:", clf.score(X_test,y_test)*100)
	# print(clf.predict(cv.transform([preprocess("Nico")])))
	
	## save vectorizer and model
	with open('model/logistic_clf.pkl', 'wb') as f:
    	    pickle.dump((cv,clf), f)
