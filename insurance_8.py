#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 13:09:14 2019

@author: sbk
"""


import pandas as pd
import numpy as np

data = pd.ExcelFile(r"C:\Users\SUJITH KUMAR\Downloads\Insurance-master\Data-All.xlsx")
print(data.sheet_names)
df1 = data.parse('Sheet1')
print(df1.info())


# Replace the spaces in clumn names to underscore and coonvert column header to lower case

df1.columns = df1.columns.str.replace(' ','_')

df1.columns = df1.columns.str.lower()


#df1.tail()
#df1.describe()


# Count the number of unique rows in all the columns
for col in df1.columns:
    print(col, df1[col].nunique())


print(df1.index)


# ## Data Preprocessing
def Data_Preprocess(df1):
    

    # Removal of top of four rows
    df2 = df1[4:]
    df2.shape
    # Features to be removed from the data
    print('Removing features...')
    print(df2.columns[[0, 1, 2, 5, 8, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 23, 25, 28, 29, 30, 31, 32, 33]])
    df2.drop(df2.columns[[0, 1, 2, 5, 8, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21, 23, 25, 28, 29, 30, 31, 32, 33]], axis=1, inplace=True)

    # cause of injury types and counts
    df2.case_of_injury_group.value_counts()


    # Standardization of floating value features using MinMax method
    df2["experience_years"] = (df2["experience_years"]-df2["experience_years"].min()) / (df2["experience_years"].max()-df2["experience_years"].min())
    df2["age_at_accident_date"] = (df2["age_at_accident_date"]-df2["age_at_accident_date"].min()) / (df2["age_at_accident_date"].max()-df2["age_at_accident_date"].min())
    return df2

def Data_Engineering(df2):
    # Label encoder to convert objects to numerical

    from sklearn.preprocessing import LabelEncoder

    LE = LabelEncoder()

    df2['market'] = LE.fit_transform(df2['market'])
    df2['type_of_injury'] = LE.fit_transform(df2['type_of_injury'])
    df2['case_of_injury_group'] = LE.fit_transform(df2['case_of_injury_group'])
    df2['nature_of_injury'] = LE.fit_transform(df2['nature_of_injury'])
    df2['body_part_group'] = LE.fit_transform(df2['body_part_group'])
    df2['occupation'] = LE.fit_transform(df2['occupation'].astype(str))
    df2['accident_state'] = LE.fit_transform(df2['accident_state'])
    df2['sex'] = LE.fit_transform(df2['sex'])
    df2['classcode'] = LE.fit_transform(df2['classcode'].astype(str))


    print(df2.head())

    # Standardization of labelencoded values

    df2["nature_of_injury"] = (df2["nature_of_injury"]-df2["nature_of_injury"].min()) / (df2["nature_of_injury"].max()-df2["nature_of_injury"].min())
    df2["accident_state"] = (df2["accident_state"]-df2["accident_state"].min()) / (df2["accident_state"].max()-df2["accident_state"].min())
    df2["type_of_injury"] = (df2["type_of_injury"]-df2["type_of_injury"].min()) / (df2["type_of_injury"].max()-df2["type_of_injury"].min())
    df2["occupation"] = (df2["occupation"]-df2["occupation"].min()) / (df2["occupation"].max()-df2["occupation"].min())
    df2["classcode"] = (df2["classcode"]-df2["classcode"].min()) / (df2["classcode"].max()-df2["classcode"].min())
    df2["body_part_group"] = (df2["body_part_group"]-df2["body_part_group"].min()) / (df2["body_part_group"].max()-df2["body_part_group"].min())
    df2["market"] = (df2["market"]-df2["market"].min()) / (df2["market"].max()-df2["market"].min())
    df2["sex"] = (df2["sex"]-df2["sex"].min()) / (df2["sex"].max()-df2["sex"].min())

    
    print(df2.head())
    return df2


df2 = Data_Preprocess(df1)
df2 = Data_Engineering(df2)

# Feature Correlation
print(df2.corr(method ='pearson'))


# Divide data into features and target
features = df2.drop('case_of_injury_group', 1)
labels = df2[['case_of_injury_group']]



def feature_contribution(features, labels):

    # Determining the featues that most contribute to the cause of injury

    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import chi2


    # Feature extraction
    test = SelectKBest(score_func=chi2, k=4)
    fit = test.fit(features, labels)

    # Summarize scores
    np.set_printoptions(precision=3)
    print(fit.scores_)

    features_1 = fit.transform(features)
    # Summarize selected features
    print(features_1[0:10,:])

feature_contribution(features, labels)


# Splitting data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]
print('Number of train and test data:')
print(number_of_train, number_of_test)


print(features.shape)
print(labels.shape)
print(features.isnull().values.any())



from keras.utils import to_categorical
#one-hot encode target column
train_y_2 = to_categorical(Y_train)

#vcheck that target column has been converted
train_y_2[0:5]



from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping 

def build_model():
    #create model
    model = Sequential()

    #get number of columns in training data
    n_cols = X_train.shape[1]

    #add layers to model
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(11, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


K_model = build_model()
early_stopping_monitor = EarlyStopping(patience=10)
K_model.fit(X_train, train_y_2, epochs=10, validation_split=0.2, callbacks=[early_stopping_monitor])





