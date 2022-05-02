#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 15:27:13 2022

@author: weirukuo
"""
## This file includes python code used in bankruptcy_prediction.ipynb. 
## Each figure can be saved as a file by uncommenting the line below starting with plt.savefig

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
# Decision tree model
from sklearn.tree import DecisionTreeClassifier
# Random forest model
from sklearn.ensemble import RandomForestClassifier
# Model Evaluation
from sklearn.metrics import classification_report,confusion_matrix

# Remove warnings. Mainly for warnings from matplotlib or seaborn due to version change
import warnings
warnings.filterwarnings('ignore')

# Read the data
data = pd.read_csv('bankruptcy_data.csv')
# Check the first few lines of the dataframe
data.head()
# Select the columns of interest
data=data[['Bankrupt?',' ROA(B) before interest and depreciation after tax',' Operating Gross Margin',' Current Ratio',' Debt ratio %',' Working Capital to Total Assets',' Cash Flow to Sales',' Cash Flow to Liability',' Gross Profit to Sales',' Liability to Equity']]

# Check no missing data
data.isnull().sum()
# See columnsinformation including data type and total number
data.info()
# Bankrupt? is the label. Plot the count of the label class
sns.countplot(x='Bankrupt?',data=data,palette='Set1')
#plt.savefig('bankrupt_.png',bbox_inches='tight',dpi=100)

# Since it's an imbalanced dataset, we reduce the number of the higher class (Bankrupt?=0) and redefine the dataframe
# Get the count where Bankrupt?=1
print(len(data[data['Bankrupt?']==1]))
# Randomly select the same size from Bankrupt?=0
df_not_bankrupt=data[data['Bankrupt?']==0].sample(220)
# Combine dataframes of Bankrupt=1 and 0
data=data[data['Bankrupt?']==1].append(df_not_bankrupt)
# Sort by the original index to mix the data
data=data.sort_index()

# Check statistics
data.describe()

# Two histogram of ROA in one graph based on two classes
plt.figure(figsize=(10,6))
data[data['Bankrupt?']==1][' ROA(B) before interest and depreciation after tax'].hist(alpha=0.5,color='blue',bins=30,label='Bankrupt?=1')
data[data['Bankrupt?']==0][' ROA(B) before interest and depreciation after tax'].hist(alpha=0.5,color='red',bins=30,label='Bankrupt?=0')
plt.legend()
plt.xlabel('ROA(B)')
#plt.savefig('roa_histogram_by_bankrupt_class.png',bbox_inches='tight',dpi=100)

# A joint plot including scatter plot and histogram showing relationship between two columns
plt.figure(figsize=(11,7))
sns.jointplot(x=' ROA(B) before interest and depreciation after tax',y=' Operating Gross Margin',data=data,color='purple')
#plt.savefig('roa_and_operating_gross_margin_joint_plot.png',bbox_inches='tight',dpi=100)

# A heatmap showing correlations between all columns
plt.figure(figsize=(11,7))
sns.heatmap(data.drop('Bankrupt?',axis=1).corr(),annot=True)
#plt.savefig('corr_heatmap.png',bbox_inches='tight',dpi=100)

# Showing a scatter plot with regression lines for two columns
plt.figure(figsize=(11,7))
sns.lmplot(x=" Cash Flow to Sales",y=" Cash Flow to Liability",hue="Bankrupt?",data=data,palette='Set1')
plt.xlim(0.6710,0.6720) # Zoom in
plt.ylim(0,1)
#plt.savefig('cash_flow_to_Sales_and_cash_flow_to_liability_lmplot.png',bbox_inches='tight',dpi=100)

# Box plot of each column in one graph
plt.figure(figsize=(11,4))
sns.boxplot(data=data.drop('Bankrupt?',axis=1), orient="h",palette='Set2')
#plt.savefig('box_plot.png',bbox_inches='tight',dpi=100)

# Distribution plot of one column
plt.figure(figsize=(11,7))
sns.distplot(data[' Working Capital to Total Assets'])
#plt.('working_capital_to_total_assets_distribution.png',bbox_inches='tight',dpi=100)

# Delete two columns
del data[' Gross Profit to Sales']
del data[' Cash Flow to Sales']

# Split up the data into a training set and a test set
X=data.drop('Bankrupt?',axis=1)
y=data['Bankrupt?']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Decision tree model
# Create an instance of DecisionTreeClassifier() called dtree and fit it to the training data
dtree= DecisionTreeClassifier()
dtree.fit(X_train,y_train)
# Predict based on the test data
predictions= dtree.predict(X_test)
# Print model evaluation metric
print("Decision tree")
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))

# Same steps with random forests
# Create an instance of RandomForestClassifier() called rfc and fit it to the training data
rfc= RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
rfc_pred=rfc.predict(X_test)
print("Random forest")
print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))

