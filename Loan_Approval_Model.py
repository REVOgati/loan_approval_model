#!/usr/bin/env python
# coding: utf-8

# This is a project that uses Machine Learning modelling in order to predict the likelyhood of an applicant's loan approval.
#     
# -The dataset was fetched from Kaggle and was posted by "Amit Parjapat" 
# -For the modelling process, I used Pandas Library, Numpy Library, Scikit Learn .
# 
# GOALS: Have a descriptive analysis of the data and variable relationships
#        Develop a predicitve model on plausibility of loan approval.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn


# In[ ]:


raw_data = pd.read_csv('loan_prediction.csv') #importing the csv file to our code,gets converted to dataframe 


# -Data Manipulation and Analysis should be a simple and clear process.
# 
# -I will start with description and general visualization of the data in order to understand the nature of loan applicants.
# 
# -Then,after cleaning the data, I will use the data gathered to develop Machine Learning models that give the best predictions based on this data.

# In[ ]:


raw_data.head() #Defaulty prints the first 5 entries


# In[ ]:


raw_data.info() #Returns description of dataframe such as datatypes


# In[ ]:


columns_names =  raw_data.columns.tolist()
print(columns_names)


# In[ ]:


raw_data.describe() #Gives Statistical Information about dataframe


# In[ ]:


#Checking for duplication of rows
raw_data.Loan_ID.value_counts(dropna=False)


# It is visible that there are 614 unique entries = Total number of rows

# In[ ]:


#Remove column irrelevant to analysis
raw_data.drop('Loan_ID', axis=1, inplace=True)


# In[ ]:


#General analysis of the data and visualization.
Gender_distribution= raw_data['Gender'].value_counts()
print(Gender_distribution)
Gender_distribution.plot(kind='pie',autopct='%2.2f%%',title="Visualization of Loan applicants based on gender")


# It is seen that Male are the highest number of Applicants compared to Female.

# In[ ]:


raw_data['Married'].value_counts().plot(kind='pie',autopct='%2.2f%%',title="Visualization of Loan applicants based on Marital Status")


# It is seen that Married people are the highest number of Applicants compared to Single people.

# In[ ]:


raw_data['Dependents'].value_counts().plot(kind='pie',autopct='%2.2f%%',title="Visualization of Loan applicants based on number of Dependants")


# A dependant is a person who relies on another as a primary source of income. 
# 

# In[ ]:


raw_data['Education'].value_counts().plot(kind='pie',autopct='%2.2f%%',title="Visualization of Loan applicants based on Education status")


# More applicants are graduates compared to non-graduates.
# There are several reasons this may be. However, one assumption may be student loans that lay heavily on a % of graduates.

# In[ ]:


raw_data['Self_Employed'].value_counts().plot(kind='pie',autopct='%2.2f%%',title="Visualization of Loan applicants based on Employment status")


# Above shows that people who are self_Employed hardly apply for loans unlike those that are not. It gives an assumption that self_employed citizen are likely to be more financially stable than those depedent on being employed.
# 
# Below , I pick the columns of Self_Employed and Loan_Status to evaluated the relationship:

# In[ ]:



#I will create 4 lists with instances where one is either self_employed or not against whether the loan status is approved = Y or not =N
self_employed_yes = raw_data['Self_Employed'].tolist().count('Yes')
self_employed_no = raw_data['Self_Employed'].tolist().count('No')
loan_status_y = raw_data['Loan_Status'].tolist().count('Y')
loan_status_n = raw_data['Loan_Status'].tolist().count('N')

print('self_employed_yes = ' ,self_employed_yes )
print('self_employed_no = ' ,self_employed_no )
print('loan_status_y = ' ,loan_status_y )
print('loan_status_n = ' ,loan_status_n )

self_employed_yes_loan_y = raw_data.loc[(raw_data['Self_Employed'] == 'Yes') & (raw_data['Loan_Status'] == 'Y'), 'Self_Employed'].tolist()
self_employed_yes_loan_n = raw_data.loc[(raw_data['Self_Employed'] == 'Yes') & (raw_data['Loan_Status'] == 'N'), 'Self_Employed'].tolist()
self_employed_no_loan_y = raw_data.loc[(raw_data['Self_Employed'] == 'No') & (raw_data['Loan_Status'] == 'Y'), 'Self_Employed'].tolist()
self_employed_no_loan_n = raw_data.loc[(raw_data['Self_Employed'] == 'No') & (raw_data['Loan_Status'] == 'N'), 'Self_Employed'].tolist()

# Creating a simple bar chart
labels = ['SYLY', 'SYLN',
          'SNLY', 'SNLN']
values = [len(self_employed_yes_loan_y), len(self_employed_yes_loan_n),
          len(self_employed_no_loan_y), len(self_employed_no_loan_n)]

plt.bar(labels, values)
plt.xlabel('Combinations')
plt.ylabel('Count')
plt.title('Counts of Combinations of Self_Employed and Loan_Status')
plt.show()
print('''SYLY = Self-Employed Yes, Loan Status Y 
         SYLN = Self-Employed Yes, Loan Status N
         SNLY = Self-Employed No, Loan Status Y
         SNLN = Self-Employed No, Loan Status N''')




# From above, it is clear that:
# 1.Those who are not Self_Employed are likely to get Loan Approvals: An assumption may be that they may be employed and their basic salary that remains unchanged is used in evaluation and assures creditors of loan repayment. Unlike Self_employed applcants who may not have a stable income as they don't have an expected amount to receive.
#   - However, this is also as a result of the number of self_employed_no and loan_status_yes being the highest in the lot.
# 
# We will thus use percentage rate to understand this: 

# In[ ]:


raw_data['Credit_History'].value_counts().plot(kind='pie',autopct='%2.2f%%',title="Visualization of Loan applicants based on Credit History")


# In[ ]:


#Analyzing loan approvals based or Property_Area
property_area_col= raw_data['Property_Area'].value_counts()
loan_status_col= raw_data['Loan_Status'].value_counts()
grouped_data = raw_data.groupby([property_area_col,loan_status_col]).size().unstack()

# Getting the categories and their positions
categories = grouped_data.columns
x = range(len(grouped_data))

# Plotting the bar chart
fig, ax = plt.subplots()
bar_width = 0.35

bar1 = ax.bar(x, grouped_data[categories[0]], bar_width, label=categories[0])
bar2 = ax.bar([i + bar_width for i in x], grouped_data[categories[1]], bar_width, label=categories[1])

# Adding labels and title
ax.set_xlabel('Property Area')
ax.set_ylabel('Count')
ax.set_title('Loan Status by Property Area')
ax.set_xticks([i + bar_width/2 for i in x])
ax.set_xticklabels(grouped_data.index)

# Adding a legend
ax.legend()

# Display the plot
plt.show()


# A huge % of loan applicants have a positive credit history represented by 1.0 . This means most loan applicants are actively repaying past loans.

# In[ ]:


raw_data['Loan_Status'].value_counts().plot(kind='pie',autopct='%2.2f%%',title="Visualization of Loan Approvals")


# A higher percentage of loan applicants were granted. This may be because of several factors. Here are some preliminary assumptions:
#  - A higher number of loan applicants had positive credit history
#  - A higher number of loan applicants had 0 dependants as many lenders believe dependants may impact one's financial capability of repayment
#  - Most applicants were not Self Employed. This means that a majority might have been employed in which their job security and basic salary that remains unchanged ensures repayment. However, this assumption may be biased as some applicants may have been jobless.

# VISUAL ANALYSIS: 
#   -The data has both numerical and categorical data.\n",
#   -For successful analysis, categorical data has to be encoded resulting in a numerical representation, to thus allow analysis.But this occurs at a later stage.To start:
#   
# PROCESS 1: Data Cleaning
# 
# *Handle missing values: Identify and handle missing values in the dataset by either imputing them or removing rows/columns with missing data.
# 
# *Handle outliers: Detect and handle outliers that might negatively affect the model's performance or introduce bias.
# 

# In[ ]:


#Handling Missing values
missing_values = raw_data.isna().sum()
columns_with_missing_values = missing_values[missing_values > 0]

print("Columns with missing values and their sum of missing values:")
for column, count in columns_with_missing_values.items():
    print(column, count)


# In[ ]:


raw_data.head()


# In[ ]:


#To fill missings values, use of SimpleImputer from sklearn.
#I will first clasify the entire dataset into the two in order to be able to re-use the groups in future manipulation that needs separation of categorical and numerical data.

numerical_data_columns =  raw_data[['Dependents','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History']]
categorical_data_columns = raw_data[['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']]


# In[ ]:


from sklearn.impute import SimpleImputer
#I willl use the following strategies: most_frequent for numerical columns and most_frequent for the categorical columns.
numerical_imputer = SimpleImputer(strategy='most_frequent')
categorical_imputer = SimpleImputer(strategy = 'most_frequent')
numerical_imputed_data = numerical_imputer.fit_transform(numerical_data_columns)
categorical_imputed_data = categorical_imputer.fit_transform(categorical_data_columns)


# Confirming filling of missing values

# In[ ]:


#To confirm , we will join the two sets of imputed data and check if the dataframe has any missing values

#numerical_imputed_data = numerical_imputed_data.iloc[:, 1:]
#categorical_imputed_data = categorical_imputed_data.iloc[:, 1:]
numerical_data_df = pd.DataFrame(numerical_imputed_data, columns=['Dependents','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History'])
categorical_data_df = pd.DataFrame(categorical_imputed_data,columns =['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status'] )
#numerical_data_df = numerical_data_df.iloc[1:]
#categorical_data_df = numerical_data_df.iloc[1:]
Imputed_data = pd.concat([numerical_data_df, categorical_data_df],axis=1)


# In[ ]:


checking_missing_values = Imputed_data.isna().sum()
print(checking_missing_values)


# From above, we confirm we have taken care of missing values and there are no missing values in the data.

# In[ ]:


Imputed_data.head(10)


# Above shows the data will all the columns names back in place and all missing values imputed

# Encoding for the categorical columns:

# In[ ]:


#I will perform either one-hot encoding or ordinal encoding on the categorical columns depending on the nature of columns
#ordinal encoding - Columns that have a hierarchical arrangement(rank)
#one-hot encoding - Columns that do not have ranking e.g color and thus uses binary representation of either 0 or 1 .

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoding_columns = ['Education','Self_Employed','Property_Area','Loan_Status']


#Starting with ordinal encoding
#Specification of hierarchy for all the 4:

# Specify hierarchy for 'Education'
education_hierarchy = ['Not Graduate','Graduate']  

encoder1 = OrdinalEncoder(categories=[education_hierarchy])#Defining encoder to use
Imputed_data['Education'] = encoder1.fit_transform(Imputed_data[['Education']])

# Specify hierarchy for 'Self_Employed'
selfEmployed_hierarchy = ['No','Yes'] 

encoder2 = OrdinalEncoder(categories=[selfEmployed_hierarchy])
Imputed_data['Self_Employed'] = encoder2.fit_transform(Imputed_data[['Self_Employed']])

# Specify hierarchy for 'Property_Area'
propertyArea_hierarchy = ['Rural','Semiurban','Urban'] 

encoder3 = OrdinalEncoder(categories=[propertyArea_hierarchy])
Imputed_data['Property_Area'] = encoder3.fit_transform(Imputed_data[['Property_Area']])

# Specify hierarchy for 'Loan_Status'
loanStatus_hierarchy = ['N','Y'] 

encoder3 = OrdinalEncoder(categories=[loanStatus_hierarchy])
Imputed_data['Loan_Status'] = encoder3.fit_transform(Imputed_data[['Loan_Status']])


# In[ ]:


Imputed_data.head()


# The new table above proves that we have ordinally the columns with hierarchical order.

# In[ ]:


#For one_hot encoding as python automatically converts the data to binary representations without need for specification.
#However, this will create new columns of each:, gender will have 2 and Maried will have 2: thus we will have 4 columns in place of original 2.
from sklearn.preprocessing import OneHotEncoder
one_hot_encoding_columns = ['Gender','Married']


Encoded_data = pd.get_dummies(Imputed_data, columns=['Gender', 'Married'])

# Rename columns
Encoded_data.rename(columns={'Gender_Male': 'gender_male', 'Gender_Female': 'gender_female',
                           'Married_No': 'married_no', 'Married_Yes': 'married_yes'}, inplace=True)

Encoded_data.head(20)


# The above table has the Dataframe encoded for categorical columns and all missing values have been dealt with. 
# However, in the dependants column, we have the value 3+, which will read as a string and alter calculations. 
# 

# In[ ]:


#I will convert all values of 3+ in the 'Dependents' column to 3 in order to ensure consistency of numerical values in the data.

for index,value in enumerate(Encoded_data['Dependents']):
    if value == "3+":
        Encoded_data.at[index,'Dependents'] = 3


# In[ ]:


Encoded_data


# The data is now clean and ready for modelling.

# # Modelling Process

# I like explaining my think process as I make this model: 
# 
# For this Phase: First, I will split the data into features(X) and target(Y). Then, I will split the data into two sets,the training set and validation set.
# 
# The training set will be used to make the model, whereby, the model will study the relationship between the training data'X and it's respective target features. It will then use this model to make predictions,
# 
# At predictions, the model will use the Validation data'X to come up with target values. Whereby, it will use the knowledge it gained, and come up with Y values based on the inputed X values of the validation set.
# 
# Then, we will compare the accuracy of the model where , we will compare the predictions of the model with actual Y values of the validation dataset.
# 
# This is just the first phase, in the future improvement phases, I will incoporate pipelines and XGboost for better models.

# In[ ]:


Clean_data = Encoded_data #Getting a new dataframe to avoid any altercation of previous steps.
Y = Clean_data.Loan_Status #Target data
print(Y)


# In[ ]:


X = Clean_data.drop('Loan_Status',axis = 1) #dropping target variable to only remain with features


# In[ ]:


print(X)


# In[ ]:


#Using train_test split to get the 2 pairs for both training data and validation data

from sklearn.model_selection import train_test_split

train_X,validation_X,train_Y,validation_Y = train_test_split(X,Y)


# In[ ]:


#Here, we will create the model using the training data

#I will use a randomforestregressor (It basically ensembles many decisiontrees)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

my_Regressor = RandomForestRegressor(random_state = 30, n_estimators=100 )
#random_state: To ensure reproducibility of results,i.e even if I run this code in different environments, I will still get the same results as long as the random_state remains at 30
#n_estimators: This specifies the numbers of decsion trees in the random_forest

my_Regressor.fit(train_X,train_Y) #fitting the model


# In[ ]:


#Predictions
val_predictions =my_Regressor.predict(validation_X) #model will use the Validation data'X to come up with target values
print(val_predictions)


# In[ ]:


#From above, the predictions from the validation datasets are not whole numbers. 
#Then loan status can either be 1.0 for Yes  or 0.0 for No
#Therefore, we roundoff the predictions

final_val_predictions = np.round(val_predictions) # Roundsoff to the nearest whole number


# In[ ]:


print(final_val_predictions)


# # Model Validation

# In[ ]:


#Use of mae(The mean_absolute_error does the average absolute difference between the predicted and actual values.
my_mae = mean_absolute_error(final_val_predictions,validation_Y)
print(my_mae)


# # Model Improvement: (Hyperparameter tuning)

# In[ ]:


#I willl use pipelines to encompass the above process in order to do different manipulations without having to repeat several steps
#GridSearchCV: performs an exhaustive search over specified hyperparameter values for an estimator, uses the best estimator from those given

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#The GridSearchCV will , from a set of different parameter values, find the best set that it will use to fit the model


# In[ ]:


my_pipeline = Pipeline([
    ('my_model', RandomForestRegressor(random_state = 30, n_estimators=100 ))  #  specify the default parameters of the Random Forest Regressor
])

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'my_model__n_estimators': [150, 200, 300],  # Example parameter values for n_estimators
    'my_model__max_depth': [5, 10, 15]  # Example parameter values for max_depth
}

grid_search = GridSearchCV(my_pipeline,param_grid,cv=4)
grid_search.fit(train_X,train_Y)

my_best_model = grid_search.best_estimator_

val_predictions_2 = my_best_model.predict(validation_X)

mae = mean_absolute_error(np.round(val_predictions_2),validation_Y)

print("Mean absolute error after tuning: ",mae)

print("Best Parameters:", grid_search.best_params_)


# In[ ]:




