#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Importing libraries
import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
get_ipython().run_line_magic('matplotlib', 'inline')

import os

working_directory = os.getcwd() #getting the directory 
# Reading the train.csv by removing the
# last column since it's an empty column
DATA_PATH = working_directory + '/Downloads/Training.csv'
data = pd.read_csv(DATA_PATH).dropna(axis = 1)
 
# Checking whether the dataset is balanced or not
disease_counts = data["prognosis"].value_counts()
temp_df = pd.DataFrame({
    "Disease": disease_counts.index,
    "Counts": disease_counts.values
})
 
plt.figure(figsize = (18,8))
sns.barplot(x = "Disease", y = "Counts", data = temp_df)
plt.xticks(rotation=90)
plt.show()


# Encoding the target value into numerical
# value using LabelEncoder
encoder = LabelEncoder()
data["prognosis"] = encoder.fit_transform(data["prognosis"])


# In[9]:


#splitting the data 
X = data.iloc[:,:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test =train_test_split(
  X, y, test_size = 0.2, random_state = 24)

#Print the shape of TRAIN AND TEST
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")


# In[10]:


# Training and testing SVM Classifier
svm_model = SVC()
svm_model.fit(X_train, y_train)
preds = svm_model.predict(X_test)
 
print(f"Accuracy on train data by SVM Classifier: {accuracy_score(y_train, svm_model.predict(X_train))*100}")
 
print(f"Accuracy on test data by SVM Classifier: {accuracy_score(y_test, preds)*100}")
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for SVM Classifier on Test Data")
plt.show()


# In[11]:


# Training and testing Random Forest Classifier
rf_model = RandomForestClassifier(random_state=18)
rf_model.fit(X_train, y_train)
preds = rf_model.predict(X_test)
print(f"Accuracy on train data by Random Forest Classifier: {accuracy_score(y_train, rf_model.predict(X_train))*100}")
 
print(f"Accuracy on test data by Random Forest Classifier: {accuracy_score(y_test, preds)*100}")
 
cf_matrix = confusion_matrix(y_test, preds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Random Forest Classifier on Test Data")
plt.show()


# In[12]:


#decision Tree
#Decision Tree
dt_clf = DecisionTreeClassifier(splitter='best', criterion='entropy', min_samples_leaf=2)
dt_clf.fit(X_train, y_train)
dpreds = dt_clf.predict(X_test) #desicion tree prediction
print(f"Accuracy on train data by Random Forest Classifier: {accuracy_score(y_train, dt_clf.predict(X_train))*100}")
 
print(f"Accuracy on test data by Random Forest Classifier: {accuracy_score(y_test, dpreds)*100}")
 
cf_matrix = confusion_matrix(y_test, dpreds)
plt.figure(figsize=(12,8))
sns.heatmap(cf_matrix, annot=True)
plt.title("Confusion Matrix for Random Forest Classifier on Test Data")
plt.show()


# In[ ]:


"""From the above confusion matrices, 
we can see that the models are performing very well on the unseen data. Now we will be training 
the models on the whole train data present in the dataset that we downloaded and then test our 
combined model on test data present in the dataset.

"""


# In[13]:


# Training the models on whole data
final_svm_model = SVC()
final_nb_model = GaussianNB()
final_rf_model = RandomForestClassifier(random_state=18)
final_d_model = DecisionTreeClassifier(splitter='best', criterion='entropy', min_samples_leaf=2)
final_svm_model.fit(X, y)
final_nb_model.fit(X, y)
final_rf_model.fit(X, y) #fitting random forest
final_d_model.fit(X, y) #fitting decision tree
# Reading the test data

test_data = pd.read_csv(working_directory + '/Downloads/Testing.csv').dropna(axis=1) #test data
 
test_X = test_data.iloc[:, :-1]
test_Y = encoder.transform(test_data.iloc[:, -1])
 
# Making prediction by take mode of predictions
# made by all the classifiers
svm_preds = final_svm_model.predict(test_X)
nb_preds = final_nb_model.predict(test_X)
rf_preds = final_rf_model.predict(test_X) #final random forest
fd_preds = final_d_model.predict(test_X) #final decision tree
 
final_preds = [mode([i,j,k])[0][0] for i,j,
               k in zip(svm_preds, fd_preds, rf_preds)]
 
print(f"Accuracy on Test dataset by the combined model: {accuracy_score(test_Y, final_preds)*100}")
 
cf_matrix = confusion_matrix(test_Y, final_preds)
plt.figure(figsize=(12,8))
 
sns.heatmap(cf_matrix, annot = True)
plt.title("Confusion Matrix for Combined Model on Test Dataset")
plt.show()


# In[ ]:


"""We can see that our combined model has classified all the data points accurately. 
We have come to the final part of this whole implementation, we will be creating a function that 
takes symptoms separated by commas as input and outputs the predicted 
disease using the combined model based on the input symptoms."""


# In[15]:


symptoms = X.columns.values

# Creating a symptom index dictionary to encode the
# input symptoms into numerical form
symptom_index = {}
for index, value in enumerate(symptoms):
    symptom = " ".join([i.capitalize() for i in value.split("_")])
    symptom_index[symptom] = index
 
data_dict = {
    "symptom_index":symptom_index,
    "predictions_classes":encoder.classes_
}
 
# Defining the Function
# Input: string containing symptoms separated by commmas
# Output: Generated predictions by models
def predictDisease(symptoms):
    symptoms = symptoms.split(",")

    # creating input data for the models
    input_data = [0] * len(data_dict["symptom_index"])
    for symptom in symptoms:
        index = data_dict["symptom_index"][symptom]
        input_data[index] = 1
         
    # reshaping the input data and converting it
    # into suitable format for model predictions
    input_data = np.array(input_data).reshape(1,-1)
     
    # generating individual outputs
    rf_prediction = data_dict["predictions_classes"][final_rf_model.predict(input_data)[0]] #random forest
    dt_prediction = data_dict["predictions_classes"][final_d_model.predict(input_data)[0]] #decision tree
     
    # making final prediction by taking mode of all predictions
    final_prediction = mode([rf_prediction, dt_prediction])[0][0]
    predictions = {
        "rf_model_prediction": rf_prediction,
        "dt_prediction": dt_prediction,
        "final_prediction":final_prediction
    }
    return predictions
 
# Testing the function
print(predictDisease("Itching,Skin Rash,Nodal Skin Eruptions"))

