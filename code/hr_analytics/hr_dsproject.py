# Data Science Project- HR Recruitment Analytics
# Wong Ann Nee, October2020

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn import preprocessing, ensemble  
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  
from sklearn.naive_bayes import GaussianNB  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier  
from sklearn.ensemble import AdaBoostClassifier  
from sklearn.ensemble import ExtraTreesClassifier  
from sklearn.ensemble import BaggingClassifier  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.neural_network import MLPClassifier  
from sklearn.svm import SVC

import keras
import keras_metrics
from keras.models import Sequential
from keras.layers import Dense


# Import Dataset
hr=pd.read_csv("C:\\hr.csv") 
pd.set_option('display.max_column',None)
hr.head()
hr.columns
hr.count()
hr.shape[0]
pd.value_counts(hr.Accept_offer)



###################### Step 1 : Data Exploration #####################

#Exploration of all variables, unique values
all_attr=['Id', 'Age', 'Gender', 'Marital_status', 'Num_comp_worked','Min_experience', 'Last_salary', 'Salary_hike', 'Work_hours','Traveling_required', 'Comp_buyout', 'Job_type','Possess_vehicle', 'Work_location', 'Traveltowork_distance_&_time', 'Work_area_living_cost', 'Public_transport_nearby', 'Current_employment', 'Processtime(week)', 'Bonus', 'Allowance', 'Overtime_pay', 'Extra_EPF', 'Emp_shares_scheme', 'Performance_incentive', 'Annual_leave', 'Medical', 'Medical_Family', 'Child_day_care', 'Nursing_room', 'Parking', 'Comp_size', 'Comp_rating', 'Work_life', 'Stresslevel', 'Accept_offer']

for i in all_attr:
    print(i, hr[i].unique())    

# Plot features with missing variables
missing = pd.DataFrame(hr.isna().sum())
missing = missing.reset_index()
missing = missing.rename(columns={'index':'Features', 0:'Missing_values'})
missing = missing.sort_values('Missing_values',ascending=False)
missing = missing.drop(0,)

%matplotlib qt
missing.plot.bar(x='Features', y='Missing_values', rot=90, color={"orange"})
plt.title("Missing Variables")
plt.show()

# Explore correlation of Nursing_room 
list1 = pd.DataFrame(hr.corr().iloc[:,29])
list1 = list1.sort_values ('Nursing_room', ascending=False)
list1
hr.Comp_size.unique()

# Exploration of 'Nursing_room' 
%matplotlib qt
chart1 = hr.groupby(['Comp_size', 'Nursing_room'])['Comp_size'].count().unstack('Nursing_room').fillna(0)
chart1[[0,1]].plot(kind='bar', stacked=True)
plt.show()

import seaborn as sns

corr = hr.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
# Heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(corr,
            vmax=.5,
            mask=mask,
            # annot=True, fmt='.2f',
            linewidths=.2, cmap="YlGnBu")



################### Step 2 : Replace Missing Values ##################

# Replace missing value with correlation to Comp_size
# 0,1,2 Comp_size replace with '0' Nursing Room
# 3,4 Comp_size replace with '1' Nursing_room
size=[0,1,2,3,4]
nursing=[0,0,0,1,1]
for i in range(5):
    hr.loc[(hr.Nursing_room.isnull() & (hr.Comp_size==size[i])),'Nursing_room'] = nursing[i]
hr.Nursing_room.isna().sum()




##################### Step 3 : Split Train and Test dataset ############

# Split to test and train dataset 70:30 ratio
x=hr.drop(['Accept_offer','Id'], axis=1)
y=hr.Accept_offer
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0)



############# Step 4a : Feature Importance with Random Forest ########

modelrf = RandomForestClassifier()
modelrf = modelrf.fit(x_train,y_train)
importancesrf = pd.DataFrame({'feature':x_train.columns,'importance':np.round(modelrf.feature_importances_,3)})
importancesrf = importancesrf.sort_values('importance',ascending=False)
importancesrf

%matplotlib qt
importancesrf.plot.bar(x='feature', y='importance', rot=90, color={"red"})
plt.title("Feature Importance")
plt.show()

######## Step 4b : Feature Importance with Decision Tree #############

modeldt = DecisionTreeClassifier()
modeldt = modeldt.fit(x_train,y_train)
importancesdt = pd.DataFrame({'feature':x_train.columns,'importance':np.round(modeldt.feature_importances_,3)})
importancesdt = importancesdt.sort_values('importance',ascending=False)
importancesdt
# Plot feature importance
%matplotlib qt
importancesdt.plot.bar(x='feature', y='importance', rot=90, color={"red"})
plt.title("Feature Importance")
plt.show()



######################## Step 5a : Modelling  ############################

# The models to be used
models = []
model1 = GradientBoostingClassifier()
model2 = DecisionTreeClassifier()
model3 = RandomForestClassifier()
model4 = LogisticRegression()
model5 = GaussianNB()
model6 = ExtraTreesClassifier()
model7 = BaggingClassifier()
model8 = MLPClassifier()
model9 = AdaBoostClassifier()
model10 = KNeighborsClassifier()
model11 = SVC(kernel = "linear")
model12 = SVC(kernel = "rbf")
model13 = SVC(kernel = "poly")
model14 = SVC(kernel = "sigmoid")
                   
model_list = {'GB': model1, 'DT': model2, 'RF': model3,'LR': model4, 'GN': model5, 'ET': model6, 'BG': model7, 'MLP': model8, 'AB': model9, 'KNN':model10, 'SVCLinear':model11, 'SVCrbf':model12, 'SVCpoly':model13,'SVCsigmoid':model14}

result=pd.DataFrame()
results = pd.DataFrame()

for key in model_list:
    model = model_list[key].fit(x_train, y_train)
    start1=time.time()
    predictions_tr = model.predict(x_train)
    end1=time.time()
    train_time=end1-start1
    
    start2=time.time()
    predictions_te = model.predict(x_test)
    end2=time.time()
    test_time=end2-start2

    accuracy_tr = accuracy_score(y_train, predictions_tr)
    recall_tr = recall_score(y_train, predictions_tr)
    accuracy_tr = float("{0:.3f}".format(accuracy_tr))
    recall_tr = float("{0:.3f}".format(recall_tr))
    accuracy_te = accuracy_score(y_test, predictions_te)
    recall_te = recall_score(y_test, predictions_te)
    accuracy_te = float("{0:.3f}".format(accuracy_te))
    recall_te = float("{0:.3f}".format(recall_te))
    train_time = float("{0:.3f}".format(train_time))
    test_time = float("{0:.3f}".format(test_time))
    
    print(key)
    print("Train accuracy:", accuracy_tr)
    print("Train recall  :", recall_tr)
    print("Test accuracy :", accuracy_te)
    print("Test recall   :", recall_te)
    print("Train time    :", train_time)
    print("Test time     :", test_time)
    print()

    result = pd.DataFrame({"model":[key], "train_acc":[accuracy_tr], "train_recall":[recall_tr], "test_acc":[accuracy_te], "test_recall":[recall_te], "train_time":[train_time],"test_time":[test_time]})
    results = results.append(result,ignore_index = True)
    
print(results)


############### Step 5b : Modelling Keras MLP/ANN ####################
###############              BEST MODEL          #####################

# define the keras model
modelnn = Sequential()
modelnn.add(Dense(20, input_dim=34, activation='relu'))
modelnn.add(Dense(10, activation='relu'))
modelnn.add(Dense(1, activation='sigmoid'))
modelnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', keras.metrics.Recall()])

# fit the keras model on the train and test dataset
start3=time.time()
model=modelnn.fit(x_train, y_train, epochs=50, batch_size=10)
#y_pred=model.predict(x_train)
end3=time.time()
train_time_nn=end3-start3

start4=time.time()
modelnn.fit(x_test, y_test, epochs=50, batch_size=10)   
end4=time.time()
test_time_nn=end4-start4      

print("Train time NN :", train_time_nn)
print("Test time  NN :", test_time_nn)

