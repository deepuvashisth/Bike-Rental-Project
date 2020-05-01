import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.read_csv('day.csv')

data.head()
data.describe()

#Taking care of date
data['dteday'] = pd.to_datetime(data['dteday'], format = "%Y-%m-%d")
data['dteday'] = data['dteday'].dt.day

#Conversion of variables
#data['day'] = data['day'].astype('category')
data['season'] = data['season'].astype('category')
data['mnth'] = data['mnth'].astype('category')
data['holiday'] = data['holiday'].astype('category')
data['weekday'] = data['weekday'].astype('category')
data['weathersit'] = data['weathersit'].astype('category')

#Outlier Analysis
sns.boxplot(data=data['casual'])
sns.boxplot(data=data[‘registered’])
sns.boxplot(data=data['instant'])
sns.boxplot(data=data['temp'])
sns.boxplot(data=data['atemp'])
sns.boxplot(data=data['hum'])
sns.boxplot(data=data['windspeed'])

Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1

data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]

#Feature selection
#Using Pearson Correlation
plt.figure(figsize=(12,10))
cor = data.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#Correlation with output variable
cor_target = abs(cor["cnt"])

#Selecting highly correlated features
relevant_features = cor_target[cor_target>0.5]
relevant_features

#Droping low correlatedvariables
del data['hum']
del data['windspeed']
del data['dteday']

#Dropping multicollinear variables
del data['instant']
del data['atemp']

#Splitting in train and test
x = data.values[:,0:10]
y = data.values[:,10]

x = pd.DataFrame(x)
y = pd.DataFrame(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#Function for calculating Mean Absolute Percentage Error
def MAPE(y_true, y_pred): 
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mape = np.mean(np.abs((y_true-y_pred)/y_true))*100
    return mape
 

#Decision Tree
from sklearn import tree
dt_model = tree.DecisionTreeRegressor(max_depth=1).fit(x_train, y_train)

dt_predicts = dt_model.predict(x_test)

MAPE(y_test, dt_predicts)
#mape = 68.14



#Random Forest
from sklearn.ensemble import RandomForestRegressor
RF_model = RandomForestRegressor(n_estimators = 100).fit(x_train, y_train)

rf_predict = RF_model.predict(x_test)   

MAPE(y_test, rf_predict)
#mape = 71.20


    
#Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression().fit(x_train, y_train)

lr_predicts = regressor.predict(x_test)

MAPE(y_test, lr_predicts)
#mape = 5.784490670664497e-14


#KNN Algorithm
from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors = 4).fit(x_train, y_train)

knn_predicts = knn_model.predict(x_test)

MAPE(y_test, knn_predicts)
#mape = 1.43




