from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ds = pd.read_csv('crop_production.csv')
ds.head()

ds.shape

ds.isnull().sum()

dsc = ds.dropna()
dsc.isnull().sum()

dsc.shape

dsc.info()

dsc.duplicated().sum()

dsc.describe()

dsc = dsc.drop(['Season'], axis = 1)

plt.figure(figsize = (15,10))
sns.countplot(y = dsc['State_Name'])

plt.figure(figsize = (15,10))
sns.countplot(y = dsc['Crop_Year'])

plt.figure(figsize = (15,10))
sns.countplot(y = dsc['Crop'])

states = dsc['State_Name'].unique()

yield_per_state = []
for state in states:
  yield_per_state.append(dsc[dsc['State_Name']==state]['Production'].sum())


yield_per_state

crops = dsc['Crop'].unique()

yield_per_item = []
for item in crops:
  yield_per_item.append(dsc[dsc['Crop']==item]['Production'].sum())

yield_per_item

x = dsc.drop('Production', axis = 1)
y = dsc['Production']

X_train, X_test,y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

ohe = OneHotEncoder(drop = 'first')
scaler = StandardScaler()

preprocessor = ColumnTransformer(
    transformers = [
        ('onehotencoder',ohe,[0,1,3]),
        ('standardization', scaler, [2,4])
    ],
    remainder = 'passthrough'
)

X_train_dummy = preprocessor.fit_transform(X_train)
X_test_dummy = preprocessor.transform(X_test)

X_train_dummy

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

dt = DecisionTreeRegressor()
dt.fit(X_train_dummy, y_train)
y_predict = dt.predict(X_test_dummy)

print(f"Score: {r2_score(y_test, y_predict)}")

def prediction(State_Name, District_Name, Crop_Year, Crop, Area):
  features = np.array([[State_Name, District_Name, Crop_Year, Crop, Area]])

  transformed_features = preprocessor.transform(features)
  predicted_value = dt.predict(transformed_features).reshape(1, -1)
  return predicted_value[0]

State_Name = 'Andaman and Nicobar Islands'
District_Name = 'NICOBARS'
Crop_Year = 2000
Crop = 'Rice'
Area = 1002.0

result = prediction(State_Name, District_Name, Crop_Year, Crop, Area)
print(result)

import joblib
joblib.dump(dt, 'yield_model.joblib')

joblib.dump(preprocessor, 'preprocessor.joblib')
