# HR Analytics – Employee Attrition Prediction

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('IBM_HR_Analytics.csv')

# Show data
print(df.head())

# Data info
print(df.info())

# Attrition count
sns.countplot(x='Attrition', data=df)
plt.title('Attrition Count')
plt.show()

# Salary vs Attrition
sns.boxplot(x='Attrition', y='MonthlyIncome', data=df)
plt.show()

# Encode categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Split data
from sklearn.model_selection import train_test_split
X = df.drop('Attrition', axis=1)
y = df['Attrition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))