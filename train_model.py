import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv("dataset.csv")
print("Dataset Loaded")

# -----------------------------
# DATA CLEANING
# -----------------------------

# Fix Age
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
data = data[(data['Age'] >= 18) & (data['Age'] <= 65)]

# Clean Gender
data['Gender'] = data['Gender'].str.lower()

data['Gender'] = data['Gender'].replace(
['male','m','man','male-ish','maile','mal','make','cis male'],
'Male'
)

data['Gender'] = data['Gender'].replace(
['female','f','woman','cis female'],
'Female'
)

data['Gender'] = data['Gender'].replace(
['trans-female','non-binary','queer','genderqueer'],
'Other'
)

# Select important columns
data = data[['Age','Gender','family_history','work_interfere',
             'remote_work','benefits','care_options',
             'seek_help','anonymity','leave',
             'mental_health_consequence',
             'coworkers','supervisor','treatment']]

# Remove missing values
data = data.dropna()

# Convert Yes/No
data['family_history'] = data['family_history'].map({'Yes':1,'No':0})
data['remote_work'] = data['remote_work'].map({'Yes':1,'No':0})
data['treatment'] = data['treatment'].map({'Yes':1,'No':0})

# Encode categorical columns
label = LabelEncoder()

cat_cols = ['Gender','work_interfere','benefits','care_options',
            'seek_help','anonymity','leave',
            'mental_health_consequence','coworkers','supervisor']

for col in cat_cols:
    data[col] = label.fit_transform(data[col])

# -----------------------------
# FEATURES
# -----------------------------

X = data.drop("treatment", axis=1)
y = data["treatment"]

# Balance dataset
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# MODEL (XGBoost)
# -----------------------------

model = XGBClassifier(
    n_estimators=800,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open("model.pkl","wb"))

print("\nModel Saved Successfully!")
