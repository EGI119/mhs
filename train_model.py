import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Load data
data = pd.read_csv('Student_Mental_health.csv')

# Imputasi
imputer = SimpleImputer(strategy='most_frequent')
data[['Age', 'What is your course?', 'Marital status']] = imputer.fit_transform(data[['Age', 'What is your course?', 'Marital status']])

# Drop baris kosong
data.dropna(subset=['Do you have Depression?', 'Choose your gender', 'Do you have Anxiety?', 'Do you have Panic attack?', 'Did you seek any specialist for a treatment?'], inplace=True)

# Standarisasi format
data['Your current year of Study'] = data['Your current year of Study'].str.lower().str.strip()
data['What is your CGPA?'] = data['What is your CGPA?'].str.replace(' ', '').str.replace('-', ' to ')
data['CGPA_numeric'] = data['What is your CGPA?'].apply(lambda x: np.mean(list(map(float, x.split(' to ')))) if 'to' in x else float(x))

# Hapus outlier
Q1 = data['CGPA_numeric'].quantile(0.25)
Q3 = data['CGPA_numeric'].quantile(0.75)
IQR = Q3 - Q1
data = data[~((data['CGPA_numeric'] < Q1 - 1.5 * IQR) | (data['CGPA_numeric'] > Q3 + 1.5 * IQR))]

# Mapping
binary_map = {'Yes': 1, 'No': 0}
data['Do you have Depression?'] = data['Do you have Depression?'].map(binary_map)
data['Choose your gender'] = data['Choose your gender'].map({'Male': 1, 'Female': 0})
for col in ['Marital status', 'Do you have Anxiety?', 'Do you have Panic attack?', 'Did you seek any specialist for a treatment?']:
    data[col] = data[col].map(binary_map)

data['Your current year of Study'] = data['Your current year of Study'].map({'year 1': 1, 'year 2': 2, 'year 3': 3, 'year 4': 4})

# Hapus kolom tidak perlu
data.drop(columns=['Timestamp', 'What is your course?', 'What is your CGPA?'], inplace=True)

# Fitur dan target
X = data[['Choose your gender', 'Age', 'Your current year of Study', 'CGPA_numeric', 'Marital status', 'Do you have Anxiety?', 'Do you have Panic attack?', 'Did you seek any specialist for a treatment?']]
y = data['Do you have Depression?']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.1, random_state=42)

# Feature Selection
rfe = RFE(estimator=RandomForestClassifier(n_estimators=100, random_state=42), n_features_to_select=5)
rfe.fit(X_train, y_train)
selected_features = X.columns[rfe.support_]

# Transformasi
X_train_selected = rfe.transform(X_train)
X_test_selected = rfe.transform(X_test)

# Grid Search
param_grid = {
    'n_estimators': [100],
    'max_features': ['sqrt'],
    'max_depth': [None],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'bootstrap': [True]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=StratifiedKFold(n_splits=5), n_jobs=-1)
grid.fit(X_train_selected, y_train)

# Simpan model
with open('student_model.pkl', 'wb') as f:
    pickle.dump(grid.best_estimator_, f)

# Simpan fitur
with open('features.pkl', 'wb') as f:
    pickle.dump(list(selected_features), f)

print("✅ Model dan fitur berhasil disimpan.")
