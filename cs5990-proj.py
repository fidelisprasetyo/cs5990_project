import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import GridSearchCV

import kagglehub

# download data from kaggle
path = kagglehub.dataset_download("taweilo/loan-approval-classification-data")

# read the data
df = pd.read_csv(path + '/loan_data.csv')

# encode categorical data
df_encoded = pd.get_dummies(df, drop_first=True)

# seperate class column from feature column
X = df_encoded.drop('loan_status', axis=1)
y = df_encoded['loan_status']

# split training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# create random forest classifier and traing with data
clf = RandomForestClassifier(max_depth=None, random_state=42)
clf.fit(X_train, y_train)

# check accuracy from prediction
y_pred1 = clf.predict(X_test)

# check accuracy from prediction
y_pred = clf.predict(X_test)
y_probs = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
print(accuracy)
print(report)

conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)

plt.title('Confusion Matrix Heatmap')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.grid(True)
plt.show()

# -----------------grid search & cross validation (10)-------------------

# param_grid = {
#     'n_estimators': [50, 100, 200, 300],
#     'max_depth': [None, 10, 20],
#     'class_weight' : [None, 'balanced']
# }

# grid_search = GridSearchCV(
#     estimator=RandomForestClassifier(random_state=42),
#     param_grid=param_grid,
#     cv=10,
#     scoring='f1',
#     n_jobs=-1,
#     verbose=2
# )

# grid_search.fit(X_train, y_train)

# print("Best parameters:", grid_search.best_params_)
# print("Best F1 Score:", grid_search.best_score_)

# results_df = pd.DataFrame(grid_search.cv_results_)
# top_results = results_df.sort_values(by='mean_test_score', ascending=False)
# top_results.to_csv("gridsearch_f1_scores.csv", index=False)
# ------------------------------------------------------------------------


# # # -----------------Compare no SMOTE, SMOTE, and Balanced class weight-------------------
# smote = SMOTE(random_state=42) # <--- for SMOTE
# X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train) # <--- for SMOTE

# # create random forest classifier and traing with data
# clf = RandomForestClassifier(max_depth=None, n_estimators=200, random_state=42)
# clf2 = RandomForestClassifier(max_depth=None, n_estimators=200, random_state=42, class_weight='balanced')
# clf3 = RandomForestClassifier(max_depth=None, n_estimators=200, random_state=42)

# clf.fit(X_train, y_train)
# clf2.fit(X_train, y_train)
# clf3.fit(X_train_resampled, y_train_resampled) # <--- for SMOTE

# # check accuray from prediction
# y_pred1 = clf.predict(X_test)
# y_pred2 = clf2.predict(X_test)
# y_pred3 = clf3.predict(X_test)

# print("=== No SMOTE ===")
# print(classification_report(y_test, y_pred1))

# print("=== With Class Weight ===")
# print(classification_report(y_test, y_pred2))

# print("=== With SMOTE ===")
# print(classification_report(y_test, y_pred3))
