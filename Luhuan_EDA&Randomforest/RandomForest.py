#%%
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

students = pd.read_csv('data.csv', delimiter=';')
df = students.copy()

#%%
# random forest

# List of columns to keep
cols = ['Age at enrollment', 'Gender', 'Marital status', 'Displaced', 'Debtor', 'Tuition fees up to date',
        'Scholarship holder', 'Application mode', "Daytime/evening attendance	", 'Course',
        'Curricular units 2nd sem (approved)',
        'Previous qualification', "Mother's occupation", 'Target']


df = df[cols]
df = df[df['Target'] != 'Enrolled']

# Convert 'Target' column into numerical data type
df = df.replace({'Target': {'Dropout': 0, 'Graduate': 1}})
df['Target'] = df['Target'].astype('int32')

#  categorical columns
df = pd.get_dummies(df, drop_first=True)

# Define predictor and target variables
y = df['Target']
X = df.drop('Target', axis=1)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Set up Random Forest classifier
rf = RandomForestClassifier(random_state=0)
cv_params = {'n_estimators': [25, 50, 75, 100],
             'max_depth': [10, 30, 50, 70],
             'min_samples_leaf': [0.5, 0.75, 1],
             'min_samples_split': [0.001, 0.005, 0.01],
             'max_features': ['sqrt'],
             'max_samples': [.3, .5, .7, .9]}

rf = RandomForestClassifier(random_state=0)
rf_val = GridSearchCV(rf, cv_params, cv=3, refit='accuracy', n_jobs=-1, verbose=1)
rf_val.fit(X_train, y_train)

grid_search = GridSearchCV(rf, cv_params, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)
rf_accuracy = grid_search.best_estimator_.score(X_test, y_test)
print("Test Accuracy:", rf_accuracy)

# Fit the model with best parameters from GridSearchCV
rf = RandomForestClassifier(**grid_search.best_params_, random_state=0)
rf.fit(X_train, y_train)

# Make predictions on the test set
y_preds = rf.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_preds)
precision = precision_score(y_test, y_preds)
recall = recall_score(y_test, y_preds)
f1 = f1_score(y_test, y_preds)

print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1 Score: {f1:.3f}")

# Plot confusion matrix
cm = confusion_matrix(y_test, y_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Dropout', 'Graduate'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# Calculate predicted probabilities and plot ROC curve
y_preds_prob_rf = rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_preds_prob_rf)
auc = roc_auc_score(y_test, y_preds_prob_rf)

plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve (AUC = {auc:.2f})')
plt.show()
true_labels = y_test
predicted_labels = y_preds
report = classification_report(true_labels, predicted_labels)
print(report)
