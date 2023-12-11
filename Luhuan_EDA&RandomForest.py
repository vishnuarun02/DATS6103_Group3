import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import chi2_contingency
import sklearn.metrics as metrics

students = pd.read_csv('C:/Users/luhua/Desktop/DS03/final pre/data.csv', delimiter=';')
df = students.copy()

# EDA about  Parents qualificationParents occupationDisplaced, Educational special needs, Debtor, Tuition fees up to
# date, Scholarship holder, International
# Parents qualification
# count 'Mother's qualification'  'Father's qualification'
# Frequency distribution of parenthood
plt.figure(figsize=(10, 6))
students["Mother's qualification"].value_counts().plot(kind='bar', color='skyblue')
plt.xlabel("Mother's Qualification")
plt.ylabel('Count')
plt.title("Distribution of Mother's Qualification")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
students["Father's qualification"].value_counts().plot(kind='bar', color='lightgreen')
plt.xlabel("Father's Qualification")
plt.ylabel('Count')
plt.title("Distribution of Father's Qualification")
plt.xticks(rotation=45)
plt.show()

# Quantitative distribution of parents’ occupations
plt.figure(figsize=(10, 6))
students["Mother's occupation"].value_counts().plot(kind='bar', color='skyblue')
plt.xlabel("Mother's Occupation")
plt.ylabel('Count')
plt.title("Distribution of Mother's Occupation")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10, 6))
students["Father's occupation"].value_counts().plot(kind='bar', color='lightgreen')
plt.xlabel("Father's Occupation")
plt.ylabel('Count')
plt.title("Distribution of Father's Occupation")
plt.xticks(rotation=45)
plt.show()

# Proportional display of binary variable columns
binary_cols = ['Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Scholarship holder', 'International']

plt.figure(figsize=(15, 10))
for i, col in enumerate(binary_cols, 1):
    plt.subplot(2, 3, i)
    counts = students[col].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()


mapping_dict = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}

students['Target'] = students['Target'].replace(mapping_dict)
correlation_matrix = students.corr()
print(correlation_matrix)

# Mother's qualification  Father's qualification
contingency_table = pd.crosstab(students["Mother's qualification"], students["Father's qualification"])
chi2, p, dof, expected = chi2_contingency(contingency_table)
print(f"Chi-Square Test Statistic: {chi2}")
print(f"P-value: {p}")

# Educational special needs
for col in ['Debtor', 'Tuition fees up to date', 'Scholarship holder', 'International']:
    contingency_table = pd.crosstab(students['Educational special needs'], students[col])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print(f"Chi-Square Test Statistic for Educational special needs vs {col}: {chi2}")
    print(f"P-value: {p}")


# random forest
cols = ['Application mode', 'Course', 'Previous qualification', "Mother's qualification", 'Tuition fees up to date',
        "Mother's occupation", 'Gender', 'Scholarship holder', 'Age at enrollment', 'Curricular units 1st sem ('
                                                                                    'approved)',
        'Curricular units 2nd sem (approved)', 'Target']

# Keep only relevant columns.
df = df[cols]

# Remove enrolled students.
df = df[df['Target'] != 'Enrolled']

# Convert into numerical data type.
df = df.replace({'Target': {'Dropout': 0, 'Graduate': 1}})
cols = ['Tuition fees up to date', 'Gender', 'Scholarship holder', 'Target']
df[cols] = df[cols].astype('int32')

# Perform one-hot encoding.
df = pd.get_dummies(df, drop_first=True)
results = pd.DataFrame(columns=['Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

# Predicting variable.
y = df['Target']

# Predictor features.
X = df.copy()
X = X.drop('Target', axis=1)

# Create training and test sets, 75% and 25% respectively.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


def conf_matrix_plot(model, x_data, y_data):
    model_pred = model.predict(x_data)
    cm = metrics.confusion_matrix(y_data, model_pred)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Dropout', 'Graduate'], )
    disp.plot(values_format='')
    plt.show()


def plot_roc_curve(true_y, y_probs):
    fpr, tpr, thresholds = metrics.roc_curve(true_y, y_probs)
    auc = metrics.roc_auc_score(true_y, y_probs)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve (AUC = {auc:.2f})')
    plt.show()


def print_results(algo, y_test, y_preds):
    print(f"{algo}")
    accuracy = metrics.accuracy_score(y_test, y_preds)
    precision = metrics.precision_score(y_test, y_preds)
    recall = metrics.recall_score(y_test, y_preds)
    f1_score = metrics.f1_score(y_test, y_preds)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1_score:.3f}")

    # Write results into a dataframe
    row = {'Algorithm': algo, 'Accuracy': accuracy,
           'Precision': precision, 'Recall': recall,
           'F1 Score': f1_score}
    results = pd.DataFrame(row, index=[0])
    return results


rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

# Make predictions on set tests
y_preds = rf.predict(X_test)

# Print model evaluation results
results = print_results('Random forest', y_test, y_preds)
print(results)

# Plot confusion matrix
conf_matrix_plot(rf, X_test, y_test)

# Calculate predicted probabilities and plot an ROC curve
y_preds_prob_rf = rf.predict_proba(X_test)[:, 1]
plot_roc_curve(y_test, y_preds_prob_rf)