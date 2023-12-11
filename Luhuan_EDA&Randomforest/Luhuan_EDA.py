import pandas as pd
import numpy as np
import math
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from scipy.stats import pearsonr
import sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns

students = pd.read_csv('C:/Users/luhua/Desktop/DS03/final pre/data.csv', delimiter=';')
df = students.copy()

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

# Quantitative distribution of parents’ occupation
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

binary_variables = ['Displaced', 'Educational special needs', 'Debtor', 'Tuition fees up to date', 'Scholarship holder',
                    'International']
plt.figure(figsize=(12, 10))
for i, col in enumerate(binary_variables, 1):
    plt.subplot(2, 3, i)
    df[col].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightgreen'])
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Define target variable mapping dictionary
mapping_dict = {'Dropout': 0, 'Enrolled': 1, 'Graduate': 2}
students['Target'] = students['Target'].replace(mapping_dict)

# DataFrame
selected_columns = ["Mother's qualification", "Father's qualification", "Father's occupation", "Mother's occupation"
    , 'Displaced', 'Educational special needs', 'Debtor',
                    'Tuition fees up to date', 'Scholarship holder', 'International', 'Target']

selected_df = students[selected_columns]

# Calculate the correlation between features and target variables
correlation_matrix = selected_df.corr()

# Select the target variable correlation column to generate a heat map
target_correlation = correlation_matrix['Target']

plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='YlGnBu', fmt=".2f")
plt.xticks(rotation=45, ha='right')
plt.title('Correlation Heatmap between All Features')
plt.tight_layout()
plt.show()

# Calculate Pearson correlation coefficient
pearson_coefficient, p_value = pearsonr(selected_df['Target'], selected_df['Tuition fees up to date'])
print(f"Tuition fees up to date_Pearson Correlation Coefficient: {pearson_coefficient}, p-value: {p_value}")
pearson_coefficient, p_value = pearsonr(selected_df['Target'], selected_df['Scholarship holder'])
print(f"Scholarship holder_Pearson Correlation Coefficient: {pearson_coefficient}, p-value: {p_value}")
pearson_coefficient, p_value = pearsonr(selected_df['Target'], selected_df['Debtor'])
print(f"Debtor_Pearson Correlation Coefficient: {pearson_coefficient}, p-value: {p_value}")
# Calculate Spearman correlation coefficient
spearman_coefficient, p_value = spearmanr(selected_df['Target'], selected_df['Tuition fees up to date'])
print(f"Tuition fees up to date_Spearman Correlation Coefficient: {spearman_coefficient}, p-value: {p_value}")
spearman_coefficient, p_value = spearmanr(selected_df['Target'], selected_df['Scholarship holder'])
print(f"Scholarship holder_pearman Correlation Coefficient: {spearman_coefficient}, p-value: {p_value}")
spearman_coefficient, p_value = spearmanr(selected_df['Target'], selected_df['Debtor'])
print(f"Debtor_Spearman Correlation Coefficient: {spearman_coefficient}, p-value: {p_value}")

# t-test
# Tuition fees up to date
t_statistic, p_value_tuition = ttest_ind(selected_df['Target'], selected_df['Tuition fees up to date'])
print(f"Tuition fees up to date_t statistic: {t_statistic}, p-value: {p_value_tuition}")

# Scholarship holder
t_statistic, p_value_scholarship = ttest_ind(selected_df['Target'], selected_df['Scholarship holder'])
print(f"Scholarship holder_t statistic: {t_statistic}, p-value: {p_value_scholarship}")

# Debtor
t_statistic, p_value_debtor = ttest_ind(selected_df['Target'], selected_df['Debtor'])
print(f"Debtor_t statistic: {t_statistic}, p-value: {p_value_debtor}")
