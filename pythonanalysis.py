#%%
import pandas as pd


df = pd.read_csv('/Users/bhoomikan/Desktop/Python_Project/data_analysis.csv')
df.head(10)
# %%
import matplotlib.pyplot as plt
import seaborn as sns
from dython.nominal import associations
import pandas as pd  # Assuming you have a pandas DataFrame

# Assuming 'df' is your DataFrame
df = df[df['Target'] != 'Enrolled']
# Define your columns
cols = [
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP']

n_cols = 3  # Number of columns in subplot grid
n_rows = (len(cols) + n_cols - 1) // n_cols  # Calculate the number of rows needed

# Create subplots for box plots
fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 12))
plt.subplots_adjust(hspace=0.5)

# Flatten the axs array for easy iteration
axs = axs.flatten()

# Iterate and create box plots
for index, col in enumerate(cols):
    sns.boxplot(data=df, x='Target', y=col, showfliers=False, ax=axs[index])
    axs[index].set(xlabel=None, ylabel=None, title=col)

# Adjust the layout of box plots
plt.tight_layout()


#%%
cols = [
    'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP','Target']
# Create a separate subplot for the correlation ratio
fig, ax = plt.subplots(figsize=(12, 8))
cor_ratio = associations(df[cols], nom_num_assoc='correlation_ratio', num_num_assoc='pearson', ax=ax, cmap='Blues')

# Show all the plots
plt.show()
# %%
y = df['Curricular units 1st sem (grade)']
x = df['Curricular units 2nd sem (grade)']
plt.scatter(x, y)
plt.title('Scatter Plot For Grades')
plt.xlabel('Curricular units 1st sem (grade)')
plt.ylabel('Curricular units 2nd sem (grade)')
plt.show()
#%%
y = df['Curricular units 1st sem (approved)']
x = df['Curricular units 2nd sem (approved)']
plt.scatter(x, y)
plt.title('Scatter Plot For Sem')
plt.xlabel('Curricular units 1st sem (approved)')
plt.ylabel('Curricular units 2nd sem (approved)')
plt.show()
#%%
#H0: The mean of 2nd sem approved for graduate and dropout is same
#Ha: The mean of 2nd sem approved for graduate and dropout is different
import scipy.stats as stats

dropout_df = df[df['Target']=='Dropout']
graduate_df = df[df['Target']=='Graduate']

#print(len(dropout_df))
#print(len(graduate_df))
t_stat, p_val = stats.ttest_ind(dropout_df['Curricular units 2nd sem (approved)'], graduate_df['Curricular units 2nd sem (approved)'])
print("T-statistic:", t_stat)
print("P-value:", p_val)

##The T-statistic is a large negative number (-52.07149770749867), which suggests a significant difference between the two groups. The negative sign indicates that the first group's mean is lower than the second group's mean
##A P-value of 0.0 suggests that the probability of observing the data (or more extreme) assuming the null hypothesis is true is extremely low (essentially zero). 
# %%
col_final = ['Application mode', 'Course', 'Previous qualification', "Mother's qualification", 'Tuition fees up to date', 
        "Mother's occupation", 'Gender', 'Scholarship holder', 'Age at enrollment', 'Curricular units 2nd sem (approved)', 'Target']

cat_cols = ['Marital status', 'Application mode', 'Application order', 'Course', 'Daytime/evening attendance\t', 
          'Previous qualification', 'Nacionality', "Mother's qualification", "Father's qualification", 
          "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor', 
          'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International']

df[cat_cols] = df[cat_cols].astype('category')
df = df.replace({'Target': {'Dropout': 0, 'Graduate': 1}})

#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score

df = df[col_final]

# %%
y = df['Target']
X = df.copy()
X = X.drop('Target', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
X_scaled_train = StandardScaler().fit_transform(X_train)
X_scaled_test = StandardScaler().fit_transform(X_test)
lr = LogisticRegression()
lr.fit(X_scaled_train, y_train)
y_preds = lr.predict(X_scaled_test)
conf_matrix = confusion_matrix(y_test, y_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

accuracy = accuracy_score(y_test, y_preds)
print(f"Accuracy: {accuracy}")
precision = precision_score(y_test, y_preds, average='binary')  # Use 'macro' or 'weighted' for multi-class
print(f"Precision: {precision}")
recall = recall_score(y_test, y_preds, average='binary')  # Use 'macro' or 'weighted' for multi-class
print(f"Recall: {recall}")
f1 = f1_score(y_test, y_preds, average='binary')  # Use 'macro' or 'weighted' for multi-class
print(f"F1 Score: {f1}")

y_probs = lr.predict_proba(X_scaled_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
roc_auc = roc_auc_score(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
# %%
