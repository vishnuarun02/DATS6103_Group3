#%%
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as stats
from scipy.stats import ttest_ind
from scipy.stats import chi2_contingency
from dython.nominal import associations

from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, roc_curve, roc_auc_score, classification_report, ConfusionMatrixDisplay

#%%
pd.set_option('display.max_columns', None) # to display all columns in the dataset without truncating
df_initial = pd.read_csv('/Users/krishnasurya/Documents/3_DM_project/students.csv', delimiter=';') # load the dataset
if 'Daytime/evening attendance\t' in df_initial.columns.to_list():
    df_initial = df_initial.rename(columns={'Daytime/evening attendance\t': 'Attendance Mode'})
else:
    df_initial = df_initial.rename(columns={'Daytime/evening attendance': 'Attendance Mode'})
print(df_initial.shape)

print(df_initial.duplicated().sum()) # check for duplicates
print(df_initial.info())
print(df_initial.isna().sum()) # check for NA values
df = df_initial.copy()

#%%
# Types of values in the target variable and their proportion
target_count = df['Target'].value_counts()
target_colors = ['#3bbf82', '#e67f6e', '#b780f2']

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

bars = ax1.bar(target_count.index, target_count.values, color=target_colors)
ax1.set_xlabel('Target')
ax1.set_ylabel('Count')
ax1.set_title('Bar plot of Target')

for bar, value in zip(bars, target_count.values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha='center', va='bottom')

ax2.pie(target_count, labels=target_count.index, autopct='%1.1f%%', colors=target_colors)
ax2.set_title('Distribution of Target')

plt.tight_layout()
plt.show()

print('\n - The ratio of Graduate and Dropout is not biased \n - The students that are currently Enrolled can be used for applying the developed model in the end')

#%%
# Types of values in the gender variable and their proportion
df = df.replace({'Gender': {0: 'Female', 1: 'Male'}})
gender_count = df['Gender'].value_counts()
gender_colors = ['#edabd6', '#87b5ed']

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

bars = ax1.bar(gender_count.index, gender_count.values, color=gender_colors)
ax1.set_xlabel('Gender')
ax1.set_ylabel('Count')
ax1.set_title('Bar plot of Gender')

for bar, value in zip(bars, gender_count.values):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(value), ha='center', va='bottom')

ax2.pie(gender_count, labels=gender_count.index, autopct='%1.1f%%', colors=gender_colors)
ax2.set_title('Distribution of Gender')

plt.tight_layout()
plt.show()

print('\n - There are more Female students than Male students \n - But the difference is not disproportionate. So it shouldn\'t affect the model much')

gender_color = {'Female': '#edabd6', 'Male': '#87b5ed'}
gender_contingency_table = pd.crosstab(df['Target'], df['Gender'])
gender_percent_contingency_table = pd.crosstab(df['Target'], df['Gender'], normalize='index')

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
ax1 = gender_contingency_table.plot(kind='bar', stacked=True, color=[gender_color[gender] for gender in gender_contingency_table.columns], ax=axes[0])

for bar in ax1.patches:
    width, height = bar.get_width(), bar.get_height()
    x, y = bar.get_xy() 
    ax1.annotate(f'{height:.0f}', (x + width/2, y + height/2), ha='center', va='center', color='black', fontsize=10)

ax1.set_xlabel('Target')
ax1.set_ylabel('Count')
ax1.set_title('Stacked Bar Chart - Gender Distribution within Target')
ax1.legend(title='Gender')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)

ax2 = gender_percent_contingency_table.plot(kind='bar', stacked=True, color=[gender_color[gender] for gender in gender_percent_contingency_table.columns], ax=axes[1])
ax2.set_xlabel('Target')
ax2.set_ylabel('Percentage')
ax2.set_title('Percentage Stacked Bar Chart - Gender Distribution within Target')
ax2.legend(title='Gender')
ax2.set_xticklabels(ax1.get_xticklabels(), rotation=0)

for bar in ax2.patches:
    width, height = bar.get_width(), bar.get_height()
    x, y = bar.get_xy() 
    ax2.annotate(f'{height:.0%}', (x + width/2, y + height/2), ha='center', va='center', color='black', fontsize=10)

plt.tight_layout()
plt.show()

print('The number of female students in comparision to male students are: \n - Almost equal in Dropouts \n - Slight greater in Enrolled \n - Almost three times in Graduates')

chi2, p, _, _ = chi2_contingency(gender_contingency_table)
print(f'Chi-square value: {chi2}')
print(f'P-value: {p}')
print("The Chi-square test of independence yields an extremely low p-value")
print("This means that we can reject the null hypothesis and conclude that there is a statistically significant association between the two variables 'Gender' and 'Target' ")

#%%
# Removing 'Enrolled' students from futher analysis
df = df[df['Target'] != 'Enrolled']

#%%
age_count = df['Age at enrollment'].value_counts().sort_index()
print('Number of unique values in the Age column:', len(age_count))
print('Since the range for this data is widespread, let us rearrange them into bins for better visualization')

bin_edges = [17, 25, 35, 45, 55, 65, 75]
bin_labels = ['17-25', '26-35', '36-45', '46-55', '56-65', '66-75']

df['Age Group'] = pd.cut(df['Age at enrollment'], bins=bin_edges, labels=bin_labels, right=False) # new column with age bins
age_group_counts = df['Age Group'].value_counts().sort_index() # number of occurrences in each bin

# Histogram for the new age group column
plt.bar(age_group_counts.index, age_group_counts.values, color='darkblue')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.title('Distribution of Age Groups')
for i, value in enumerate(age_group_counts.values):
    plt.text(i, value + 0.1, str(value), ha='center', va='bottom', color='black')
plt.show()

age_contingency_table = pd.crosstab(df['Age Group'], df['Target'])
print('\nContingency table for Age and Target')
print(age_contingency_table)

age_percent_contingency_table = pd.crosstab(df['Age Group'], df['Target'], normalize='index')
age_target_colors = {'Graduate': '#3bbf82', 'Dropout': '#e67f6e'}

# Stacked percentage bar chart for distribution of Dropouts and Graduates among the Age groups
ax = age_percent_contingency_table.plot(kind='barh', stacked=True, figsize=(8, 5), color=[age_target_colors[col] for col in age_percent_contingency_table.columns])
plt.ylabel('Age Group')
plt.xlabel('Percentage')
plt.title('Distribution of Dropout and Graduated Students within Age Groups')
plt.legend(title='Target')

for bar in ax.patches:
    width, height = bar.get_width(), bar.get_height()
    x, y = bar.get_xy() 
    ax.annotate(f'{width*100:.1f}%', (x + width/2, y + height/2), ha='center', va='center', color='black')

plt.show()

target_palette = {'Graduate': '#3bbf82', 'Dropout': '#e67f6e'}
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Target', y='Age at enrollment', hue='Target', data=df, palette=target_palette, showfliers=False)
plt.xlabel('Target')
plt.ylabel('Age at enrollment')
plt.title('Distribution of Age by Target')
plt.show()

fig, ax = plt.subplots(figsize=(4,4))
ax.axis('off')  # Turn off the axis to remove unnecessary decorations
table = ax.table(cellText=age_contingency_table.values,
                 colLabels=age_contingency_table.columns,
                 rowLabels=age_contingency_table.index,
                 loc='center', cellLoc='center', colColours=['#f5f5f5']*age_contingency_table.shape[1])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)

plt.title('Contingency Table for Age and Target')
plt.show()

# T-test
dropout = df[df['Target'] == 'Dropout']['Age at enrollment']
graduate = df[df['Target'] == 'Graduate']['Age at enrollment']
t_statistic, p_value_ttest = ttest_ind(dropout, graduate)
print("Independent Samples t-test p-value:", p_value_ttest)
print("The above test yields an extremely low p-value")
print("This indicates that we can reject the null hypothesis and conclude that there is a significant difference in mean age between two groups 'Dropouts' and 'Graduates'")

#%%
df.rename(columns={'Nacionality': 'Nationality'}, inplace=True)
df[['Nationality']] = df[['Nationality']].replace({'Nationality': {1: 'Portuguese', 2: 'German', 6: 'Spanish', 
                                                                   11: 'Italian', 13: 'Dutch', 14: 'English', 
                                                                   17: 'Lithuanian', 21: 'Angolan', 22: 'Cape Verdean', 
                                                                   24: 'Guinean', 25: 'Mozambican', 26: 'Santomean', 
                                                                   32: 'Turkish', 41: 'Brazilian', 62: 'Romanian', 
                                                                   100: 'Moldovan', 101: 'Mexican', 103: 'Ukrainian', 
                                                                   105: 'Russian', 108: 'Cuban', 109: 'Colombian'}})
nationality_count = df[['Nationality']].value_counts()

nationality_count.plot(kind='bar', figsize=(12, 6), colormap='plasma')
plt.xlabel('Nationality')
plt.ylabel('Count')
plt.title('Distribution of Students by Nationality')
plt.xticks(rotation=45, ha='right')
plt.show()

print('There are a huge number of Portuguese students compared to students from any other Nation')
print('Let us try to visualize the data without this entity so we can analyze the rest of the distribution better')

ax = nationality_count[1:].plot(kind='bar', figsize=(12, 6), colormap='plasma')
plt.xlabel('Nationality')
plt.ylabel('Count')
plt.title('Distribution of Students by Nationality (excluding Portuguese)')

plt.xticks(rotation=45, ha='right')
for i, v in enumerate(nationality_count[1:]):
    ax.text(i, v + 0.1, str(v), ha='center', va='bottom', color='black')
plt.show()

plt.figure(figsize=(12, 6))
target_colors = {'Graduate': '#3bbf82', 'Dropout': '#e67f6e'}
sns.countplot(x='Nationality', hue='Target', data=df[df['Nationality']!= 'Portuguese'], palette=target_colors)
plt.xlabel('Nationality')
plt.ylabel('Count')
plt.title('Grouped Bar Chart of Target in Nationality')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Target')

plt.show()

nationality_contingency_table = pd.crosstab(df['Target'], df['Nationality'])
chi2, p, _, _ = chi2_contingency(nationality_contingency_table)
print(f'Chi-square value: {chi2}')
print(f'P-value: {p}')
print("The Chi-square test of independence yields a high p-value of about 0.3 which is higher than the standard 0.05")
print("This suggests that there is not enough evidence to reject the null hypothesis")
print("Therefore we conclude that there isn't a significant association between the two variables 'Nationality' and 'Target' ")

#%%
df = df.replace({'Marital status': {1: 'Single', 2: 'Married', 3: 'Widower', 4: 'Divorced', 5: 'Facto union', 6: 'Legally separated'}})
plt.figure(figsize=(12, 6))
target_colors = {'Graduate': '#3bbf82', 'Dropout': '#e67f6e'}
ax = sns.countplot(x='Marital status', hue='Target', data=df, palette=target_colors)
plt.xlabel('Marital status')
plt.ylabel('Count')
plt.title('Grouped Bar Chart of Target in Marital status')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Target')

for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height), ha='center', va='center', xytext=(0, 10), textcoords='offset points', color='black')

plt.show()

marital_status_contingency_table = pd.crosstab(df['Target'], df['Marital status'])
chi2, p, _, _ = chi2_contingency(marital_status_contingency_table)
print(f'Chi-square value: {chi2}')
print(f'P-value: {p}')
print("The Chi-square test of independence yields an extremely low p-value")
print("This means that we can reject the null hypothesis and conclude that there is a statistically significant association between the two variables 'Marital status' and 'Target' ")

#%%
categorical_cols = ['Marital status', 'Application mode', 'Application order', 'Course', "Attendance Mode",
        'Previous qualification', 'Nationality', "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor',
        'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International', 'Target']

cramer_matrix = pd.DataFrame(index=categorical_cols, columns=categorical_cols)

for i in range(len(categorical_cols)):
    for j in range(len(categorical_cols)):
        if i != j:
            contingency_table = pd.crosstab(df[categorical_cols[i]], df[categorical_cols[j]])
            chi2, _, _, _ = chi2_contingency(contingency_table)
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape)
            cramer_v = np.sqrt(chi2 / (n * (min_dim - 1)))
            cramer_matrix.loc[categorical_cols[i], categorical_cols[j]] = cramer_v

cramer_matrix = cramer_matrix.apply(pd.to_numeric)

plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(cramer_matrix, annot=True, cmap='Reds', fmt=".2f")
heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, fontsize='small')
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize='small')
plt.title("Cramer's V Heatmap")
plt.show()

#%%
#1. variable application mode
df_filtered = df[df['Target'] != 'enrolled']

mode_order = df_filtered['Application mode'].value_counts().index

sns.set_style("whitegrid")
pastel_palette = sns.color_palette("pastel")
plt.figure(figsize=(10, 6))
sns.countplot(x='Target', hue='Application mode', data=df_filtered, hue_order=mode_order, palette=pastel_palette)

plt.xticks([0, 1], ['dropout', 'graduate'])

plt.title('Distribution of Target Variable Across Different Application Modes')
plt.xlabel('Target')
plt.ylabel('No Of Students')

plt.legend(title='Application mode')
plt.xlim(-0.50, 1.0)
plt.show()

#Chi-square test for application mode
#Contingency table
contingency_table = pd.crosstab(df['Application mode'], df['Target'])

#Chi-square test
chi2, p, _, _ = chi2_contingency(contingency_table)

# #Results
print("Chi-square statistic:", chi2)
print("P-value:", p)

# #p-value
alpha = 0.05
if p < alpha:
     print("There is a significant association between Application Mode and Target.")
else:
     print("There is no significant association between Application Mode and Target.")

#%%
#2.variable application order
#exclude category '2'
df_filtered = df[df['Target'] != 2]

#plotstyle
sns.set_style("whitegrid")
pastel_palette = sns.color_palette("pastel")
#countplot
plt.figure(figsize=(10, 6))
sns.countplot(x='Application order', hue='Target', data=df_filtered, palette=pastel_palette)

plt.title('Distribution of Target Variable Across Order of Application')
plt.xlabel('Application Order')
plt.ylabel('Number of Students')

legend_labels = {0: 'Graduated', 1: 'Dropped Out'}
plt.legend(title='Target', labels=legend_labels.values())

plt.show()

#chi-square test for application mode

#typecasting 'Application order' to  categorical variable
df['Application order'] = pd.Categorical(df['Application order'])

#contingency table
contingency_table = pd.crosstab(df['Application order'], df['Target'])

#chi-square test
chi2, p, _, _ = chi2_contingency(contingency_table)

#results
print(f"Chi-square Statistic: {chi2}")
print(f"P-value: {p}")

#p-value
alpha = 0.05
if p < alpha:
    print("There is a significant relationship between Application Order and Target.")
else:
    print("There is no significant relationship between Application Order and Target.")

#%%
#3.variable Daytime/evening attendance
# Map the numerical attendance to categorical



df_analysis = df.copy()
# mapping attendance 
df_analysis['Attendance Mode'] = df_analysis['Daytime/evening attendance'].map({1: 'Daytime', 0: 'Evening'})

# Filter out 'Enrolled' students
df_analysis_filtered = df_analysis[df_analysis['Target'] != 1]

#new column
target_mapping = {0: 'Dropped Out', 2: 'Graduated'}
df_analysis_filtered['Target Category'] = df_analysis_filtered['Target'].map(target_mapping)

#plot
attendance_outcomes = df_analysis_filtered.groupby(['Attendance Mode', 'Target Category']).size().unstack(fill_value=0)
plt.figure(figsize=(10, 6))
attendance_outcomes.plot(kind='bar', stacked=True, color=['#77dd77', '#836953'])
plt.title('Distribution of Target Outcomes for Daytime vs Evening Attendance')
plt.xlabel('Attendance Mode')
plt.ylabel('Number of Students')
plt.xticks(rotation=0)
plt.legend(title='Student Outcome')
plt.tight_layout()
plt.show()


#%%
#4.variable Course
#course IDs wrt names
course_names = { 
     33: 'Biofuel Production Technologies',
     171: 'Animation and Multimedia Design',
     8014: 'Social Service (evening attendance)',
     9003: 'Agronomy',
     9070: 'Communication Design',
     9085: 'Veterinary Nursing',
     9119: 'Informatics Engineering',
     9130: 'Equinculture',
     9147: 'Management',
     9238: 'Social Service',
     9254: 'Tourism',
     9500: 'Nursing',
     9556: 'Oral Hygiene',
     9670: 'Advertising and Marketing Management',
     9773: 'Journalism and Communication',     9853: 'Basic Education', 9991: 'Management (evening attendance)'
}

df['Course Name'] = df['Course'].map(course_names)

#exclude category '2'
df_filtered = df[df['Target'].isin([0, 1])]

#percentage of students in each course category 
df_filtered['count'] = 1  # Add a temporary count column for aggregation
df_percentages = df_filtered.groupby(['Course Name', 'Target'])['count'].count().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
df_percentages.rename(columns={'count': 'Percentage'}, inplace=True)

#plot
plt.figure(figsize=(20, 10))
sns.barplot(x='Course Name', y='Percentage', hue='Target', data=df_percentages, palette=pastel_palette)

#title and labels
plt.title('Distribution of Target Outcomes Across Courses (Percentage)', fontsize=16)
plt.xlabel('Courses', fontsize=20)
plt.ylabel('Percentage of Students', fontsize=20)
plt.xticks(rotation=80, fontsize=18)
plt.yticks(fontsize=24)

#convert y-axis labels to percentages
vals = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in vals])

plt.legend(title='Outcome', fontsize=15, title_fontsize=15, labels=['Graduate', 'Dropout'])

plt.tight_layout()
plt.show()

df_filtered.drop('count', axis=1, inplace=True)


#stat_test
#contingency table
contingency_table = pd.crosstab(df['Course Name'], df['Target'])

#chi-square test
chi2, p, _, _ = chi2_contingency(contingency_table)

#results
print(f"Chi-square Statistic: {chi2}")
print(f"P-value: {p}")

#p-value
alpha = 0.05
if p < alpha:
     print("There is a significant relationship between Course and Target.")
else:
     print("There is no significant relationship between Course  and Target.")


#%%
# cols = [
#     'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
#     'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
#     'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
#     'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
#     'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
#     'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
#     'Unemployment rate', 'Inflation rate', 'GDP']

# n_cols = 3  # Number of columns in subplot grid
# n_rows = (len(cols) + n_cols - 1) // n_cols  # Calculate the number of rows needed

# # Create subplots for box plots
# fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 12))
# plt.subplots_adjust(hspace=0.5)

# # Flatten the axs array for easy iteration
# axs = axs.flatten()

# # Iterate and create box plots
# for index, col in enumerate(cols):
#     sns.boxplot(data=df, x='Target', y=col, showfliers=False, ax=axs[index])
#     axs[index].set(xlabel=None, ylabel=None, title=col)

# # Adjust the layout of box plots
# plt.tight_layout()

# def histplot_visual(data: pd.DataFrame, columns: list[str]) -> None:
#   """Create a histogram plot using a subset of variables specified.

#   Args:
#     data: Input data-frame containing variables we wish to plot.
#     columns: Listing of column-names we wish to plot (must be contained within data).
#   """
#   fig, ax = plt.subplots(3, 5, figsize=(15, 6))
#   fig.suptitle('Histogram for each numeric variable in our data',y=1, size=20)
#   ax=ax.flatten()
#   for i,feature in enumerate(columns):
#     # Setting option `kde=True` allows for a Kernel Density Estimate (i.e. PDF).
#     sns.histplot(data=data[feature],ax=ax[i], kde=True)
#   plt.tight_layout()

# histplot_visual(data=df,columns=cols)

# cols = [
#     'Curricular units 1st sem (credited)', 'Curricular units 1st sem (enrolled)',
#     'Curricular units 1st sem (evaluations)', 'Curricular units 1st sem (approved)',
#     'Curricular units 1st sem (grade)', 'Curricular units 1st sem (without evaluations)',
#     'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
#     'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
#     'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)',
#     'Unemployment rate', 'Inflation rate', 'GDP','Target']
# # Create a separate subplot for the correlation ratio
# fig, ax = plt.subplots(figsize=(12, 8))
# cor_ratio = associations(df[cols], nom_num_assoc='correlation_ratio', num_num_assoc='pearson', ax=ax, cmap='Blues')

# # Show all the plots
# plt.show()

# y = df['Curricular units 1st sem (grade)']
# x = df['Curricular units 2nd sem (grade)']
# plt.scatter(x, y)
# plt.title('Scatter Plot For Grades')
# plt.xlabel('Curricular units 1st sem (grade)')
# plt.ylabel('Curricular units 2nd sem (grade)')
# plt.show()

# y = df['Curricular units 1st sem (approved)']
# x = df['Curricular units 2nd sem (approved)']
# plt.scatter(x, y)
# plt.title('Scatter Plot For Sem')
# plt.xlabel('Curricular units 1st sem (approved)')
# plt.ylabel('Curricular units 2nd sem (approved)')
# plt.show()

# #H0: The mean of 2nd sem approved for graduate and dropout is same
# #Ha: The mean of 2nd sem approved for graduate and dropout is different

# dropout_df = df[df['Target']=='Dropout']
# graduate_df = df[df['Target']=='Graduate']

# print("Performing t-test on Curriculum units 2nd sem(approved) for the groups graduate and dropout")
# print("H0: The mean of Curriculum units 2nd sem(approved) for graduate and dropout is same")
# print("HA: The mean of Curriculum units 2nd sem(approved) for graduate and dropout is different")
# t_stat, p_val = stats.ttest_ind(dropout_df['Curricular units 2nd sem (approved)'], graduate_df['Curricular units 2nd sem (approved)'])
# print("T-statistic:", t_stat)
# print("P-value:", p_val)

# ##The T-statistic is a large negative number (-52.07149770749867), which suggests a significant difference between the two groups. The negative sign indicates that the first group's mean is lower than the second group's mean
# ##A P-value of 0.0 suggests that the probability of observing the data (or more extreme) assuming the null hypothesis is true is extremely low (essentially zero). 

# col_final = ['Age at enrollment', 'Gender', 'Marital status', 'Displaced', 'Debtor', 'Tuition fees up to date',
#         'Scholarship holder', 'Application mode', 'Attendance Mode', 'Course', 'Curricular units 2nd sem (approved)',
#         'Previous qualification', "Mother's occupation", 'Target']

# cat_cols = ['Marital status', 'Application mode', 'Application order', 'Course', 'Attendance Mode', 
#           'Previous qualification', 'Nationality', "Mother's qualification", "Father's qualification", 
#           "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor', 
#           'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International']

# df[cat_cols] = df[cat_cols].astype('category')
# df = df.replace({'Target': {'Dropout': 0, 'Graduate': 1}})

#%%
# Modeling
df = df[df['Target'] != 'Enrolled']
cols = ['Age at enrollment', 'Gender', 'Marital status', 'Displaced', 'Debtor', 'Tuition fees up to date',
        'Scholarship holder', 'Application mode', 'Attendance Mode', 'Course', 'Curricular units 2nd sem (approved)',
        'Previous qualification', "Mother's occupation", 'Target']
df = df[cols]

# Convert into numerical data type.
df = df.replace({'Target': {'Dropout': 0, 'Graduate': 1}})
df = df.replace({'Gender': {'Female': 0, 'Male': 1}})
df = df.replace({'Marital status': {'Single': 0, 'Married': 1, 'Divorced': 2, 'Facto union': 3, 'Legally separated': 4, 'Widower': 5}})
cols = ['Marital status', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'Target']
df[cols] = df[cols].astype('int32')

# Predictor
X = df.copy()
X = X.drop('Target', axis = 1)

# Target
y = df['Target']

# Train Test split: 75% and 25%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Scaling the training data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def model_metrics(y_test, y_pred):
    model_metrics_dict = {}
    model_metrics_dict['accuracy'] = metrics.accuracy_score(y_test, y_pred)
    model_metrics_dict['precision'] = metrics.precision_score(y_test, y_pred)
    model_metrics_dict['recall'] = metrics.recall_score(y_test, y_pred)
    model_metrics_dict['f1_score'] = metrics.f1_score(y_test, y_pred)
    return model_metrics_dict

def confusion_matrix_func(y_test, y_pred): # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Dropout', 'Graduate'], yticklabels=['Dropout', 'Graduate'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for KNN Model')
    plt.show()

def roc_auc(y_test, y_prob): 
    fpr, tpr, thresholds = roc_curve(y_test, y_prob) # ROC curve
    roc_auc = roc_auc_score(y_test, y_prob) # AUC-ROC score
    plt.figure(figsize=(8, 6)) # Plot ROC
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for KNN Model')
    plt.legend(loc='lower right')
    plt.show()

#%%
# KNN model
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': [3, 5, 7, 9, 11, 13],
              'weights': ['uniform', 'distance']}
grid_search = GridSearchCV(knn, param_grid,
                           cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)
print("Best Hyperparameters:", grid_search.best_params_)

knn = grid_search.best_estimator_
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

model_metrics_dict = model_metrics(y_test, y_pred)
print('Accuracy: ', round(model_metrics_dict['accuracy'], 4))
print('Precision: ', round(model_metrics_dict['precision'], 4))
print('Recall: ', round(model_metrics_dict['recall'], 4))
print('F1 score: ', round(model_metrics_dict['f1_score'], 4))

confusion_matrix_func(y_test, y_pred)

y_prob = grid_search.best_estimator_.predict_proba(X_test)[:, 1] #predicted probabilities for positive class
roc_auc(y_test, y_prob)

print("Classification Report for KNN model:\n", classification_report(y_test, y_pred))

#%%
# Random Forest model

# Set up Random Forest classifier and GridSearchCV
rf = RandomForestClassifier(random_state=0)
cv_params = {'n_estimators' : [25, 50, 75, 100],
              'max_depth' : [10, 30, 50, 70],
              'min_samples_leaf' : [0.5, 0.75, 1],
              'min_samples_split' : [0.001, 0.005, 0.01],
              'max_features' : ['sqrt'],
              'max_samples' : [.3,.5,.7,.9]}

rf = RandomForestClassifier(random_state=0)
rf_val = GridSearchCV(rf, cv_params, cv=3, refit='accuracy', n_jobs = -1, verbose = 1)
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

model_metrics_dict = model_metrics(y_test, y_pred)
print('Accuracy: ', round(model_metrics_dict['accuracy'], 4))
print('Precision: ', round(model_metrics_dict['precision'], 4))
print('Recall: ', round(model_metrics_dict['recall'], 4))
print('F1 score: ', round(model_metrics_dict['f1_score'], 4))

confusion_matrix_func(y_test, y_pred)

y_prob = grid_search.best_estimator_.predict_proba(X_test)[:, 1] #predicted probabilities for positive class
roc_auc(y_test, y_prob)

print("Classification Report for Random Forest model:\n", classification_report(y_test, y_pred))

#%%
def perform(y_pred):
    print("Precision : ", precision_score(y_test, y_pred, average = 'micro'))
    print("Recall : ", recall_score(y_test, y_pred, average = 'micro'))
    print("Accuracy : ", accuracy_score(y_test, y_pred))
    print("F1 Score : ", f1_score(y_test, y_pred, average = 'micro'))
    cm = confusion_matrix(y_test, y_pred)
    print("\n", cm)
    print("\n")
    print("*"*27 + "\n" + " "*16 + "Classification Report\n" + "*"*27)
    print(classification_report(y_test, y_pred))
    print("**"*27+"\n")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap=sns.color_palette("pastel"), cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(ticks=[0.5, 1.5], labels=['Dropout', 'Non-Dropout'])
    plt.yticks(ticks=[0.5, 1.5], labels=['Dropout', 'Non-Dropout'])
    plt.show()

#Naive bayes model
model_nb = GaussianNB()
model_nb.fit(X_train, y_train) 
y_pred_nb = model_nb.predict(X_test)  
#model performance
perform(y_pred_nb) 

#Support vector machine model
model_svc = SVC(C=0.1, kernel='linear')
model_svc.fit(X_train, y_train)  
y_pred_svc = model_svc.predict(X_test) 
#model performance
perform(y_pred_svc)


# %%
# Bhoomika's model