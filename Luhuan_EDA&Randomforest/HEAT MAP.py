#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  # Importing matplotlib.pyplot

from scipy.stats import chi2_contingency

#%%
# Assume 'df' is your DataFrame containing categorical columns
students = pd.read_csv('data.csv', delimiter=';')
df = students.copy()
cols = ['Marital status', 'Application mode', 'Application order', 'Course', "Daytime/evening attendance	",
        'Previous qualification', 'Nacionality', "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs', 'Debtor',
        'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International', 'Target']

#%%
# Create an empty DataFrame
cramer_matrix = pd.DataFrame(index=cols, columns=cols)

for i in range(len(cols)):
    for j in range(len(cols)):
        if i != j:
            # Create a contingency table for each pair of categorical variables
            contingency_table = pd.crosstab(df[cols[i]], df[cols[j]])

            # Calculate the chi-squared statistic, p-value, degrees of freedom, and expected values
            chi2, _, _, _ = chi2_contingency(contingency_table)

            # Calculate Cramer's V
            n = contingency_table.sum().sum()
            min_dim = min(contingency_table.shape)
            cramer_v = np.sqrt(chi2 / (n * (min_dim - 1)))

            # Store Cramer's V value in the matrix
            cramer_matrix.loc[cols[i], cols[j]] = cramer_v

#%%
# Convert the values to numeric
cramer_matrix = cramer_matrix.apply(pd.to_numeric)

# Plot the heatmap using Seaborn
plt.figure(figsize=(14, 12))
heatmap = sns.heatmap(cramer_matrix, annot=True, cmap='Reds', fmt=".2f")

heatmap.set_xticklabels(heatmap.get_xticklabels(), rotation=45, fontsize='small')
heatmap.set_yticklabels(heatmap.get_yticklabels(), rotation=0, fontsize='small')

plt.title("Cramer's V Heatmap")
plt.show()
