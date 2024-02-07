import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Replace 'your_dataset.csv' with your actual file path
df = pd.read_csv('dataset/btc_2014_to_2024.csv')


df.hist(figsize=(10, 10))  # Visualize histograms for all features
plt.subplots_adjust(bottom=0.1)  # Adjust spacing to avoid overlapping labels
plt.show()

sns.distplot(df['Close'], kde=True)  # Focus on the target variable (Close)
plt.show()

sns.pairplot(df, diag_kind='kde')  # Visualize pairwise relationships
plt.show()

sns.scatterplot(x='Date', y='Close', data=df)  # Focus on time series behavior
plt.show()

#Analyse missing values
print(df.isnull().sum())
stats=df.describe()
print(stats)
df.dtypes

# Check for duplicate rows
duplicate_rows = df[df.duplicated()]

# Display duplicate rows if any
if not duplicate_rows.empty:
    print("Duplicate rows found:")
    print(duplicate_rows)
else:
    print("No duplicate rows found.")
