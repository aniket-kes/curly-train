import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create a small dummy dataset
data = pd.DataFrame({
    'Area': [850, 950, 1100, 1250, 1500, 2000, 2400, 3000, 3500, 4000],
    'Price': [5000000, 6000000, 7000000, 8500000, 10000000, 12000000, 15000000, 20000000, 25000000, 30000000],
    'Bedrooms': [2, 2, 3, 3, 3, 4, 4, 5, 5, 5],
    'ParkingSpaces': [1, 1, 2, 2, 2, 3, 3, 4, 4, 5]
})

#data = pd.read_csv('/path')

# Display the first few rows
print("First few rows of the dataset:")
print(data.head())

# Summary statistics
print("\nSummary statistics:")
print(data.describe())

# Data types and missing values
print("\nData types and missing values:")
print(data.info())

# Introduce missing values for demonstration
data.loc[2, 'Price'] = np.nan
data.loc[7, 'Area'] = np.nan

# Handling missing values (fill missing price with mean)
numeric_columns = data.select_dtypes(include=['float', 'int']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Univariate analysis - Distribution of 'Area'
plt.figure(figsize=(8, 5))
sns.histplot(data['Area'], kde=True)
plt.title('Distribution of Area')
plt.show()

# Bivariate analysis - Scatter plot of Area vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Area', y='Price', data=data)
plt.title('Scatter plot: Area vs Price')
plt.show()

# Dropping 'ParkingSpaces' and 'Bedrooms' columns
data_dropped = data.drop(['ParkingSpaces', 'Bedrooms'], axis=1)

# Correlation matrix for the remaining attributes
corr_matrix_dropped = data_dropped.corr()

# Plot the heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix_dropped, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Group analysis - Group by 'Bedrooms' and calculate the mean of other attributes
grouped_data = data.groupby('Bedrooms').mean()
print("\nGrouped data by 'Bedrooms':")
print(grouped_data)
