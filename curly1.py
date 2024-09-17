import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Load the dataset 
data = pd.read_csv('/content/Mumbai.csv') 
#Display the first few rows 
print(data.head()) 

# Summary statistics 
print(data.describe()) 

# Data types and missing values 
print(data.info()) 

#Handling missing values (example: fill with mean) 
numeric_columns = data.select_dtypes(include=['float', 'int']).columns 
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean()) 

# Univariate analysis 
plt.figure(figsize=(10, 6)) 
sns.histplot(data['Area'], kde=True) 
plt.show() 

# Bivariate analysis 
plt.figure(figsize=(10, 6)) 
sns.scatterplot(x='Area', y='Price', data=data) 
plt.show() 

# Drop the selected attributes 
 
data_dropped = data.drop(['Hospital','RainWaterHarvesting','Gymnasium', 'SwimmingPool', 
'LandscapedGardens', 'JoggingTrack', 'MaintenanceStaff', 
                          'Resale', 'No. of Bedrooms', 'IndoorGames', 'ShoppingMall', 
                          'Intercom', 'Location', 'SportsFacility', 'ATM', 'ClubHouse', 
                          'LiftAvailable', 'BED', 'VaastuCompliant', 'Microwave', 
                          'GolfCourse', 'TV', 'DiningTable', 'Sofa', 'Wardrobe', 
                          'Refrigerator', 'School', '24X7Security', 'PowerBackup', 'CarParking', 
                          'StaffQuarter', 'Cafeteria', 'MultipurposeRoom', 'MultipurposeRoom', 
                          'WashingMachine', 'Gasconnection', 'AC', 'Wifi', "Children'splayarea"], axis=1) 
 
# data_dropped 
# Calculate the correlation matrix for the remaining attributes 
corr_matrix_dropped = data_dropped.corr() 
 
# Plot the heatmap 
plt.figure(figsize=(12, 8)) 
sns.heatmap(corr_matrix_dropped, annot=True, cmap=('coolwarm')) 
plt.show() 
 
# Group analysis 
grouped_data = data.groupby('Price').mean() 
print(grouped_data)
