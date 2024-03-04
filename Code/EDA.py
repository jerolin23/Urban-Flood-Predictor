
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
from folium import Choropleth, CircleMarker
from folium.plugins import MarkerCluster
from scipy.stats import zscore
import seaborn as sns
from IPython.display import display
import webbrowser
from sklearn.cluster import KMeans


file = 'Hyderabad.csv'
df = pd.read_csv('Hyderabad.csv')

summary_stats = df.describe()
print(summary_stats)

missing_values = df.isnull().sum()
print(missing_values)

data_types = df.dtypes
print(data_types)

# Univariate Analysis
sns.histplot(df['bu_2000'], bins=20, kde=True)
plt.title('Built-up Area in 2000')
plt.show()

sns.boxplot(x='w_area', data=df)
plt.title('Distribution of w_area')
plt.show()

sns.heatmap(df[['bu_2000', 'bu_2022', 'ndvi_2000', 'ndvi_2022']].corr(), annot=True)
plt.title('Correlation Heatmap')
plt.show()

df[['PRECTOTCORR_2000', 'PRECTOTCORR_2022']].plot(kind='line', figsize=(10, 6))
plt.title('Line Chart of PRECTOTCORR_2000 and PRECTOTCORR_2022')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend(['PRECTOTCORR_2000', 'PRECTOTCORR_2022'])
plt.show()

#Distribution of Builtup Area
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df['bu_2000'].plot(kind='hist', title='Distribution of Built-up Area 2000')
plt.subplot(1, 2, 2)
df['bu_2022'].plot(kind='hist', title='Distribution of Built-up Area 2022')
plt.show()

column_to_plot = 'PRECTOTCORR_2000'

# Plot the distribution using a histogram
plt.figure(figsize=(10, 6))
plt.hist(df[column_to_plot], bins=30, edgecolor='black')
plt.title(f'Distribution of {column_to_plot}')
plt.xlabel(column_to_plot)
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

column_to_plot = 'PRECTOTCORR_2022'

# Plot the distribution using a histogram
plt.figure(figsize=(10, 6))
plt.hist(df[column_to_plot], bins=30, edgecolor='black')
plt.title(f'Distribution of {column_to_plot}')
plt.xlabel(column_to_plot)
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Assuming 'buchg_abs' and 'PRECTOTCORR_2000' are column names in your DataFrame
columns_of_interest = ['buchg_abs', 'PRECTOTCORR_2000']

# Select the columns of interest from the DataFrame
selected_columns = df[columns_of_interest]

# Calculate the correlation matrix
correlation_matrix = selected_columns.corr()

# Plot the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Plot between buchg_abs and PRECTOTCORR_2000')
plt.show()

x_column = 'buchg_abs'
y_column = 'PRECTOTCORR_2000'
# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df[x_column], df[y_column], alpha=0.5)
plt.title(f'Scatter Plot between {x_column} and {y_column}')
plt.xlabel(x_column)
plt.ylabel(y_column)
plt.grid(True)
plt.show()

#Builtup Area Change
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df['buchg_abs'].plot(kind='hist', title='Distribution of Built-up Change (Absolute)')
plt.subplot(1, 2, 2)
df['buchg_per'].plot(kind='hist', title='Distribution of Built-up Change (Percentage)')
plt.show()

#NDVI change
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
df['ndvichg_abs'].plot(kind='hist', title='Distribution of NDVI Change (Absolute)')
plt.subplot(1, 2, 2)
df['ndvichg_per'].plot(kind='hist', title='Distribution of NDVI Change (Percentage)')
plt.show()

# ELevation and Slope
elevation_stats = df[['elv-min', 'elv_max', 'elv_rng', 'elv_mean']].describe()
slope_stats = df[['slp_min', 'slp_max', 'slp_rng', 'slp_mean']].describe()

print("Elevation Statistics:")
print(elevation_stats)
print("\nSlope Statistics:")
print(slope_stats)

#Geographical Distribution
# Assuming 'Latitude' and 'Longitude' columns may contain NaN values
df_clean = df.dropna(subset=['Latitude', 'Longitude'])

# Create a map centered around the mean latitude and longitude
m = folium.Map(location=[df_clean['Latitude'].mean(), df_clean['Longitude'].mean()], zoom_start=10)

# Create a HeatMap layer for Built-up Area changes in 2000-2022
heat_data = [[row['Latitude'], row['Longitude'], row['bu_2022'] - row['bu_2000']] for _, row in df_clean.iterrows()]
HeatMap(heat_data, name='Built-up Area Change (2000-2022)', gradient={0.2: 'blue', 0.4: 'green', 0.6: 'yellow', 0.8: 'orange', 1.0: 'red'}).add_to(m)

# Save the map as an HTML file
m.save("folium_heatmap.html")

# Open the HTML file in the default web browser
webbrowser.open("folium_heatmap.html")


#Correlation Analysis
numeric_columns = df_clean.select_dtypes(include=['float64', 'int64']).columns
numeric_df = df_clean[numeric_columns]

# Compute the correlation matrix
correlation_matrix = numeric_df.corr()

# Display the correlation matrix
print(correlation_matrix)

# Demograohic Analysis
demographic_stats = df[['tot_p', 'tot_m', 'tot_f']].describe()
print(demographic_stats)

#Meteorological Features
meteorological_stats_2000 = df[['TS_2000', 'QV2M_2000', 'PRECTOTCORR_2000', 'WS10M_MAX_2000', 'WS50M_MAX_2000']].describe()
meteorological_stats_2022 = df[['TS_2022', 'QV2M_2022', 'PRECTOTCORR_2022', 'WS10M_MAX_2022', 'WS50M_MAX_2022']].describe()

print("Meteorological Statistics for 2000:")
print(meteorological_stats_2000)
print("\nMeteorological Statistics for 2022:")
print(meteorological_stats_2022)

# Time Series Analysis

# Example: Plot time series of Built-up Area for 2000
plt.figure(figsize=(10, 5))
df['bu_2000'].plot(title='Time Series of Built-up Area in 2000')
plt.xlabel('Date')
plt.ylabel('Built-up Area')
plt.show()

#Outlier Detection
# Using z-score for outlier detection (replace 'feature_column' with the actual column name)
from scipy.stats import zscore

z_scores = zscore(df['bu_2000'])
outliers = (z_scores > 3) | (z_scores < -3)
outlier_indices = df.index[outliers]

# Print or handle the outlier indices as needed
print("Outlier Indices:", outlier_indices)

#K-means Clustering

X = df[['bu_2000', 'bu_2022']]  # Use relevant features for clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust the number of clusters as needed
df['cluster'] = kmeans.fit_predict(X)

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(df['bu_2000'], df['bu_2022'], c=df['cluster'], cmap='viridis')
plt.title('K-means Clustering')
plt.xlabel('bu_2000')
plt.ylabel('bu_2022')
plt.show()

#Feature Engineering
# Example: Create a density feature based on built-up area and area size
df['bu_density'] = df['bu_2000'] / df['w_area']

# Example: Create a ratio feature between total population and built-up area
df['pop_bu_ratio'] = df['tot_p'] / df['bu_2000']

# Visualization
selected_features = ['bu_2000', 'ndvi_2000', 'elv_mean', 'tot_p']
sns.pairplot(df[selected_features])
plt.show()