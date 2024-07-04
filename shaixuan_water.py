#!@shijian : 2024/3/4  16:54
# !@Author  : Mawq
# !@File    : dbscan_shaixuan_water.py
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

def save_data(data, file_path, new_suffix):
    file_dir, file_name = os.path.split(file_path)
    file_prefix, file_suffix = os.path.splitext(file_name)
    new_name = file_prefix + new_suffix
    new_path = os.path.join(file_dir, new_name)
    data.to_csv(new_path, index=False)
    return new_path

print("open water data：")
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
atlas03_data = pd.read_csv(file_path)

# Select feature columns for clustering (select only 'Along-Track (m)')
features = atlas03_data[['Along-Track (m)']]

# Standardizing Data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Clustering using DBSCAN
dbscan = DBSCAN(eps=0.01, min_samples=10)
atlas03_data['Cluster'] = dbscan.fit_predict(scaled_features)

# Traverse each cluster and determine whether the difference between the maximum and minimum values of Along-Track (m) is less than 100
for cluster_id in atlas03_data['Cluster'].unique():
    cluster_data = atlas03_data[atlas03_data['Cluster'] == cluster_id]
    min_along_track = cluster_data['Along-Track (m)'].min()
    max_along_track = cluster_data['Along-Track (m)'].max()

    if max_along_track - min_along_track < 100:
        atlas03_data.loc[atlas03_data['Cluster'] == cluster_id, 'Category'] = 1
    else:
        atlas03_data.loc[atlas03_data['Cluster'] == cluster_id, 'Category'] = 2

land_data = atlas03_data[atlas03_data['Category'] == 1]
water_data = atlas03_data[atlas03_data['Category'] == 2]

print("open land data：")
file_path_2 = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
additional_data = pd.read_csv(file_path_2)
# Merge land data
merged_data = pd.concat([land_data, additional_data], ignore_index=True)

new_path = save_data(merged_data, file_path_2, '-land.csv')
new_path = save_data(water_data, file_path, '-water.csv')


plt.figure(figsize=(10, 7))
'''plt.scatter(atlas03_data['Along-Track (m)'], atlas03_data['Height (m HAE)'], c=atlas03_data['Category'], cmap='viridis', s=1)'''
plt.scatter(land_data['Along-Track (m)'], land_data['Height (m HAE)'], c='r', s=1, label='Water Points')
plt.scatter(water_data['Along-Track (m)'], water_data['Height (m HAE)'], c='b', s=1, label='Land Points')
'''plt.title('Precise Extraction of Water Point Clouds')
'''
plt.xlabel('Along-Track (m)')
plt.ylabel('Height (m HAE)')
plt.show()

