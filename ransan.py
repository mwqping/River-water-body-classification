#!@shijian : 2024/3/15  10:08
# !@Author  : Mawq
# !@File    : ransan.py
import pandas as pd
import numpy as np
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os


np.random.seed(0)
def save_data(data, file_path, new_suffix):
    file_dir, file_name = os.path.split(file_path)
    file_prefix, file_suffix = os.path.splitext(file_name)
    new_name = file_prefix + new_suffix
    new_path = os.path.join(file_dir, new_name)
    data.to_csv(new_path, index=False)
    return new_path


def sliding_ransac(X, window_width, threshold):
    n = len(X)
    results = []
    start = 0
    while start < n:
        end = min(start + window_width, n)
        window_data = X[start:end]

        if len(window_data) > 0:
            ransac = RANSACRegressor()
            ransac.fit(window_data[:, 0].reshape(-1, 1), window_data[:, 1])
            slope = ransac.estimator_.coef_[0]
            results.append((window_data, slope))

        start += window_width // 2

    high_slope = []
    low_slope = []
    for window_data, slope in results:
        if np.abs(slope) > threshold:
            high_slope.append(window_data)
        else:
            low_slope.append(window_data)

    return high_slope, low_slope



def classify_data(atlas03_data, high_slope_windows, low_slope_windows):

    atlas03_data['Class'] = np.nan

    for window_data in high_slope_windows:
        atlas03_data.loc[atlas03_data['Along-Track (m)'].isin(window_data[:, 0]), 'Class'] = 1
    for window_data in low_slope_windows:
        atlas03_data.loc[atlas03_data['Along-Track (m)'].isin(window_data[:, 0]), 'Class'] = 2


    land_data = atlas03_data[atlas03_data['Class'] == 1]
    water_data = atlas03_data[atlas03_data['Class'] == 2]

    return land_data, water_data


root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])

data = pd.read_csv(file_path)
atlas03_data = data.copy()
X = np.array(atlas03_data[['Along-Track (m)', 'Height (m HAE)']])

window_width = 1000
slope_threshold = threshold

high_slope, low_slope = sliding_ransac(X, window_width, slope_threshold)

land_data, water_data = classify_data(atlas03_data, high_slope, low_slope)
new_path = save_data(land_data, file_path, '-land.csv')
new_path = save_data(water_data, file_path, '-water.csv')


plt.figure(figsize=(10, 6))
plt.scatter(land_data['Along-Track (m)'], land_data['Height (m HAE)'], color='red', s=1, marker='.')
plt.scatter(water_data['Along-Track (m)'], water_data['Height (m HAE)'], color='blue', s=1, marker='.')
plt.title('Sliding RANSAC Fit')
plt.xlabel('Along-Track (m)')
plt.ylabel('Height (m HAE)')
plt.show()
