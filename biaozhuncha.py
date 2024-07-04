#!@shijian : 2024/2/27  20:19
# !@Author  : Mawq
# !@File    : biaozhuncha.py
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, argrelextrema


def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    return file_path


def save_data(data, file_path, new_suffix):
    file_dir, file_name = os.path.split(file_path)
    file_prefix, file_suffix = os.path.splitext(file_name)
    new_name = file_prefix + new_suffix
    new_path = os.path.join(file_dir, new_name)
    data.to_csv(new_path, index=False)
    return new_path



def butter_lowpass_filter(data, cutoff_freq, sample_rate, order):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def find_turning_points(y_data):

    minima = argrelextrema(y_data, np.less_equal)[0]
    maxima = argrelextrema(y_data, np.greater_equal)[0]
    turning_points = np.concatenate((minima, maxima))
    return np.array(turning_points)


def sliding_window_std(y_data, x_data, window_size):
    std_values = []

    for i in range(len(x_data) - window_size + 1):
        window_data = y_data[i:i + window_size]
        window_x = x_data[i:i + window_size]
        window_center = np.mean(window_x)

        std_values.append((window_center, np.std(window_data)))

    return np.array(std_values)


def classify_std_values(std_values, threshold):
    # Classify elevation standard deviation according to classification threshold

    high_std_mask = std_values[:, 1] > threshold
    high_std = std_values[high_std_mask]
    low_std = std_values[~high_std_mask]

    # Get the index of the sliding window that meets the conditions
    low_std_indices = np.where(~high_std_mask)[0]

    return high_std, low_std, low_std_indices


def extract_low_std_windows(x_data, y_data, indices, window_size, trim_size):
    #  Extract sliding window data that is no greater than the standard deviation threshold
    low_std_windows_x = []
    low_std_windows_y = []

    for i in indices:
        window_x = x_data[i:i + window_size]
        window_y = y_data[i:i + window_size]

        low_std_windows_x.append(window_x)
        low_std_windows_y.append(window_y)

    return low_std_windows_x, low_std_windows_y


def extract_low_std_point_cloud(x_data, y_data, y_filtered, low_std_windows_y):
    #  Extract point cloud data below the threshold
    low_std_point_cloud_x = []
    low_std_point_cloud_y = []
    low_indices = []

    for i in range(len(low_std_windows_y)):
        # Use filtered curve data to search in raw data
        indices = np.where(np.isin(y_filtered, low_std_windows_y[i]))[0]

        low_std_point_cloud_x.extend(x_data[indices])
        low_std_point_cloud_y.extend(y_data[indices])
        low_indices.extend(indices[90:-60])

    low_std_point_cloud_x = np.array(low_std_point_cloud_x)
    low_std_point_cloud_y = np.array(low_std_point_cloud_y)
    low_indices = np.array(low_indices)

    return low_std_point_cloud_x, low_std_point_cloud_y, low_indices


print("open data：")
file_path = open_file_dialog()
atlas03_data = pd.read_csv(file_path)
file_dir, file_name = os.path.split(file_path)
file_prefix, file_suffix = os.path.splitext(file_name)
X = np.array(atlas03_data[['Along-Track (m)', 'Height (m HAE)']])
x = X[:, 0]
y = X[:, 1]

#  lowpass filtering. Select parameters based on data characteristics
cutoff_frequency = f
sample_rate = s
order = N
y_filtered = butter_lowpass_filter(y, cutoff_frequency, sample_rate, order)


window_size = 1000  # Set the sliding window size
std_values = sliding_window_std(y_filtered, x, window_size)

turning_points_lowpass = find_turning_points(y_filtered)

# Set the elevation standard deviation threshold
threshold_std = threshold

high_std, low_std, low_std_indices = classify_std_values(std_values, threshold_std)
# Extract the filtered curve data within the sliding window that produces an elevation standard deviation not greater than the threshold
low_std_windows_x, low_std_windows_y = extract_low_std_windows(x, y_filtered, low_std_indices, window_size, trim_size)
# Extract standard deviation data
x_windowed, std_values = std_values[:, 0], std_values[:, 1]
low_std_point_cloud_x, low_std_point_cloud_y, low_indices = extract_low_std_point_cloud(x, y, y_filtered, low_std_windows_y)


# Keep the original point cloud data within the sliding window that does not produce an elevation standard deviation below the threshold
water_atlas03_data = atlas03_data[atlas03_data.index.isin(low_indices)]
land_atlas03_data = atlas03_data[~atlas03_data.index.isin(low_indices)]
#  save data
new_path = save_data(water_atlas03_data, file_path, '-water.csv')
new_path = save_data(land_atlas03_data, file_path, '-land.csv')


#  Plotting Results
'''plt.figure(figsize=(10, 7))
plt.subplot(4, 1, 1)'''
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label='Signal Points', s=1)
plt.plot(x, y_filtered, color='red', label='Low-pass Filtered')
'''plt.scatter(x[turning_points_lowpass], y_filtered[turning_points_lowpass], color='purple', s=3,
            label='Low-pass Filtered Points')'''
plt.plot(x_windowed, std_values, label=f'Sliding Window Std', color='blue')

plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

'''plt.title(f'Scatter Plot with Low-pass Filtering - cutoff_frequency={cutoff_frequency}')
'''
plt.xlabel('Along-Track (m)', fontsize=18)
plt.ylabel('Height (m HAE)', fontsize=18)
plt.legend(loc='upper right', prop={'size': 14})
plt.show()


'''plt.subplot(4, 1, 3)'''
plt.figure(figsize=(8, 6))
plt.scatter(high_std[:, 0], high_std[:, 1], color='red', label=f'High Std', s=1)
plt.scatter(low_std[:, 0], low_std[:, 1], color='blue', label=f'Low Std', s=1)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Along-Track (m)', fontsize=18)
plt.ylabel('Standard Deviation', fontsize=18)
'''plt.title(f'Classified Standard Deviation - Threshold')
'''
plt.legend(loc='upper right', prop={'size': 14})
plt.show()
'''
plt.subplot(4, 1, 2)'''
plt.figure(figsize=(8, 6))
plt.scatter(x_windowed, std_values, label=f'Sliding Window Std', color='blue', s=5)
plt.axhline(y=threshold_std, color='r', linestyle='--', label='Threshold')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Along-Track (m)', fontsize=18)
plt.ylabel('Standard Deviation', fontsize=18)
'''plt.title(f'Standard Deviation with Sliding Window - Window Size={window_size}')
'''
plt.legend(loc='upper right', prop={'size': 14})
plt.show()


'''plt.subplot(4, 1, 4)'''
plt.figure(figsize=(8, 6))
plt.scatter(land_atlas03_data['Along-Track (m)'], land_atlas03_data['Height (m HAE)'], color='red', s=1,
            label=f'Land Points')
plt.scatter(water_atlas03_data['Along-Track (m)'], water_atlas03_data['Height (m HAE)'], color='blue', s=1,
            label=f'Water Points')
'''plt.scatter(low_std_point_cloud_x, low_std_point_cloud_y, color='blue', s=1,
            label=f'Water Points')'''
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Along-Track (m)', fontsize=18)
plt.ylabel('Height (m HAE)', fontsize=18)
'''plt.title(f'Classified Photon Points are Land and Water')
'''
plt.legend(loc='upper right', prop={'size': 14})

plt.show()

