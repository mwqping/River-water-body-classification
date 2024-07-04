#!@shijian : 2024/5/26  17:13
# !@Author  : Mawq
# !@File    : pdf.py
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, argrelextrema
from scipy.stats import gaussian_kde, iqr

def open_file_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[('CSV Files', '*.csv')])
    return file_path

def butter_lowpass_filter(data, cutoff_freq, sample_rate, order):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def sliding_window_std(y_data, x_data, window_size):
    std_values = []
    for i in range(len(x_data) - window_size + 1):
        window_data = y_data[i:i + window_size]
        window_x = x_data[i:i + window_size]
        window_center = np.mean(window_x)
        std_values.append((window_center, np.std(window_data)))
    return np.array(std_values)

print("open data：")
file_path = open_file_dialog()
atlas03_data = pd.read_csv(file_path)
file_dir, file_name = os.path.split(file_path)
file_prefix, file_suffix = os.path.splitext(file_name)
X = np.array(atlas03_data[['Along-Track (m)', 'Height (m HAE)']])
x = X[:, 0]
y = X[:, 1]

# Select parameters based on data characteristics
cutoff_frequency = f
sample_rate = s
order = N
y_filtered = butter_lowpass_filter(y, cutoff_frequency, sample_rate, order)


window_size = 1000
std_values = sliding_window_std(y_filtered, x, window_size)

# Extract standard deviation value
std_values_y = std_values[:, 1]

bins = int((std_values_y.max() - std_values_y.min()) / 0.1)

h = 0.9*min(np.std(std_values_y), iqr(std_values_y)/1.34) * len(std_values_y) ** (-1/5)

# Plot a histogram of the standard deviation. Determine the classification threshold based on the peak of the PDF curve
std_values_x, std_values_y = std_values[:, 0], std_values[:, 1]
kde = gaussian_kde(std_values_y, bw_method=h)
density = kde(std_values_y)
plt.figure(figsize=(8, 6))
plt.plot(std_values_y, density, color='red', label='PDF')
plt.hist(std_values_y, bins=bins, density=True, color='blue', alpha=0.6, label='Density')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel('Standard Deviation', fontsize=18)
plt.ylabel('Density', fontsize=18)
plt.legend(prop={'size': 14})
plt.show()
