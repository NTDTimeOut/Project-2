import pandas as pd
import os
import numpy as np
import math
from scipy.stats import skew, kurtosis, entropy

data = pd.read_csv(r'..\Data_set\results\IDGenderAgelist.csv', dtype = str)
id = list(data.ID)
gender = list(data.Gender)
size = len(id)

def process_data(name, data):
    data_process = []
    # Mean
    mean = np.mean(data[name])
    data_process.append(mean)

    # Median
    median = np.median(data[name])
    data_process.append(median)

    # Minimum
    minimum = np.min(data[name])
    data_process.append(minimum)

    # Maximum
    maximum = np.max(data[name])
    data_process.append(maximum)

    # Skewness
    data_skewness = skew(data[name])
    data_process.append(data_skewness)

    # Mean Absolute Deviation
    mean_abs_deviation = np.mean(np.abs(data[name] - np.mean(data[name])))
    data_process.append(mean_abs_deviation)

    # Standard Error of Mean
    std_error_mean = np.std(data[name]) / math.sqrt(len(data))
    data_process.append(std_error_mean)

    # Standard Deviation
    std_deviation = np.std(data[name])
    data_process.append(std_deviation)

    # Kurtosis
    data_kurtosis = kurtosis(data[name])
    data_process.append(data_kurtosis)

    # Variance
    variance = np.var(data[name])
    data_process.append(variance)

    # Entropy
    data_entropy = entropy(data[name])
    data_process.append(data_entropy)

    # Root Mean Square
    rms = math.sqrt(np.mean(np.square(data[name])))
    data_process.append(rms)

    # Vector Sum
    vector_sum = np.sum(data[name])
    data_process.append(vector_sum)

    # Vector Sum of Mean
    vector_sum_mean = np.sum(data[name]) / len(data)
    data_process.append(vector_sum_mean)

    # Vector Sum of Median
    vector_sum_median = np.sum(np.median(data[name]))
    data_process.append(vector_sum_median)

    # Vector Sum of Maximum
    vector_sum_maximum = np.sum(np.max(data[name]))
    data_process.append(vector_sum_maximum)

    # Vector Sum of Minimum
    vector_sum_minimum = np.sum(np.min(data[name]))
    data_process.append(vector_sum_minimum)

    # Vector Sum of Standard Deviation
    vector_sum_std_deviation = np.sum(np.std(data[name]))
    data_process.append(vector_sum_std_deviation)

    # Vector Sum of Square Error of Deviation
    vector_sum_sq_err_deviation = np.sum(np.square(data[name] - np.mean(data[name])))
    data_process.append(vector_sum_sq_err_deviation)

    # Vector Sum of Skewness
    vector_sum_skewness = np.sum(skew(data[name]))
    data_process.append(vector_sum_skewness)

    # Vector Sum of Mean Absolute Deviation
    vector_sum_mean_abs_deviation = np.sum(np.mean(np.abs(data[name] - np.mean(data[name]))))
    data_process.append(vector_sum_mean_abs_deviation)

    # Vector Sum of Kurtosis
    vector_sum_kurtosis = np.sum(kurtosis(data[name]))
    data_process.append(vector_sum_kurtosis)

    # Vector Sum of Variance
    vector_sum_variance = np.sum(np.var(data[name]))
    data_process.append(vector_sum_variance)

    # Vector Sum of Root Mean Square
    vector_sum_rms = np.sum(math.sqrt(np.mean(np.square(data[name]))))
    data_process.append(vector_sum_rms)

    return data_process

def data_processing_train (id,i):
    id = id
    try:
        dataset_file = f'../Data_set/Data/T0_ID{id}_Walk1.csv' #[] this_i = 0
        print(dataset_file)
        dataset_id = os.path.basename(dataset_file).split('_')[1]
        data = pd.read_csv(dataset_file, skiprows = 2, names=['Gx','Gy','Gz','Ax','Ay','Az'])
    except FileNotFoundError: print('in error'); return
    
    Ax_process = process_data('Ax',data)
    Ay_process = process_data('Ay',data)
    Az_process = process_data('Az',data)
    Gx_process = process_data('Gx',data)
    Gy_process = process_data('Gy',data)
    Gz_process = process_data('Gz',data)

    df = pd.DataFrame({
        'Ax': Ax_process,
        'Ay': Ay_process,
        'Az': Az_process,
        'Gx': Gx_process,
        'Gy': Gy_process,
        'Gz': Gz_process,
    })

    # Đường dẫn thư mục tùy ý
    folder_path = '../Data_set/Data_train2'
    if(gender[i]=='0'):
        output_file = os.path.join(folder_path, dataset_id + '_Walk1_0.csv')
    else:
        output_file = os.path.join(folder_path, dataset_id + '_Walk1_1.csv')
    # Xuất DataFrame vào file Excel
    print(output_file)
    try:
        df.to_csv(output_file, index=False)
    except FileExistsError: print('out error'); exit()

def data_processing_test (id,i):
    id = id
    try:
        dataset_file = f'../Data_set/Data2/T0_ID{id}_Walk2.csv' #[] this_i = 0
        print(dataset_file)
        dataset_id = os.path.basename(dataset_file).split('_')[1]
        data = pd.read_csv(dataset_file, skiprows = 2, names=['Gx','Gy','Gz','Ax','Ay','Az'])

    except FileNotFoundError: print('in error'); return
    
    Ax_process = process_data('Ax',data)
    Ay_process = process_data('Ay',data)
    Az_process = process_data('Az',data)
    Gx_process = process_data('Gx',data)
    Gy_process = process_data('Gy',data)
    Gz_process = process_data('Gz',data)

    df = pd.DataFrame({
        'Ax': Ax_process,
        'Ay': Ay_process,
        'Az': Az_process,
        'Gx': Gx_process,
        'Gy': Gy_process,
        'Gz': Gz_process,
    })

    # Đường dẫn thư mục tùy ý
    folder_path = '../Data_set/Data_test2'
    if(gender[i]=='0'):
        output_file = os.path.join(folder_path, dataset_id + '_Walk2_0.csv')
    else:
        output_file = os.path.join(folder_path, dataset_id + '_Walk2_1.csv')
    # Xuất DataFrame vào file Excel
    print(output_file)
    try:
        df.to_csv(output_file, index=False)
    except FileExistsError: print('out error'); exit()

for i in range(size):
    data_processing_test (id[i],i)
    data_processing_train(id[i],i)
