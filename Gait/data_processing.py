import pandas as pd
import os
import numpy as np

data = pd.read_csv(r'Data_set\results\IDGenderAgelist.csv', dtype = str)
id = list(data.ID)
gender = list(data.Gender)
size = len(id)

# max_data_length = 100 # Độ dài mong muốn cho mỗi tệp CSV

def truncate_or_pad_data(data, length):
    if len(data) < length:
        # Đệm dữ liệu bằng số 0 ở cuối
        padded_data = np.pad(data, (0, length - len(data)), mode='constant')
        return padded_data
    elif len(data) > length:
        # Cắt bớt dữ liệu theo độ dài mong muốn
        truncated_data = data[:length]
        return truncated_data
    else:
        return data

def data_processing_train (id,i, max_data_length):
    id = id
    try:
        dataset_file = f'./Data_set/Data/T0_ID{id}_Walk1.csv' #[] this_i = 0
        print(dataset_file)
        dataset_id = os.path.basename(dataset_file).split('_')[1]
        dataset = pd.read_csv(dataset_file, skiprows = 2, names=['Gx','Gy','Gz','Ax','Ay','Az'])
        N = dataset['Ax'].size
    except FileNotFoundError: print('in error'); return
    t_step = 0.01
    t = np.arange(0,(N-0.5)*t_step, t_step)

    Gx_y = dataset['Gx']
    Gy_y = dataset['Gy']
    Gz_y = dataset['Gz']
    Ax_y = dataset['Ax']
    Ay_y = dataset['Ay']
    Az_y = dataset['Az']

    Gx = np.fft.fft(Gx_y)
    Gx_plot = Gx[0:int(t.size/2+1)]

    Gy = np.fft.fft(Gy_y)
    Gy_plot = Gy[0:int(t.size/2+1)]

    Gz = np.fft.fft(Gz_y)
    Gz_plot = Gz[0:int(t.size/2+1)]

    Ax = np.fft.fft(Ax_y)
    Ax_plot = Ax[0:int(t.size/2+1)]

    Ay = np.fft.fft(Ay_y)
    Ay_plot = Ay[0:int(t.size/2+1)]

    Az = np.fft.fft(Az_y)
    Az_plot = Az[0:int(t.size/2+1)]

    t_plot = np.linspace(0, 100, Gx_plot.size)

    frequency = 50  # Tần số cần tìm

    indices = np.where(t_plot == frequency)[0]

    if len(indices) == 0:
        # Tìm vị trí của phần tử gần nhất với hoặc bằng frequency trong mảng t_plot
        closest_index = np.argmin(np.abs(t_plot - frequency))
        indices = np.append(indices, closest_index)  # Thêm closest_index vào indices
        print("Vị trí của phần tử gần 50:", indices)

    else:
        # In ra vị trí của các phần tử có tần số bằng frequency
        print("Vị trí của phần tử có tần số 50:", indices)

    # Thiết kế bộ lọc thông thấp
    cutoff_frequency = indices[0]  # Tần số cắt (cutoff frequency)

    # Áp dụng bộ lọc
    filtered_Gx = Gx_plot[0:cutoff_frequency]
    filtered_Gy = Gy_plot[0:cutoff_frequency]
    filtered_Gz = Gz_plot[0:cutoff_frequency]
    filtered_Ax = Ax_plot[0:cutoff_frequency]
    filtered_Ay = Ay_plot[0:cutoff_frequency]
    filtered_Az = Az_plot[0:cutoff_frequency]

    t_plot2 = np.linspace(0, 50, filtered_Gx.size)

# Cắt bớt hoặc đệm dữ liệu đã lọc theo độ dài mong muốn
    filtered_Gx = truncate_or_pad_data(filtered_Gx, max_data_length)
    filtered_Gy = truncate_or_pad_data(filtered_Gy, max_data_length)
    filtered_Gz = truncate_or_pad_data(filtered_Gz, max_data_length)
    filtered_Ax = truncate_or_pad_data(filtered_Ax, max_data_length)
    filtered_Ay = truncate_or_pad_data(filtered_Ay, max_data_length)
    filtered_Az = truncate_or_pad_data(filtered_Az, max_data_length)

    df = pd.DataFrame({
        'Gx': np.abs(filtered_Gx.astype(complex)),
        'Gy': np.abs(filtered_Gy.astype(complex)),
        'Gz': np.abs(filtered_Gz.astype(complex)),
        'Ax': np.abs(filtered_Ax.astype(complex)),
        'Ay': np.abs(filtered_Ay.astype(complex)),
        'Az': np.abs(filtered_Az.astype(complex))
    })

    # Đường dẫn thư mục tùy ý
    folder_path = './Data_set/Data_train'
    if(gender[i]=='0'):
        output_file = os.path.join(folder_path, dataset_id + '_Walk1_0.csv')
    else:
        output_file = os.path.join(folder_path, dataset_id + '_Walk1_1.csv')
    # Xuất DataFrame vào file Excel
    print(output_file)
    try:
        df.to_csv(output_file, index=False)
    except FileExistsError: print('out error'); exit()

def data_processing_test (id,i, max_data_length):
    id = id
    try:
        dataset_file = f'./Data_set/Data2/T0_ID{id}_Walk2.csv' #[] this_i = 0
        print(dataset_file)
        dataset_id = os.path.basename(dataset_file).split('_')[1]
        dataset = pd.read_csv(dataset_file, skiprows = 2, names=['Gx','Gy','Gz','Ax','Ay','Az'])
        N = dataset['Ax'].size
    except FileNotFoundError: print('in error'); return
    t_step = 0.01
    t = np.arange(0,(N-0.5)*t_step, t_step)

    Gx_y = dataset['Gx']
    Gy_y = dataset['Gy']
    Gz_y = dataset['Gz']
    Ax_y = dataset['Ax']
    Ay_y = dataset['Ay']
    Az_y = dataset['Az']

    Gx = np.fft.fft(Gx_y)
    Gx_plot = Gx[0:int(t.size/2+1)]

    Gy = np.fft.fft(Gy_y)
    Gy_plot = Gy[0:int(t.size/2+1)]

    Gz = np.fft.fft(Gz_y)
    Gz_plot = Gz[0:int(t.size/2+1)]

    Ax = np.fft.fft(Ax_y)
    Ax_plot = Ax[0:int(t.size/2+1)]

    Ay = np.fft.fft(Ay_y)
    Ay_plot = Ay[0:int(t.size/2+1)]

    Az = np.fft.fft(Az_y)
    Az_plot = Az[0:int(t.size/2+1)]

    t_plot = np.linspace(0, 100, Gx_plot.size)

    frequency = 50  # Tần số cần tìm

    indices = np.where(t_plot == frequency)[0]

    if len(indices) == 0:
        # Tìm vị trí của phần tử gần nhất với hoặc bằng frequency trong mảng t_plot
        closest_index = np.argmin(np.abs(t_plot - frequency))
        indices = np.append(indices, closest_index)  # Thêm closest_index vào indices
        print("Vị trí của phần tử gần 50:", indices)

    else:
        # In ra vị trí của các phần tử có tần số bằng frequency
        print("Vị trí của phần tử có tần số 50:", indices)

    # Thiết kế bộ lọc thông thấp
    cutoff_frequency = indices[0]  # Tần số cắt (cutoff frequency)

    # Áp dụng bộ lọc
    filtered_Gx = Gx_plot[0:cutoff_frequency]
    filtered_Gy = Gy_plot[0:cutoff_frequency]
    filtered_Gz = Gz_plot[0:cutoff_frequency]
    filtered_Ax = Ax_plot[0:cutoff_frequency]
    filtered_Ay = Ay_plot[0:cutoff_frequency]
    filtered_Az = Az_plot[0:cutoff_frequency]

    t_plot2 = np.linspace(0, 50, filtered_Gx.size)

# Cắt bớt hoặc đệm dữ liệu đã lọc theo độ dài mong muốn
    filtered_Gx = truncate_or_pad_data(filtered_Gx, max_data_length)
    filtered_Gy = truncate_or_pad_data(filtered_Gy, max_data_length)
    filtered_Gz = truncate_or_pad_data(filtered_Gz, max_data_length)
    filtered_Ax = truncate_or_pad_data(filtered_Ax, max_data_length)
    filtered_Ay = truncate_or_pad_data(filtered_Ay, max_data_length)
    filtered_Az = truncate_or_pad_data(filtered_Az, max_data_length)

    df = pd.DataFrame({
        'Gx': np.abs(filtered_Gx.astype(complex)),
        'Gy': np.abs(filtered_Gy.astype(complex)),
        'Gz': np.abs(filtered_Gz.astype(complex)),
        'Ax': np.abs(filtered_Ax.astype(complex)),
        'Ay': np.abs(filtered_Ay.astype(complex)),
        'Az': np.abs(filtered_Az.astype(complex))
    })

    # Đường dẫn thư mục tùy ý
    folder_path = './Data_set/Data_test'
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
    data_processing_test (id[i],i,86)
    data_processing_train(id[i],i,86)
