import pandas as pd
import os
import numpy as np

data = pd.read_csv(r'D:\Ki2020_2\Project 2\Ff\Data_set\results\IDGenderAgelist.csv', dtype = str)
id = list(data.ID)
gender = list(data.Gender)
size = len(id)


def data_processing (id,i):
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

    df = pd.DataFrame({
    'Gx': filtered_Gx.astype(complex),
    'Gy': filtered_Gy.astype(complex),
    'Gz': filtered_Gz.astype(complex),
    'Ax': filtered_Ax.astype(complex),
    'Ay': filtered_Ay.astype(complex),
    'Az': filtered_Az.astype(complex)
    })

    # Biến đổi số phức thành số thực
    df = df.apply(np.real)
    # Đường dẫn thư mục tùy ý
    folder_path = './Data_set/Data_processing/'
    if(gender[i]=='0'):
        output_file = os.path.join(folder_path, dataset_id + '_Walk1_0.csv')
    else:
        output_file = os.path.join(folder_path, dataset_id + '_Walk1_1.csv')
    # Xuất DataFrame vào file Excel
    print(output_file)
    try:
        df.to_csv(output_file, index=False)
    except FileExistsError: print('out error'); exit()

for i in range(size):
    data_processing (id[i],i)
