import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv(r'.\Data_set\results\IDGenderAgelist.csv', dtype = str)
id = list(data.ID)
gender = list(data.Gender)
size = len(id)
results = []

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
        dataset_file = f'../Data_set/Data/T0_ID{id}_Walk1.csv' #[] this_i = 0
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
    folder_path = '../Data_set/Data_train'
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
        dataset_file = f'...Data_set/Data2/T0_ID{id}_Walk2.csv' #[] this_i = 0
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
    folder_path = '../Data_set/Data_test'
    if(gender[i]=='0'):
        output_file = os.path.join(folder_path, dataset_id + '_Walk2_0.csv')
    else:
        output_file = os.path.join(folder_path, dataset_id + '_Walk2_1.csv')
    # Xuất DataFrame vào file Excel
    print(output_file)
    try:
        df.to_csv(output_file, index=False)
    except FileExistsError: print('out error'); exit()

def train_test():
    # Đường dẫn đến thư mục chứa dữ liệu
    data_processing_dir = "../Data_set/Data_train/"


    # Xác định các đặc trưng (features) và nhãn (labels)
    features = ["Gx", "Gy", "Gz", "Ax", "Ay", "Az"]

    # Chuẩn bị dữ liệu huấn luyện
    train_data = []
    train_labels = []

    # Đọc dữ liệu từ thư mục Data_processing
    for file_name in os.listdir(data_processing_dir):
        if file_name.endswith("_0.csv"):
            label = 0  # Nhãn nữ
        elif file_name.endswith("_1.csv"):
            label = 1  # Nhãn nam
        else:
            continue

        file_path = os.path.join(data_processing_dir, file_name)
        df = pd.read_csv(file_path)
        flattened_data = df[features].values.flatten()  # Chuyển đổi dữ liệu thành 2 chiều
        train_data.append(flattened_data)
        train_labels.append(label)
    # Xây dựng mô hình SVM
    svm_model = SVC(kernel='rbf')
    svm_model.fit(train_data, train_labels)

    test_path = "../Data_set/Data_test/"


    test_data = []
    test_filenames = []
    test_labels = []

    # Duyệt qua các tệp tin trong thư mục test_path
    for filename in os.listdir(test_path):
        # Kiểm tra nếu tệp tin có đuôi .csv
        if filename.endswith(".csv"):
            file_path = os.path.join(test_path, filename)
            df = pd.read_csv(file_path)
            
            # Lấy dữ liệu từ 6 cột Gx, Gy, Gz, Ax, Ay, Az
            features = df[['Gx', 'Gy', 'Gz', 'Ax', 'Ay', 'Az']].values
            test_data.append(features)  # Thêm dữ liệu vào test_data
            test_filenames.append(filename)  # Thêm tên tệp tin vào test_filenames
            test_labels.append(0 if filename.endswith("_0.csv") else 1)  # Thêm nhãn vào test_labels (0: nữ, 1: nam)

    test_data = np.array(test_data).reshape(len(test_data), -1)  # Chuyển đổi test_data thành mảng numpy 2D

    predictions = svm_model.predict(test_data)  # Dự đoán giới tính trên dữ liệu kiểm tra

    # In kết quả dự đoán và tên tệp tin tương ứng
    for filename, label, prediction in zip(test_filenames, test_labels, predictions):
        print("File:", filename)
        print("Dự đoán giới tính:", prediction)
        print("--------------------")

    accuracy = accuracy_score(test_labels, predictions)  # Tính độ chính xác bằng cách so sánh nhãn thực tế và nhãn dự đoán
    print("Độ chính xác của mô hình SVM trên dữ liệu kiểm tra: {:.2f}%".format(accuracy * 100))
    temp = accuracy
    results.append(temp*100)

for i in range(0,91) :
    for j in range(size):
        data_processing_test(id[j],j,i+30)
        data_processing_train(id[j],j,i+30)
    train_test()


df = pd.DataFrame({
    'predictions': np.abs(results)
})
df.to_csv('../results/results_rbf.csv', index=False)