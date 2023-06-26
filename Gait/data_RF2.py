import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

results = []

# Đường dẫn đến thư mục chứa dữ liệu
data_train_dir = "./Data_set/Data_train/"


# Xác định các đặc trưng (features) và nhãn (labels)
features = ["Gx", "Gy", "Gz", "Ax", "Ay", "Az"]

# Chuẩn bị dữ liệu huấn luyện
train_data = []
train_labels = []

# Đọc dữ liệu từ thư mục Data_processing
for file_name in os.listdir(data_train_dir):
    if file_name.endswith("_0.csv"):
        label = 0  # Nhãn nữ
    elif file_name.endswith("_1.csv"):
        label = 1  # Nhãn nam
    else:
        continue

    file_path = os.path.join(data_train_dir, file_name)
    df = pd.read_csv(file_path)
    flattened_data = df[features].values.flatten()  # Chuyển đổi dữ liệu thành 2 chiều
    train_data.append(flattened_data)
    train_labels.append(label)

for i in range(1,20) :
    # Xây dựng mô hình RandomForest
    rf_model = RandomForestClassifier(n_estimators=350)
    rf_model.fit(train_data, train_labels)

    test_path = "./Data_set/Data_test/"

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

    predictions = rf_model.predict(test_data)  # Dự đoán giới tính trên dữ liệu kiểm tra

    # In kết quả dự đoán và tên tệp tin tương ứng
    for filename, label, prediction in zip(test_filenames, test_labels, predictions):
        # print("File:", filename)
        # print("Dự đoán giới tính:", prediction)
        # print("--------------------")
        prediction

    accuracy = accuracy_score(test_labels, predictions)  # Tính độ chính xác bằng cách so sánh nhãn thực tế và nhãn dự đoán
    # print("Độ chính xác của mô hình Random Forest trên dữ liệu kiểm tra: {:.2f}%".format(accuracy * 100))
    print(i)
    print("{:.2f}%".format(accuracy * 100))

    temp = accuracy
    results.append(temp*100)

df = pd.DataFrame({
    'predictions': np.abs(results)
})
df.to_csv('./results_rf2.csv', index=False)


    