import pandas as pd
import numpy as np
from datetime import datetime
from imblearn.combine import SMOTETomek

test_date = datetime(2015, 3, 10)

x_param = [
    'MinTemp', 'MaxTemp', 'Evaporation', 'Sunshine', 
    'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 
    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 
    'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 
    'Temp9am', 'Temp3pm'
]

data = pd.read_csv('E:/Seminar/RandomForest/Data/Weather_Data_Clean.csv')

#Cân bằng dữ liệu huấn luyện
def resampling(x_train, y_train):
    smote_tomek = SMOTETomek(random_state=0)
    x_train, y_train = smote_tomek.fit_resample(x_train, y_train)

    return x_train, y_train

# Lấy dữ liệu cho việc điều chỉnh mô hình phân loại
def tuning_class_data():

    train_data = data[data['Date'] < test_date]

    x_train = train_data[x_param]
    y_train = train_data['rain_group']

    x_train, y_train = resampling(x_train, y_train)

    return x_train, y_train

# Lấy dữ liệu cho việc điều chỉnh mô hình hồi quy
def tuning_reg_data(strong:bool):
    if strong:
        train_data = data[(data['Date'] < test_date) &
                          (data['rain_group'] == 2)]
    elif not strong:
        train_data = data[(data['Date'] < test_date) &
                          (data['rain_group'] == 1)]

    x_train = train_data[x_param]
    y_train = train_data['Rainfall']


    return x_train, y_train

# Lấy dữ liệu huấn luyện và kiểm tra
def get_est_data():
    train_data = data[data['Date'] < test_date]
    test_data = data[data['Date'] >= test_date]

    return train_data, test_data



