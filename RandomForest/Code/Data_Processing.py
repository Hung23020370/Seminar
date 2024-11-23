import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('E:/Seminar/RandomForest/Data/Weather_Data.csv')
df = df.drop(columns=['RainToday', 'RainTomorrow'])
df.dropna(inplace=True)

conditions = [
    (df['Rainfall'] == 0),                # Không mưa
    (df['Rainfall'] > 0) & (df['Rainfall'] <= 10),  # Mưa nhỏ
    (df['Rainfall'] > 10)                 # Mưa lớn
]

labels = [0,1,2]
df['rain_group'] = np.select(conditions, labels)

# Mã hóa nhãn cho cột danh mục
label_encoder = LabelEncoder()
df['WindGustDir'] = label_encoder.fit_transform(df['WindGustDir'])
df['WindDir9am'] = label_encoder.fit_transform(df['WindDir9am'])
df['WindDir3pm'] = label_encoder.fit_transform(df['WindDir3pm'])

df.to_csv('E:/Seminar/RandomForest/Data/Weather_Data_Clean.csv', index=False)







