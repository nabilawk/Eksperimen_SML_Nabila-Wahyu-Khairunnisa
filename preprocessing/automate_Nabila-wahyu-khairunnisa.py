import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(path):
    """
    Fungsi untuk memuat dataset
    """
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    """
    Fungsi preprocessing data sesuai eksperimen
    """

    # 1. Replace nilai 0 menjadi NaN
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols] = df[cols].replace(0, np.nan)

    # 2. Handle missing value
    df.fillna(df.mean(), inplace=True)

    # 3. Pisahkan fitur dan target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # 4. Normalisasi data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Gabungkan kembali
    df_clean = pd.DataFrame(X_scaled, columns=X.columns)
    df_clean['Outcome'] = y

    return df_clean


def save_data(df, path):
    """
    Fungsi untuk menyimpan data hasil preprocessing
    """
    df.to_csv(path, index=False)


if __name__ == "__main__":
    import os

    input_path = "namadataset_raw/diabetes.csv"
    output_path = "preprocessing/namadataset_preprocessing/diabetes_clean.csv"

    # buat folder otomatis
    os.makedirs("preprocessing/namadataset_preprocessing", exist_ok=True)

    df = load_data(input_path)
    df_clean = preprocess_data(df)
    save_data(df_clean, output_path)

    print("Preprocessing berhasil disimpan!")