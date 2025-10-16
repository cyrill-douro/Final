import os
import pandas as pd

def load_data(file_path):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден!")
    
    df = pd.read_csv(file_path)
    print(f"Данные успешно загружены. Размер: {df.shape}")
    return df