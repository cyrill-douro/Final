import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def prepare_data(df):
 
    # Создаем бинарную классификацию (высокие/низкие оценки)
    df['rating_class'] = np.where(df['review.point'] >= 94, 'High', 'Low')

    # Кодируем категориальные переменные
    le = LabelEncoder()
    df['category_encoded'] = le.fit_transform(df['category'])

    # Кодируем целевую переменную в числовой формат
    label_encoder_y = LabelEncoder()
    df['rating_class_encoded'] = label_encoder_y.fit_transform(df['rating_class'])

    class_mapping = dict(zip(label_encoder_y.classes_, 
                           label_encoder_y.transform(label_encoder_y.classes_)))
    print(f"Кодирование классов: {class_mapping}")

    # Выделяем признаки и целевую переменную
    exclude_columns = ['name', 'review.point', 'rating_class', 'category', 'rating_class_encoded']
    feature_columns = [col for col in df.columns if col not in exclude_columns]
    
    print(f"Используемые признаки: {len(feature_columns)}")

    
    X = df[feature_columns]
    y = df['rating_class_encoded']

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Сохраняем индексы тренировочных данных
    X_train_indices = X_train.index

    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Размер train: {X_train.shape}")
    print(f"Размер test: {X_test.shape}")

    return (X_train_scaled, X_test_scaled, y_train, y_test, 
            feature_columns, scaler, label_encoder_y, X_train_indices)