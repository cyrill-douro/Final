import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import data_loader

def build_autoencoder(input_dim, encoding_dim=10):
    """
    Построение автоэнкодера
    """
    # Энкодер
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation='relu')(input_layer)
    encoded = Dense(16, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    
    # Декодер
    decoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder

def run_autoencoder():
    """
    Запуск кластеризации с использованием автоэнкодера
    """
    print("=== AUTOENCODER КЛАСТЕРИЗАЦИЯ ===")
    
    # Загрузка данных
    X, df, feature_names = data_loader.load_and_prepare_data()
    
    # Нормализация данных для автоэнкодера
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Построение и обучение автоэнкодера
    input_dim = X_scaled.shape[1]
    autoencoder, encoder = build_autoencoder(input_dim, encoding_dim=8)
    
    # Обучение автоэнкодера
    history = autoencoder.fit(X_scaled, X_scaled,
                           epochs=100,
                           batch_size=32,
                           shuffle=True,
                           validation_split=0.2,
                           verbose=0)
    
    # График ошибки обучения
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Ошибка обучения')
    plt.plot(history.history['val_loss'], label='Ошибка валидации')
    plt.title('Ошибка автоэнкодера')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Получение скрытых представлений
    encoded_data = encoder.predict(X_scaled)
    
    # Кластеризация в скрытом пространстве
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(encoded_data)
    
    # Метрики качества
    silhouette_avg = silhouette_score(encoded_data, labels)
    calinski_harabasz = calinski_harabasz_score(encoded_data, labels)
    davies_bouldin = davies_bouldin_score(encoded_data, labels)
    
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz:.3f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
    
    # Визуализация скрытого пространства
    pca = PCA(n_components=2)
    encoded_pca = pca.fit_transform(encoded_data)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(encoded_pca[:, 0], encoded_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel('Первая главная компонента скрытого пространства')
    plt.ylabel('Вторая главная компонента скрытого пространства')
    plt.title('Кластеризация в скрытом пространстве автоэнкодера')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Silhouette plot для скрытого пространства
    data_loader.plot_silhouette(encoded_data, labels)
    
    # Анализ реконструкции
    reconstructed = autoencoder.predict(X_scaled)
    reconstruction_error = np.mean(np.square(X_scaled - reconstructed), axis=1)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(reconstruction_error)), reconstruction_error, c=labels, cmap='viridis')
    plt.colorbar(label='Кластер')
    plt.xlabel('Образец')
    plt.ylabel('Ошибка реконструкции')
    plt.title('Ошибка реконструкции по кластерам')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return labels, autoencoder, encoder