import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import data_loader

def run_k_means():
    """
    Запуск K-Means кластеризации с анализом результатов
    """
    print("=== K-MEANS КЛАСТЕРИЗАЦИЯ ===")
    
    # Загрузка данных
    X, df, feature_names = data_loader.load_and_prepare_data()
    
    # Определение оптимального числа кластеров методом локтя
    inertia = []
    k_range = range(2, 11)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    
    # График метода локтя
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, 'bo-')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Inertia')
    plt.title('Метод локтя для определения оптимального k')
    plt.grid(True)
    plt.show()
    
    # Выбираем оптимальное k (например, 4)
    optimal_k = 4
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Метрики качества
    silhouette_avg = silhouette_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    
    print(f"Оптимальное число кластеров: {optimal_k}")
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz:.3f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
    
    # Визуализация с помощью PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel('Первая главная компонента')
    plt.ylabel('Вторая главная компонента')
    plt.title(f'K-Means кластеризация (k={optimal_k}) - PCA проекция')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Silhouette plot
    data_loader.plot_silhouette(X, labels)
    
    # Анализ центроидов кластеров
    print("\nАнализ центроидов кластеров:")
    centroids = kmeans.cluster_centers_
    
    # Визуализация центроидов
    plt.figure(figsize=(15, 8))
    for i in range(optimal_k):
        plt.subplot(2, 2, i+1)
        plt.bar(range(len(feature_names)), centroids[i])
        plt.title(f'Центроид кластера {i}')
        plt.xticks(range(len(feature_names)), feature_names, rotation=90)
        plt.tight_layout()
    plt.show()
    
    # Статистика по кластерам
    df['cluster'] = labels
    cluster_stats = df.groupby('cluster').agg({
        'review.point': ['mean', 'std'],
        'price': ['mean', 'std'],
        'name': 'count'
    }).round(2)
    
    print("\nСтатистика по кластерам:")
    print(cluster_stats)
    
    return labels, kmeans