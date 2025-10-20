import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import data_loader

def run_mean_shift():
    """
    Запуск Mean Shift кластеризации
    """
    print("=== MEAN SHIFT КЛАСТЕРИЗАЦИЯ ===")
    
    # Загрузка данных
    X, df, feature_names = data_loader.load_and_prepare_data()
    
    # Оценка bandwidth с разными параметрами
    try:
        bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=500)  # Увеличиваем quantile
        print(f"Оцененная bandwidth: {bandwidth:.3f}")
        
        # Если bandwidth слишком маленькая, устанавливаем минимальное значение
        if bandwidth < 1.0:
            bandwidth = 1.5
            print(f"Используем bandwidth: {bandwidth:.3f}")
            
    except:
        bandwidth = 2.0
        print(f"Используем фиксированную bandwidth: {bandwidth:.3f}")
    
    # Mean Shift кластеризация с bin_seeding=False
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=False, cluster_all=True)
    labels = ms.fit_predict(X)
    
    n_clusters = len(np.unique(labels))
    
    # Проверяем, что есть хотя бы 2 кластера
    if n_clusters < 2:
        print("Обнаружено менее 2 кластеров. Пробуем другую bandwidth...")
        # Пробуем разные значения bandwidth
        for bw in [2.0, 2.5, 3.0, 1.5]:
            try:
                ms = MeanShift(bandwidth=bw, bin_seeding=False, cluster_all=True)
                labels = ms.fit_predict(X)
                n_clusters = len(np.unique(labels))
                if n_clusters >= 2:
                    print(f"Найдено {n_clusters} кластеров с bandwidth={bw}")
                    break
            except:
                continue
    
    # Метрики качества (только если есть хотя бы 2 кластера)
    if n_clusters >= 2:
        silhouette_avg = silhouette_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
    else:
        silhouette_avg = -1
        calinski_harabasz = 0
        davies_bouldin = float('inf')
        print("Внимание: обнаружен только 1 кластер. Метрики не могут быть вычислены.")
    
    print(f"Количество кластеров: {n_clusters}")
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
    plt.title(f'Mean Shift кластеризация ({n_clusters} кластеров)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Silhouette plot (только если есть хотя бы 2 кластера)
    if n_clusters >= 2:
        data_loader.plot_silhouette(X, labels)
    
    # Визуализация центров кластеров (если они есть)
    if hasattr(ms, 'cluster_centers_') and len(ms.cluster_centers_) > 0:
        cluster_centers = ms.cluster_centers_
        cluster_centers_pca = pca.transform(cluster_centers)
        
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
        plt.scatter(cluster_centers_pca[:, 0], cluster_centers_pca[:, 1], 
                   c='red', marker='X', s=200, label='Центры кластеров')
        plt.colorbar(scatter)
        plt.xlabel('Первая главная компонента')
        plt.ylabel('Вторая главная компонента')
        plt.title('Центры кластеров Mean Shift')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Анализ размера кластеров
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(10, 6))
    plt.bar(unique, counts)
    plt.xlabel('Кластер')
    plt.ylabel('Количество образцов')
    plt.title('Распределение образцов по кластерам')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\nРаспределение по кластерам:")
    for cluster, count in zip(unique, counts):
        print(f"Кластер {cluster}: {count} образцов")
    
    return labels, ms