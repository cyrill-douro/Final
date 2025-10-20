import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
import data_loader

def run_agglomerative():
    """
    Запуск агломеративной кластеризации
    """
    print("=== АГЛОМЕРАТИВНАЯ КЛАСТЕРИЗАЦИЯ ===")
    
    # Загрузка данных
    X, df, feature_names = data_loader.load_and_prepare_data()
    
    # Построение дендрограммы для определения числа кластеров
    plt.figure(figsize=(15, 8))
    Z = linkage(X, method='ward')
    dendrogram(Z, truncate_mode='lastp', p=20)
    plt.title('Дендрограмма агломеративной кластеризации')
    plt.xlabel('Индекс образца')
    plt.ylabel('Расстояние')
    plt.show()
    
    # Выбираем оптимальное число кластеров (например, 4)
    optimal_n_clusters = 4
    agg_clustering = AgglomerativeClustering(n_clusters=optimal_n_clusters, linkage='ward')
    labels = agg_clustering.fit_predict(X)
    
    # Метрики качества
    silhouette_avg = silhouette_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    
    print(f"Число кластеров: {optimal_n_clusters}")
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
    plt.title(f'Агломеративная кластеризация (k={optimal_n_clusters}) - PCA проекция')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Silhouette plot
    data_loader.plot_silhouette(X, labels)
    
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
    
    return labels, agg_clustering