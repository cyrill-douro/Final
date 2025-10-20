import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
import data_loader

def run_affinity_propagation():
    """
    Запуск Affinity Propagation кластеризации
    """
    print("=== AFFINITY PROPAGATION КЛАСТЕРИЗАЦИЯ ===")
    
    # Загрузка данных
    X, df, feature_names = data_loader.load_and_prepare_data()
    
    # Применяем PCA для уменьшения размерности (Affinity Propagation чувствителен к размерности)
    pca = PCA(n_components=0.95)  # Сохраняем 95% дисперсии
    X_reduced = pca.fit_transform(X)
    
    print(f"Исходная размерность: {X.shape[1]}")
    print(f"Размерность после PCA: {X_reduced.shape[1]}")
    
    # Affinity Propagation кластеризация
    af = AffinityPropagation(random_state=42)
    labels = af.fit_predict(X_reduced)
    
    n_clusters = len(np.unique(labels))
    
    # Метрики качества
    silhouette_avg = silhouette_score(X_reduced, labels)
    calinski_harabasz = calinski_harabasz_score(X_reduced, labels)
    davies_bouldin = davies_bouldin_score(X_reduced, labels)
    
    print(f"Количество кластеров: {n_clusters}")
    print(f"Silhouette Score: {silhouette_avg:.3f}")
    print(f"Calinski-Harabasz Score: {calinski_harabasz:.3f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.3f}")
    
    # Визуализация
    pca_vis = PCA(n_components=2)
    X_pca_vis = pca_vis.fit_transform(X_reduced)
    
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.xlabel('Первая главная компонента')
    plt.ylabel('Вторая главная компонента')
    plt.title(f'Affinity Propagation кластеризация ({n_clusters} кластеров)')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Silhouette plot
    data_loader.plot_silhouette(X_reduced, labels)
    
    # Анализ экземпляров-прототипов (exemplars)
    if hasattr(af, 'cluster_centers_indices_'):
        n_exemplars = len(af.cluster_centers_indices_)
        print(f"\nКоличество экземпляров-прототипов: {n_exemplars}")
        
        # Визуализация прототипов
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_pca_vis[:, 0], X_pca_vis[:, 1], c=labels, cmap='viridis', alpha=0.3)
        exemplars = X_pca_vis[af.cluster_centers_indices_]
        plt.scatter(exemplars[:, 0], exemplars[:, 1], c='red', marker='X', s=200, label='Прототипы')
        plt.colorbar(scatter)
        plt.xlabel('Первая главная компонента')
        plt.ylabel('Вторая главная компонента')
        plt.title('Прототипы кластеров (Affinity Propagation)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    # Распределение по кластерам
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
    
    return labels, af