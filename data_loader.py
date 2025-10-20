import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data():
    """
    Загрузка и подготовка данных для кластеризации
    """
    # Загрузка данных
    df = pd.read_csv('scotch_review_corrected.csv')
    
    # Выделяем числовые признаки для кластеризации
    feature_columns = [
        'review.point', 'price', 'has_smoke', 'has_peat', 'has_sherry', 
        'has_vanilla', 'has_fruit', 'has_sweet', 'has_orange', 'has_fruits',
        'has_toffee', 'has_spices', 'has_palate', 'has_spicy', 'has_pepper',
        'has_apple', 'has_ginger', 'has_citrus', 'has_wood', 'has_caramel',
        'has_smoky', 'has_fruity', 'has_spice', 'has_oak', 'has_honey', 'has_chocolate'
    ]
    
    # Проверяем, что все колонки существуют
    available_columns = [col for col in feature_columns if col in df.columns]
    
    X = df[available_columns].copy()
    
    # Заполняем пропущенные значения
    X = X.fillna(X.mean())
    
    # Масштабируем данные
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df, available_columns

def plot_feature_importance(feature_names, importance_scores, title="Важность признаков"):
    """
    Визуализация важности признаков
    """
    plt.figure(figsize=(12, 8))
    indices = np.argsort(importance_scores)[::-1]
    
    plt.bar(range(len(feature_names)), importance_scores[indices])
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_silhouette(X, labels, metric='euclidean'):
    """
    Построение Silhouette plot
    """
    from sklearn.metrics import silhouette_samples, silhouette_score
    from sklearn.metrics.pairwise import pairwise_distances
    
    n_clusters = len(np.unique(labels))
    
    # Вычисляем silhouette score
    silhouette_avg = silhouette_score(X, labels, metric=metric)
    sample_silhouette_values = silhouette_samples(X, labels, metric=metric)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = plt.cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)
        
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.set_xlabel("Silhouette coefficient values")
    ax.set_ylabel("Cluster label")
    ax.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax.set_yticks([])
    ax.set_title(f"Silhouette plot (avg score: {silhouette_avg:.3f})")
    plt.tight_layout()
    plt.show()
    
    return silhouette_avg