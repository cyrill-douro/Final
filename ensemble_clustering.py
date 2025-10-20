import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import data_loader

def create_ensemble_clusters(X, n_base_clusters=3, n_final_clusters=4):
    """
    Создание ансамбля кластеризаторов
    """
    base_clusterings = []
    
    # 1. K-Means с разными начальными условиями
    for i in range(n_base_clusters):
        kmeans = KMeans(n_clusters=n_final_clusters, random_state=42+i, n_init=1)
        base_clusterings.append(kmeans.fit_predict(X))
    
    # 2. Agglomerative с разными связями
    linkages = ['ward', 'complete', 'average']
    for linkage in linkages[:min(n_base_clusters, len(linkages))]:
        agg = AgglomerativeClustering(n_clusters=n_final_clusters, linkage=linkage)
        base_clusterings.append(agg.fit_predict(X))
    
    # 3. Создание матрицы согласованности
    n_samples = X.shape[0]
    n_base = len(base_clusterings)
    consensus_matrix = np.zeros((n_samples, n_samples))
    
    for clustering in base_clusterings:
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                if clustering[i] == clustering[j]:
                    consensus_matrix[i, j] += 1
                    consensus_matrix[j, i] += 1
    
    # Нормализация
    consensus_matrix /= n_base
    
    return base_clusterings, consensus_matrix

def cluster_ensemble_voting(base_clusterings, n_final_clusters=4):
    """
    Метод голосования для ансамблевой кластеризации
    """
    n_samples = len(base_clusterings[0])
    n_base = len(base_clusterings)
    
    # Создаем матрицу принадлежности к кластерам
    cluster_matrix = np.column_stack(base_clusterings)
    
    # Применяем K-Means к матрице кластеризаций
    ensemble_kmeans = KMeans(n_clusters=n_final_clusters, random_state=42, n_init=10)
    final_labels = ensemble_kmeans.fit_predict(cluster_matrix)
    
    return final_labels

def cluster_ensemble_consensus(consensus_matrix, n_final_clusters=4):
    """
    Метод консенсуса для ансамблевой кластеризации
    """
    # Используем спектральную кластеризацию на матрице согласованности
    from sklearn.cluster import SpectralClustering
    spectral = SpectralClustering(n_clusters=n_final_clusters, 
                                affinity='precomputed',
                                random_state=42)
    final_labels = spectral.fit_predict(consensus_matrix)
    
    return final_labels

def run_ensemble():
    """
    Запуск ансамблевой кластеризации
    """
    print("=== АНСАМБЛЕВАЯ КЛАСТЕРИЗАЦИЯ ===")
    
    # Загрузка данных
    X, df, feature_names = data_loader.load_and_prepare_data()
    
    # Создание ансамбля базовых кластеризаторов
    
    base_clusterings, consensus_matrix = create_ensemble_clusters(X, n_base_clusters=5, n_final_clusters=4)
    
    # Метод 1: Голосование
    
    labels_voting = cluster_ensemble_voting(base_clusterings, n_final_clusters=4)
    
    # Метод 2: Консенсус
    
    labels_consensus = cluster_ensemble_consensus(consensus_matrix, n_final_clusters=4)
    
    # Сравнение методов ансамбля
    methods = {
        'Голосование': labels_voting,
        'Консенсус': labels_consensus
    }
    
    results = []
    best_method = None
    best_score = -1
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for idx, (method_name, labels) in enumerate(methods.items()):
        # Метрики качества
        silhouette_avg = silhouette_score(X, labels)
        calinski_harabasz = calinski_harabasz_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        n_clusters = len(np.unique(labels))
        
        results.append({
            'Method': method_name,
            'Silhouette': silhouette_avg,
            'Calinski-Harabasz': calinski_harabasz,
            'Davies-Bouldin': davies_bouldin,
            'N_Clusters': n_clusters
        })
        
        # Визуализация с помощью PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # PCA проекция
        scatter = axes[idx, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
        axes[idx, 0].set_xlabel('Первая главная компонента')
        axes[idx, 0].set_ylabel('Вторая главная компонента')
        axes[idx, 0].set_title(f'{method_name} - PCA проекция\nSilhouette: {silhouette_avg:.3f}')
        plt.colorbar(scatter, ax=axes[idx, 0])
        
        # Распределение по кластерам
        unique, counts = np.unique(labels, return_counts=True)
        axes[idx, 1].bar(unique, counts, color=plt.cm.viridis(np.linspace(0, 1, len(unique))))
        axes[idx, 1].set_xlabel('Кластер')
        axes[idx, 1].set_ylabel('Количество образцов')
        axes[idx, 1].set_title(f'{method_name} - Распределение')
        axes[idx, 1].grid(True, alpha=0.3)
        
        # Определяем лучший метод
        if silhouette_avg > best_score:
            best_score = silhouette_avg
            best_method = method_name
            best_labels = labels
    
    plt.tight_layout()
    plt.show()
    
    # Вывод результатов
    print("\nРезультаты ансамблевых методов:")
    for result in results:
        print(f"{result['Method']}:")
        print(f"  Silhouette: {result['Silhouette']:.3f}")
        print(f"  Calinski-Harabasz: {result['Calinski-Harabasz']:.3f}")
        print(f"  Davies-Bouldin: {result['Davies-Bouldin']:.3f}")
        print(f"  Кластеров: {result['N_Clusters']}")
    
    print(f"\nЛучший метод: {best_method} (Silhouette: {best_score:.3f})")
    
    # Детальный анализ лучшего метода
    print(f"\nДетальный анализ метода '{best_method}':")
    
    # Silhouette plot для лучшего метода
    data_loader.plot_silhouette(X, best_labels)
    
    # Анализ согласованности базовых кластеризаторов
    print("\nАнализ согласованности базовых кластеризаторов:")
    from sklearn.metrics import adjusted_rand_score
    
    agreement_matrix = np.zeros((len(base_clusterings), len(base_clusterings)))
    for i in range(len(base_clusterings)):
        for j in range(len(base_clusterings)):
            agreement_matrix[i, j] = adjusted_rand_score(base_clusterings[i], base_clusterings[j])
    
    plt.figure(figsize=(10, 8))
    plt.imshow(agreement_matrix, cmap='YlOrRd', interpolation='nearest')
    plt.colorbar(label='Adjusted Rand Index')
    plt.title('Матрица согласованности базовых кластеризаторов')
    plt.xlabel('Базовый кластеризатор')
    plt.ylabel('Базовый кластеризатор')
    plt.show()
    
    # Обучаем случайный лес для предсказания кластеров
    X_train, X_test, y_train, y_test = train_test_split(
        X, best_labels, test_size=0.3, random_state=42, stratify=best_labels
    )
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Важность признаков
    feature_importance = rf.feature_importances_
    
    plt.figure(figsize=(12, 8))
    indices = np.argsort(feature_importance)[::-1]
    
    plt.bar(range(len(feature_names)), feature_importance[indices])
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
    plt.title('Важность признаков для разделения на кластеры')
    plt.tight_layout()
    plt.show()
    
    # Анализ характеристик кластеров - ИСПРАВЛЕННАЯ ЧАСТЬ
    df_copy = df.copy()
    df_copy['ensemble_cluster'] = best_labels
    
    # Используем только числовые колонки для агрегации
    numeric_columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
    
    # Убедимся, что нужные колонки есть
    analysis_columns = ['review.point', 'price']
    available_columns = [col for col in analysis_columns if col in numeric_columns]
    
    # Добавляем бинарные признаки
    binary_features = [col for col in feature_names if col in df_copy.columns]
    available_columns.extend(binary_features)
    
    cluster_analysis = df_copy.groupby('ensemble_cluster')[available_columns].agg({
        'review.point': ['mean', 'std', 'count'],
        'price': ['mean', 'std']
    }).round(3)
    
    # Для бинарных признаков добавляем среднее
    for feature in binary_features:
        if feature not in ['review.point', 'price']:
            cluster_analysis[(feature, 'mean')] = df_copy.groupby('ensemble_cluster')[feature].mean().round(3)
    
    print("\nСтатистика по кластерам:")
    print(cluster_analysis)
    
    # Визуализация характеристик кластеров
    cluster_means = df_copy.groupby('ensemble_cluster').mean(numeric_only=True)
    
    # Выбираем ключевые характеристики для визуализации
    key_features = ['review.point', 'price', 'has_peat', 'has_sherry', 'has_smoke', 'has_fruit']
    available_key_features = [f for f in key_features if f in cluster_means.columns]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(available_key_features):
        if i < len(axes):
            axes[i].bar(cluster_means.index, cluster_means[feature], 
                      color=plt.cm.viridis(np.linspace(0, 1, len(cluster_means))))
            axes[i].set_xlabel('Кластер')
            axes[i].set_ylabel(feature)
            axes[i].set_title(f'Среднее значение {feature} по кластерам')
            axes[i].grid(True, alpha=0.3)
    
    # Скрываем неиспользуемые subplots
    for i in range(len(available_key_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    return best_labels, base_clusterings, consensus_matrix

def run_ensemble_clustering():
    """
    Функция для запуска ансамблевой кластеризации
    """
    return run_ensemble()