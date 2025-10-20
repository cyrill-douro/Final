import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import kmeans_clustering
import agglomerative_clustering
import autoencoder_clustering
import affinity_propagation
import mean_shift_clustering
import ensemble_clustering

def compare_clustering_methods():
    """
    Сравнение эффективности всех методов кластеризации
    """
    print("=== СРАВНЕНИЕ МЕТОДОВ КЛАСТЕРИЗАЦИИ ===")
    
    # Запуск всех методов
    methods = {
        'K-Means': kmeans_clustering.run_k_means,
        'Agglomerative': agglomerative_clustering.run_agglomerative,
        'Autoencoder': autoencoder_clustering.run_autoencoder,
        'Affinity Propagation': affinity_propagation.run_affinity_propagation,
        'Mean Shift': mean_shift_clustering.run_mean_shift,
        'Ensemble': ensemble_clustering.run_ensemble_clustering
    }
    
    results = []    
    for method_name, method_func in methods.items():
        print(f"\n___{method_name}___")
        try:
            if method_name == 'Ensemble':
                labels, base_clusters, consensus_matrix = method_func()
            elif method_name == 'Autoencoder':
                labels, autoencoder, encoder = method_func()
            else:
                labels, model = method_func()
            
            # Вычисление метрик
            X, df, feature_names = kmeans_clustering.data_loader.load_and_prepare_data()
            
            silhouette = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)
            davies = davies_bouldin_score(X, labels)
            n_clusters = len(np.unique(labels))
            
            results.append({
                'Method': method_name,
                'Silhouette': silhouette,
                'Calinski-Harabasz': calinski,
                'Davies-Bouldin': davies,
                'N_Clusters': n_clusters
            })
            
            
            
        except Exception as e:
            print(f"Ошибка в методе {method_name}: {e}")
            continue
    
    # Создание DataFrame с результатами
    results_df = pd.DataFrame(results)
    print("\n" + "="*50)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*50)
    print(results_df.round(3))
    
    # Визуализация сравнения метрик
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Цвета для всех методов
    colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
    
    # Silhouette Score
    axes[0, 0].bar(results_df['Method'], results_df['Silhouette'], color=colors)
    axes[0, 0].set_title('Silhouette Score (чем выше, тем лучше)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].set_ylabel('Silhouette Score')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Calinski-Harabasz Score
    axes[0, 1].bar(results_df['Method'], results_df['Calinski-Harabasz'], color=colors)
    axes[0, 1].set_title('Calinski-Harabasz Score (чем выше, тем лучше)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].set_ylabel('Calinski-Harabasz Score')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Davies-Bouldin Score
    axes[1, 0].bar(results_df['Method'], results_df['Davies-Bouldin'], color=colors)
    axes[1, 0].set_title('Davies-Bouldin Score (чем ниже, тем лучше)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].set_ylabel('Davies-Bouldin Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Количество кластеров
    axes[1, 1].bar(results_df['Method'], results_df['N_Clusters'], color=colors)
    axes[1, 1].set_title('Количество кластеров')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].set_ylabel('Количество кластеров')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Радарная диаграмма для сравнения методов
    if len(results_df) > 1:  # Только если есть хотя бы 2 метода для сравнения
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, polar=True)
        
        # Нормализация метрик для радарной диаграммы
        normalized_df = results_df.copy()
        for metric in ['Silhouette', 'Calinski-Harabasz']:
            min_val = results_df[metric].min()
            max_val = results_df[metric].max()
            if max_val > min_val:
                normalized_df[metric] = (results_df[metric] - min_val) / (max_val - min_val)
            else:
                normalized_df[metric] = 0.5
        
        # Для Davies-Bouldin инвертируем (чем меньше, тем лучше)
        min_db = results_df['Davies-Bouldin'].min()
        max_db = results_df['Davies-Bouldin'].max()
        if max_db > min_db:
            normalized_df['Davies-Bouldin'] = 1 - ((results_df['Davies-Bouldin'] - min_db) / (max_db - min_db))
        else:
            normalized_df['Davies-Bouldin'] = 0.5
        
        metrics_radar = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']
        angles = np.linspace(0, 2*np.pi, len(metrics_radar), endpoint=False).tolist()
        angles += angles[:1]  # Замыкаем круг
        
        for idx, method in enumerate(results_df['Method']):
            values = normalized_df[normalized_df['Method'] == method][metrics_radar].values[0].tolist()
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_radar)
        ax.set_ylim(0, 1)
        plt.title('Сравнение методов кластеризации (нормализованные метрики)')
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.show()
    
    # Анализ лучших методов по каждой метрике
    if not results_df.empty:
        print("\n" + "_"*50)
        print("АНАЛИЗ")
        print("_"*50)
        
        # Лучший по Silhouette
        best_silhouette = results_df.loc[results_df['Silhouette'].idxmax()]
        print(f"Лучший по Silhouette Score: {best_silhouette['Method']} ({best_silhouette['Silhouette']:.3f})")
        
        # Лучший по Calinski-Harabasz
        best_calinski = results_df.loc[results_df['Calinski-Harabasz'].idxmax()]
        print(f"Лучший по Calinski-Harabasz: {best_calinski['Method']} ({best_calinski['Calinski-Harabasz']:.3f})")
        
        # Лучший по Davies-Bouldin (чем меньше, тем лучше)
        best_davies = results_df.loc[results_df['Davies-Bouldin'].idxmin()]
        print(f"Лучший по Davies-Bouldin: {best_davies['Method']} ({best_davies['Davies-Bouldin']:.3f})")
        
        # Общая рекомендация
        print(f"\nРекомендация: {best_silhouette['Method']} показывает наилучшее качество кластеризации")
    
    return results_df

def run_comparison():
    """
    Запуск сравнения всех методов
    """
    return compare_clustering_methods()