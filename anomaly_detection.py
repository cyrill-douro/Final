import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import data_loader
import warnings
warnings.filterwarnings('ignore')

def detect_anomalies_isolation_forest(X, contamination=0.1):
    """
    Обнаружение аномалий с помощью Isolation Forest
    """
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomalies = iso_forest.fit_predict(X)
    anomaly_scores = iso_forest.decision_function(X)
    return anomalies, anomaly_scores

def detect_anomalies_lof(X, contamination=0.1):
    """
    Обнаружение аномалий с помощью Local Outlier Factor
    """
    lof = LocalOutlierFactor(contamination=contamination, n_neighbors=20)
    anomalies = lof.fit_predict(X)
    anomaly_scores = lof.negative_outlier_factor_
    return anomalies, anomaly_scores

def detect_anomalies_ocsvm(X, contamination=0.1):
    """
    Обнаружение аномалий с помощью One-Class SVM
    """
    oc_svm = OneClassSVM(nu=contamination, kernel='rbf', gamma='scale')
    anomalies = oc_svm.fit_predict(X)
    anomaly_scores = oc_svm.decision_function(X)
    return anomalies, anomaly_scores

def detect_anomalies_elliptic_envelope(X, contamination=0.1):
    """
    Обнаружение аномалий с помощью Elliptic Envelope
    """
    envelope = EllipticEnvelope(contamination=contamination, random_state=42)
    anomalies = envelope.fit_predict(X)
    anomaly_scores = envelope.decision_function(X)
    return anomalies, anomaly_scores

def run_anomaly_detection(contamination=0.1):
    """
    Запуск обнаружения аномалий несколькими методами
    """
    
    # Загрузка данных
    X, df, feature_names = data_loader.load_and_prepare_data()
    
    # Методы обнаружения аномалий
    methods = {
        'Isolation Forest': detect_anomalies_isolation_forest,
        'Local Outlier Factor': detect_anomalies_lof,
        'One-Class SVM': detect_anomalies_ocsvm,
        'Elliptic Envelope': detect_anomalies_elliptic_envelope
    }
    
    results = {}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, (method_name, method_func) in enumerate(methods.items()):
        print(f"\n___{method_name}___")
        
        try:
            anomalies, scores = method_func(X, contamination)
            
            # Подсчет аномалий
            n_anomalies = np.sum(anomalies == -1)
            n_normal = np.sum(anomalies == 1)
            
            print(f"Обнаружено аномалий: {n_anomalies}")
            print(f"Нормальных образцов: {n_normal}")
            
            # Визуализация с помощью PCA
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            # Создаем маски для нормальных и аномальных точек
            normal_mask = anomalies == 1
            anomaly_mask = anomalies == -1
            
            # Визуализация
            ax = axes[idx]
            scatter_normal = ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], 
                                     c='blue', alpha=0.6, label='Нормальные', s=30)
            scatter_anomaly = ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], 
                                      c='red', alpha=0.8, label='Аномалии', s=50, marker='x')
            
            ax.set_xlabel('Первая главная компонента')
            ax.set_ylabel('Вторая главная компонента')
            ax.set_title(f'{method_name}\nАномалий: {n_anomalies}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Сохраняем результаты
            results[method_name] = {
                'anomalies': anomalies,
                'scores': scores,
                'n_anomalies': n_anomalies,
                'n_normal': n_normal
            }
            
        except Exception as e:
            print(f"Ошибка в методе {method_name}: {e}")
            axes[idx].set_title(f'{method_name} - Ошибка')
            continue
    
    plt.tight_layout()
    plt.show()
    
    # Сравнение методов
    print("\n" + "_"*20)
    print("СРАВНЕНИЕ")
    print("_"*20)
    
    comparison_data = []
    for method_name, result in results.items():
        comparison_data.append({
            'Method': method_name,
            'Anomalies': result['n_anomalies'],
            'Normal': result['n_normal'],
            'Anomaly Rate': result['n_anomalies'] / len(X)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.round(3))
    
    # Визуализация сравнения методов
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.bar(comparison_df['Method'], comparison_df['Anomalies'], color='lightcoral')
    plt.title('Количество обнаруженных аномалий')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(comparison_df['Method'], comparison_df['Anomaly Rate'], color='lightblue')
    plt.title('Доля аномалий')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Анализ консенсуса методов
    
    if len(results) > 1:
        # Создаем матрицу голосов
        vote_matrix = np.column_stack([result['anomalies'] for result in results.values()])
        
        # Подсчитываем голоса (сколько методов считают точку аномалией)
        anomaly_votes = np.sum(vote_matrix == -1, axis=1)
        
        # Консенсус: точка считается аномалией если большинство методов согласны
        consensus_threshold = len(results) // 2 + 1
        consensus_anomalies = (anomaly_votes >= consensus_threshold).astype(int)
        consensus_anomalies[consensus_anomalies == 1] = -1
        consensus_anomalies[consensus_anomalies == 0] = 1
        
        n_consensus_anomalies = np.sum(consensus_anomalies == -1)
        print(f"Консенсусные аномалии (согласны >= {consensus_threshold} методов): {n_consensus_anomalies}")
        
        # Визуализация консенсуса
        plt.figure(figsize=(10, 8))
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Распределение голосов
        plt.subplot(2, 2, 1)
        vote_counts = np.bincount(anomaly_votes, minlength=len(results)+1)
        plt.bar(range(len(vote_counts)), vote_counts, color='skyblue')
        plt.xlabel('Количество методов, считающих точку аномалией')
        plt.ylabel('Количество точек')
        plt.title('Распределение голосов методов')
        plt.grid(True, alpha=0.3)
        
        # Консенсусные аномалии
        plt.subplot(2, 2, 2)
        normal_mask = consensus_anomalies == 1
        anomaly_mask = consensus_anomalies == -1
        
        plt.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], 
                   c='blue', alpha=0.6, label='Нормальные', s=30)
        plt.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], 
                   c='red', alpha=0.8, label='Консенсусные аномалии', s=50, marker='x')
        plt.xlabel('Первая главная компонента')
        plt.ylabel('Вторая главная компонента')
        plt.title(f'Консенсусные аномалии\n({n_consensus_anomalies} точек)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Матрица согласия методов
        plt.subplot(2, 2, 3)
        agreement_matrix = np.zeros((len(results), len(results)))
        method_names = list(results.keys())
        
        for i in range(len(method_names)):
            for j in range(len(method_names)):
                from sklearn.metrics import adjusted_rand_score
                agreement = adjusted_rand_score(results[method_names[i]]['anomalies'], 
                                             results[method_names[j]]['anomalies'])
                agreement_matrix[i, j] = agreement
        
        plt.imshow(agreement_matrix, cmap='RdYlBu', vmin=-1, vmax=1)
        plt.colorbar(label='Adjusted Rand Index')
        plt.xticks(range(len(method_names)), method_names, rotation=45)
        plt.yticks(range(len(method_names)), method_names)
        plt.title('Согласие между методами')
        
        plt.tight_layout()
        plt.show()
        
        # Добавляем консенсус в результаты
        results['Consensus'] = {
            'anomalies': consensus_anomalies,
            'scores': anomaly_votes,
            'n_anomalies': n_consensus_anomalies,
            'n_normal': len(X) - n_consensus_anomalies
        }
    
    # Используем лучший метод для детального анализа
    best_method = max(results.items(), key=lambda x: x[1]['n_anomalies'])[0]
    best_anomalies = results[best_method]['anomalies']
    
    print(f"Детальный анализ с использованием {best_method}:")
    
    # Анализ характеристик аномалий
    df_analysis = df.copy()
    df_analysis['is_anomaly'] = best_anomalies == -1
    
    # Статистика по аномалиям и нормальным точкам
    stats = df_analysis.groupby('is_anomaly').agg({
        'review.point': ['mean', 'std', 'min', 'max'],
        'price': ['mean', 'std', 'min', 'max'],
        'name': 'count'
    }).round(2)
    
    print("\nСтатистика по аномальным и нормальным образцам:")
    print(stats)
    
    # Визуализация распределения ключевых признаков
    key_features = ['review.point', 'price', 'has_peat', 'has_sherry']
    available_features = [f for f in key_features if f in df_analysis.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, feature in enumerate(available_features):
        if i < len(axes):
            # Boxplot для числовых признаков
            if df_analysis[feature].dtype in [np.int64, np.float64]:
                normal_data = df_analysis[df_analysis['is_anomaly'] == False][feature]
                anomaly_data = df_analysis[df_analysis['is_anomaly'] == True][feature]
                
                axes[i].boxplot([normal_data, anomaly_data], 
                              labels=['Нормальные', 'Аномалии'])
                axes[i].set_ylabel(feature)
                axes[i].set_title(f'Распределение {feature}')
                axes[i].grid(True, alpha=0.3)
            
            # Bar plot для бинарных признаков
            else:
                cross_tab = pd.crosstab(df_analysis['is_anomaly'], df_analysis[feature])
                cross_tab.plot(kind='bar', ax=axes[i])
                axes[i].set_title(f'Распределение {feature}')
                axes[i].legend(title=feature)
                axes[i].grid(True, alpha=0.3)
    
    # Скрываем неиспользуемые subplots
    for i in range(len(available_features), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    
    # Вывод топ аномальных скотчей
    print("\n___ТОП АНОМАЛЬНЫХ СКОТЧЕЙ___")
    anomaly_indices = np.where(best_anomalies == -1)[0]
    
    if len(anomaly_indices) > 0:
        top_anomalies = df.iloc[anomaly_indices].nlargest(10, 'price')[['name', 'review.point', 'price']]
        print("Топ аномальных скотчей по цене:")
        print(top_anomalies.to_string(index=False))
    
    return results, df_analysis

def run_anomaly_detection_module():
    """
    Функция для запуска модуля обнаружения аномалий
    """
    return run_anomaly_detection()