import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA


def anomaly_detection(X_train):

    # Isolation Forest для поиска аномалий
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    anomalies_iso = iso_forest.fit_predict(X_train)

    # Local Outlier Factor для поиска аномалий
    lof = LocalOutlierFactor(contamination=0.1)
    anomalies_lof = lof.fit_predict(X_train)

    # Комбинированный подход
    combined_anomalies = (anomalies_iso == -1) & (anomalies_lof == -1)

    print(f"Аномалии (Isolation Forest): {np.sum(anomalies_iso == -1)}")
    print(f"Аномалии (LOF): {np.sum(anomalies_lof == -1)}")
    print(f"Аномалии (комбинированные): {np.sum(combined_anomalies)}")

    # Визуализация результатов
    _plot_anomaly_results(X_train, anomalies_iso, anomalies_lof, combined_anomalies)
    _plot_anomaly_statistics(anomalies_iso, anomalies_lof, combined_anomalies)

    return anomalies_iso, anomalies_lof, combined_anomalies


def _plot_anomaly_results(X_train, anomalies_iso, anomalies_lof, combined_anomalies):
    """Визуализация результатов обнаружения аномалий."""
    plt.figure(figsize=(18, 12))

    # PCA для визуализации
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)

    # Isolation Forest
    plt.subplot(2, 3, 1)
    normal = anomalies_iso == 1
    anomaly = anomalies_iso == -1
    plt.scatter(X_pca[normal, 0], X_pca[normal, 1], c='blue', alpha=0.6,
                label='Нормальные', s=30)
    plt.scatter(X_pca[anomaly, 0], X_pca[anomaly, 1], c='red', alpha=0.8,
                label='Аномалии', s=50)
    plt.title('Аномалии (Isolation Forest)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    # LOF
    plt.subplot(2, 3, 2)
    normal = anomalies_lof == 1
    anomaly = anomalies_lof == -1
    plt.scatter(X_pca[normal, 0], X_pca[normal, 1], c='blue', alpha=0.6,
                label='Нормальные', s=30)
    plt.scatter(X_pca[anomaly, 0], X_pca[anomaly, 1], c='red', alpha=0.8,
                label='Аномалии', s=50)
    plt.title('Аномалии (Local Outlier Factor)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    # Комбинированные
    plt.subplot(2, 3, 3)
    normal = ~combined_anomalies
    anomaly = combined_anomalies
    plt.scatter(X_pca[normal, 0], X_pca[normal, 1], c='blue', alpha=0.6,
                label='Нормальные', s=30)
    plt.scatter(X_pca[anomaly, 0], X_pca[anomaly, 1], c='red', alpha=0.8,
                label='Аномалии', s=50)
    plt.title('Аномалии (Комбинированные)', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)

    # Распределение аномалий по компонентам PCA
    plt.subplot(2, 3, 4)
    plt.hist(X_pca[normal, 0], bins=30, alpha=0.7, color='blue', label='Нормальные')
    plt.hist(X_pca[anomaly, 0], bins=30, alpha=0.7, color='red', label='Аномалии')
    plt.xlabel('PCA Component 1')
    plt.ylabel('Частота')
    plt.title('Распределение по PCA Component 1')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(2, 3, 5)
    plt.hist(X_pca[normal, 1], bins=30, alpha=0.7, color='blue', label='Нормальные')
    plt.hist(X_pca[anomaly, 1], bins=30, alpha=0.7, color='red', label='Аномалии')
    plt.xlabel('PCA Component 2')
    plt.ylabel('Частота')
    plt.title('Распределение по PCA Component 2')
    plt.legend()
    plt.grid(alpha=0.3)

    
def _plot_anomaly_statistics(anomalies_iso, anomalies_lof, combined_anomalies):
    """Визуализация статистики аномалий."""
    plt.figure(figsize=(12, 8))

    # Сравнение методов
    plt.subplot(2, 2, 1)
    methods = ['Isolation Forest', 'LOF', 'Комбинированные']
    counts = [
        np.sum(anomalies_iso == -1),
        np.sum(anomalies_lof == -1),
        np.sum(combined_anomalies)
    ]
    
    bars = plt.bar(methods, counts, color=['orange', 'green', 'red'], alpha=0.8)
    plt.title('Количество аномалий по методам')
    plt.ylabel('Количество')
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(count), ha='center', va='bottom')

    # Пересечение аномалий
    plt.subplot(2, 2, 2)
    iso_only = np.sum((anomalies_iso == -1) & (anomalies_lof == 1))
    lof_only = np.sum((anomalies_iso == 1) & (anomalies_lof == -1))
    both = np.sum(combined_anomalies)
    
    plt.pie([iso_only, lof_only, both], 
            labels=['Только ISO', 'Только LOF', 'Оба метода'],
            autopct='%1.1f%%', colors=['lightblue', 'lightgreen', 'lightcoral'])
    plt.title('Пересечение обнаруженных аномалий')

    # Соотношение нормальных/аномальных
    plt.subplot(2, 2, 3)
    normal_count = np.sum(~combined_anomalies)
    anomaly_count = np.sum(combined_anomalies)
    
    plt.pie([normal_count, anomaly_count], 
            labels=['Нормальные', 'Аномалии'],
            autopct='%1.1f%%', colors=['lightblue', 'red'])
    plt.title('Соотношение нормальных и аномальных образцов')

    # Сводная статистика
    plt.subplot(2, 2, 4)
    plt.axis('off')
    stats_text = f"""
    Статистика аномалий:
    
    Всего образцов: {len(anomalies_iso)}
    Нормальные: {normal_count} ({normal_count/len(anomalies_iso)*100:.1f}%)
    Аномалии: {anomaly_count} ({anomaly_count/len(anomalies_iso)*100:.1f}%)
    
    Совпадение методов: {both/(iso_only + lof_only + both)*100:.1f}%
    """
    plt.text(0.1, 0.9, stats_text, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    plt.tight_layout()
    plt.show()