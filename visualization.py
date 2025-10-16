import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA


def visualization(classification_results, y_test, label_encoder_y,
                feature_columns, X_train, anomalies_iso,
                anomalies_lof, combined_anomalies, df, X_train_indices):
    """
    Комплексная визуализация всех результатов.

    Parameters:
    classification_results (dict): Результаты классификации
    y_test: Тестовые метки
    label_encoder_y: Кодировщик меток
    feature_columns (list): Список признаков
    X_train: Тренировочные данные
    anomalies_iso: Аномалии Isolation Forest
    anomalies_lof: Аномалии LOF
    combined_anomalies: Комбинированные аномалии
    df (pd.DataFrame): Исходные данные
    X_train_indices: Индексы тренировочных данных
    """
    # Визуализация важности признаков
    _plot_feature_importance(classification_results['models'], feature_columns)
    
    # Дополнительные визуализации
    _plot_additional_visualizations(df, X_train_indices, combined_anomalies, 
                                  feature_columns, classification_results['models'])


def _plot_feature_importance(models_dict, feature_columns):
    """Визуализация важности признаков."""
    plt.figure(figsize=(14, 10))
    feature_importance = models_dict['Random Forest'].feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=True)

    # Берем топ-15 признаков
    top_features = feature_importance_df['feature'][-15:].tolist()
    top_importances = feature_importance_df['importance'][-15:].tolist()
    
    plt.barh(top_features, top_importances,
             color='lightcoral', edgecolor='darkred', alpha=0.8)
    plt.title('Топ-15 важных признаков (Random Forest)', 
             fontsize=16, fontweight='bold')
    plt.xlabel('Важность', fontsize=12)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_additional_visualizations(df, X_train_indices, combined_anomalies, feature_columns, models_dict):
    """Дополнительные визуализации."""
    plt.figure(figsize=(18, 12))

    # Распределение цен
    plt.subplot(2, 2, 1)
    plt.hist(df['price'], bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.title('Распределение цен', fontsize=14, fontweight='bold')
    plt.xlabel('Цена', fontsize=12)
    plt.ylabel('Частота', fontsize=12)
    plt.yscale('log')
    plt.grid(alpha=0.3)

    # Зависимость оценки от цены (только тренировочные данные)
    plt.subplot(2, 2, 2)
    train_df = df.iloc[X_train_indices]
    
    scatter = plt.scatter(train_df['price'], train_df['review.point'], alpha=0.6,
                         c=combined_anomalies, cmap='coolwarm',
                         edgecolors='black', linewidth=0.5, s=60)
    plt.colorbar(scatter, label='Аномалия (1-да, 0-нет)')
    plt.title('Зависимость оценки от цены (Тренировочные данные)', fontsize=14, fontweight='bold')
    plt.xlabel('Цена', fontsize=12)
    plt.ylabel('Оценка', fontsize=12)
    plt.xscale('log')
    plt.grid(alpha=0.3)

    # Корреляционная матрица
    plt.subplot(2, 2, 3)
    feature_importance = models_dict['Random Forest'].feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=True)

    # Берем топ-8 признаков
    top_features = feature_importance_df['feature'][-8:].tolist()
    
    # Проверяем, что все признаки существуют в df
    existing_features = [f for f in top_features if f in df.columns]
    corr_features = existing_features + ['review.point']
    
    corr_matrix = df[corr_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
               annot_kws={"size": 10})
    plt.title('Корреляционная матрица (топ признаки)', fontsize=14, fontweight='bold')

    # Сравнение распределений оценок (только тренировочные данные)
    plt.subplot(2, 2, 4)
    train_df = df.iloc[X_train_indices]
    normal_df = train_df[~combined_anomalies]
    anomaly_df = train_df[combined_anomalies]

    box_plot = plt.boxplot([normal_df['review.point'], anomaly_df['review.point']],
               labels=['Нормальные', 'Аномалии'], patch_artist=True)

    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Сравнение распределений оценок (Тренировочные данные)', fontsize=14, fontweight='bold')
    plt.ylabel('Оценка', fontsize=12)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()