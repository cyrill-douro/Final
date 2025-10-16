import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb


def classification(X_train, X_test, y_train, y_test):
    # Инициализация моделей
    models = {
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'LDA': LinearDiscriminantAnalysis(),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }

    # Оценка отдельных моделей
    results = {}
    predictions = {}
    detailed_metrics = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Основные метрики
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # ROC-AUC если доступно
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        results[name] = accuracy
        predictions[name] = y_pred
        
        # Детальные метрики
        detailed_metrics[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        print(f"_____{name}:_____")
        print(f"  Accuracy = {accuracy:.4f}")
        print(f"  Precision = {precision:.4f}")
        print(f"  Recall = {recall:.4f}")
        print(f"  F1-Score = {f1:.4f}")
        if roc_auc is not None:
            print(f"  ROC-AUC = {roc_auc:.4f}")
        print()

     # Создание ансамбля
    ensemble = VotingClassifier(
        estimators=[
            ('et', models['Extra Trees']),
            ('rf', models['Random Forest']),
            ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('lda', LinearDiscriminantAnalysis())
        ],
        voting='soft'
    )

    # Обучение ансамбля
    ensemble.fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_test)
    y_pred_ensemble_proba = ensemble.predict_proba(X_test)[:, 1]
    
    # Метрики ансамбля
    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    ensemble_precision = precision_score(y_test, y_pred_ensemble)
    ensemble_recall = recall_score(y_test, y_pred_ensemble)
    ensemble_f1 = f1_score(y_test, y_pred_ensemble)
    ensemble_roc_auc = roc_auc_score(y_test, y_pred_ensemble_proba)
    
    results['Ensemble'] = ensemble_accuracy
    detailed_metrics['Ensemble'] = {
        'accuracy': ensemble_accuracy,
        'precision': ensemble_precision,
        'recall': ensemble_recall,
        'f1_score': ensemble_f1,
        'roc_auc': ensemble_roc_auc
    }

    print("_____АНСАМБЛЬ:_____")
    print(f"  Accuracy = {ensemble_accuracy:.4f}")
    print(f"  Precision = {ensemble_precision:.4f}")
    print(f"  Recall = {ensemble_recall:.4f}")
    print(f"  F1-Score = {ensemble_f1:.4f}")
    print(f"  ROC-AUC = {ensemble_roc_auc:.4f}")# Создание ансамбля
    ensemble = VotingClassifier(
        estimators=[
            ('et', models['Extra Trees']),
            ('rf', models['Random Forest']),
            ('xgb', xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')),
            ('lr', LogisticRegression(random_state=42, max_iter=1000)),
            ('lda', LinearDiscriminantAnalysis())
        ],
        voting='soft'
    )

    # Обучение ансамбля
    ensemble.fit(X_train, y_train)
    y_pred_ensemble = ensemble.predict(X_test)
    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    results['Ensemble'] = ensemble_accuracy

    # Визуализация результатов классификации
    _plot_classification_results(results, y_test, y_pred_ensemble, predictions, ensemble, X_test)
    _plot_model_comparison(results)
    _plot_detailed_metrics(detailed_metrics)
    _plot_metrics_radar(detailed_metrics)

    return {
        'models': models,
        'results': results,
        'ensemble': ensemble,
        'y_pred_ensemble': y_pred_ensemble,
        'predictions': predictions
    }


def _plot_classification_results(results, y_test, y_pred_ensemble, predictions, ensemble, X_test):
    """Визуализация результатов классификации."""
    plt.figure(figsize=(15, 10))
    
    # График точности моделей
    plt.subplot(2, 2, 1)
    models_names = list(results.keys())
    accuracies = list(results.values())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    bars = plt.bar(models_names, accuracies, color=colors, edgecolor='black', alpha=0.8)
    plt.title('Сравнение точности моделей', fontsize=14, fontweight='bold')
    plt.xlabel('Модели', fontsize=12)
    plt.ylabel('Точность', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y', alpha=0.3)
    
    for bar, accuracy in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{accuracy:.3f}', ha='center', va='bottom', fontsize=10)

    # Матрица ошибок ансамбля
    plt.subplot(2, 2, 2)
    cm = confusion_matrix(y_test, y_pred_ensemble)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
    plt.title('Матрица ошибок (Ансамбль)', fontsize=14, fontweight='bold')
    plt.xlabel('Предсказание')
    plt.ylabel('Фактическое')

    # Сравнение предсказаний моделей
    plt.subplot(2, 2, 3)
    comparison_data = []
    for model_name, y_pred in predictions.items():
        correct = np.sum(y_pred == y_test)
        incorrect = len(y_test) - correct
        comparison_data.append([model_name, correct, incorrect])
    
    comparison_df = pd.DataFrame(comparison_data, 
                               columns=['Model', 'Correct', 'Incorrect'])
    
    x = np.arange(len(comparison_df))
    width = 0.35
    
    plt.bar(x - width/2, comparison_df['Correct'], width, label='Правильные', alpha=0.8)
    plt.bar(x + width/2, comparison_df['Incorrect'], width, label='Неправильные', alpha=0.8)
    plt.xlabel('Модели')
    plt.ylabel('Количество предсказаний')
    plt.title('Сравнение правильных/неправильных предсказаний')
    plt.xticks(x, comparison_df['Model'], rotation=45)
    plt.legend()
    plt.grid(alpha=0.3)

    # Распределение уверенности ансамбля
    plt.subplot(2, 2, 4)
    ensemble_proba = ensemble.predict_proba(X_test)
    confidence = np.max(ensemble_proba, axis=1)
    
    plt.hist(confidence, bins=20, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Уверенность предсказания')
    plt.ylabel('Частота')
    plt.title('Распределение уверенности ансамбля')
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def _plot_model_comparison(results):
    """Визуализация сравнения моделей."""
    plt.figure(figsize=(10, 6))
    
    models = list(results.keys())
    scores = list(results.values())
    
    # Сортируем по точности
    sorted_indices = np.argsort(scores)
    models = [models[i] for i in sorted_indices]
    scores = [scores[i] for i in sorted_indices]
    
    bars = plt.barh(models, scores, color='lightgreen', edgecolor='darkgreen', alpha=0.8)
    
    for bar, score in zip(bars, scores):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=10, fontweight='bold')
    
    plt.xlabel('Точность')
    plt.title('Рейтинг моделей по точности')
    plt.xlim(0.5, 1.0)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_detailed_metrics(detailed_metrics):
    """Визуализация детальных метрик для всех моделей."""
    plt.figure(figsize=(15, 10))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_names = ['Точность', 'Precision', 'Recall', 'F1-Score']
    models = list(detailed_metrics.keys())
    
    # График сравнения всех метрик
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        plt.subplot(2, 2, i+1)
        values = [detailed_metrics[model][metric] for model in models]
        
        bars = plt.bar(models, values, alpha=0.7, edgecolor='black')
        plt.title(f'Сравнение {metric_name}', fontsize=12, fontweight='bold')
        plt.ylabel(metric_name)
        plt.xticks(rotation=45)
        plt.ylim(0.5, 1.0)
        plt.grid(axis='y', alpha=0.3)
        
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()


def _plot_metrics_radar(detailed_metrics):
    """Радар-диаграмма для сравнения моделей по всем метрикам."""
    try:
        from math import pi
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_names = ['Точность', 'Precision', 'Recall', 'F1-Score']
        
        fig = plt.figure(figsize=(12, 8))
        
        # Углы для радар-диаграммы
        angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
        angles += angles[:1]  # Замыкаем круг
        
        # Создаем subplot
        ax = fig.add_subplot(111, polar=True)
        
        # Цвета для моделей
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for idx, (model_name, metrics_dict) in enumerate(detailed_metrics.items()):
            values = [metrics_dict[metric] for metric in metrics]
            values += values[:1]  # Замыкаем круг
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx])
            ax.fill(angles, values, alpha=0.1, color=colors[idx])
        
        # Настройки осей
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_names)
        ax.set_ylim(0.5, 1.0)
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.grid(True)
        
        plt.title('Сравнение моделей по всем метрикам (Радар-диаграмма)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("Радар-диаграмма не доступна")


