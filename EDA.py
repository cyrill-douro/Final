import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List


def whisky_eda(df):
    print("\n=== КАТЕГОРИИ ВИСКИ ===")
    cats = df['category'].value_counts()
    for cat, count in cats.items():
        pct = (count / len(df)) * 100
        print(f"{cat}: {count} ({pct:.1f}%)")
            
    plt.figure(figsize=(10, 5))
    cats.plot(kind='bar', color='skyblue')
    plt.title('Распределение по категориям')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
        
    print("\n=== АНАЛИЗ ОЦЕНОК ===")
    scores = df['review.point']
    print(f"Среднее: {scores.mean():.2f}")
    print(f"Медиана: {scores.median():.2f}")
    print(f"Стандартное отклонение: {scores.std():.2f}")
    print(f"Диапазон: {scores.min()} - {scores.max()}")
            
    plt.figure(figsize=(12, 4))
            
    plt.subplot(1, 2, 1)
    plt.hist(scores, bins=20, color='lightgreen', alpha=0.7)
    plt.title('Распределение оценок')
    plt.xlabel('Оценка')
            
    plt.subplot(1, 2, 2)
    plt.boxplot(scores, vert=False)
    plt.title('Boxplot оценок')
            
    plt.tight_layout()
    plt.show()

    print("\n=== АНАЛИЗ ЦЕН ===")
    prices = df['price']
    print(f"Средняя цена: ${prices.mean():.2f}")
    print(f"Медианная цена: ${prices.median():.2f}")
    print(f"Самая дорогая: ${prices.max():.2f}")
    print(f"Самая дешевая: ${prices.min():.2f}")

    plt.figure(figsize=(10, 5))
    plt.hist(prices, bins=10, color='salmon', alpha=0.7, edgecolor='black')
    plt.title('Распределение цен')
    plt.xlabel('Цена ($)')

    print("\nТоп-5 самых дорогих:")
    expensive = df.nlargest(5, 'price')[['name', 'price', 'review.point']]
    for _, row in expensive.iterrows():
        name_short = row['name'][:50] + "..." if len(row['name']) > 50 else row['name']
        print(f"  {name_short}")
        print(f"    Цена: ${row['price']:.2f}, Оценка: {row['review.point']}")
            
    print("\n=== ВКУСОВЫЕ ПРОФИЛИ ===")
            
    flavor_cols = [col for col in df.columns if col.startswith('has_')]
    flavor_counts = df[flavor_cols].sum().sort_values(ascending=False)
            
    print("Самые популярные вкусы:")
    for flavor, count in flavor_counts.head(8).items():
        pct = (count / len(df)) * 100
        flavor_name = flavor.replace('has_', '')
        print(f"  {flavor_name}: {count} ({pct:.1f}%)")
            
    plt.figure(figsize=(10, 6))
    flavor_counts.head(10).plot(kind='barh', color='lightcoral')
    plt.title('10 самых популярных вкусовых нот')
    plt.xlabel('Количество виски')
    plt.tight_layout()
    plt.show()
        
    print("\n=== Цена - Рейтинг ===")
            
    correlation = df['price'].corr(df['review.point'])
    print(f"Корреляция цена-оценка: {correlation:.3f}")
            
    # УБИРАЕМ СОЗДАНИЕ value_ratio - это data leakage!
    # Анализ соотношения цена/качество только для EDA, не для моделей
    temp_value_ratio = df['review.point'] / df['price']
    best_value_indices = temp_value_ratio.nlargest(5).index
    best_value = df.loc[best_value_indices, ['name', 'price', 'review.point']]
            
    print("\nТоп-5 по соотношению цена/качество (только для анализа):")
    for _, row in best_value.iterrows():
        name_short = row['name'][:40] + "..." if len(row['name']) > 40 else row['name']
        print(f"  {name_short}")
        print(f"    Цена: ${row['price']:.2f}, Оценка: {row['review.point']}")
            
    plt.figure(figsize=(10, 6))
    plt.scatter(df['price'], df['review.point'], alpha=0.6)
    plt.xlabel('Цена ($)')
    plt.ylabel('Оценка')
    plt.title('Связь между ценой и оценкой')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Возвращаем исходный df без новых столбцов
    return df