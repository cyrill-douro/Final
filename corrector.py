import pandas as pd
import numpy as np
import re

def word_changer(df, selected_words):
  # Функция для проверки наличия слова в тексте
  def contains_word(text, word):
    if pd.isna(text):
        return 0
    # Поиск слова как отдельного слова (с границами слова)
    pattern = r'\b' + re.escape(word.lower()) + r'\b'
    return 1 if re.search(pattern, text.lower()) else 0

  # Создаем столбцы для каждого выбранного слова
  for word in selected_words:
    column_name = f'has_{word}'
    df[column_name] = df['description'].apply(lambda x: contains_word(x, word))

  # Функция для подсчета количества выбранных слов в каждом описании
  def count_selected_words(text):
    if pd.isna(text):
        return 0
    text_lower = text.lower()
    count = 0
    for word in selected_words:
        pattern = r'\b' + re.escape(word.lower()) + r'\b'
        if re.search(pattern, text_lower):
            count += 1
    return count


  # Статистика по найденным словам
  print("Статистика по найденным словам:")
  for word in selected_words:
    count = df[f'has_{word}'].sum()
    percentage = (count / len(df)) * 100
    print(f"{word}: {count} записей ({percentage:.1f}%)")

  print(f"\nОбщее количество записей: {len(df)}")

  # Показываем первые несколько строк с новыми столбцами
  print("\nПервые 5 строк с новыми столбцами:")
  columns_to_show = ['name'] + [f'has_{word}' for word in selected_words[:5]]
  print(df[columns_to_show].head())


def find_non_numeric_values(df, column_name):
    """
    Поиск нечисловых значений в указанном столбце.
    """
    print(f"=== АНАЛИЗ СТОЛБЦА '{column_name}' ===")
    
    if column_name not in df.columns:
        print(f"Столбец '{column_name}' не найден!")
        return None
    
    # Проверяем тип данных столбца
    print(f"Тип данных столбца: {df[column_name].dtype}")
    
    # Поиск нечисловых значений (если столбец должен быть числовым)
    if df[column_name].dtype == 'object':
        # Ищем значения, которые нельзя преобразовать в число
        non_numeric_values = []
        for value in df[column_name]:
            try:
                float(value)  # Пытаемся преобразовать в число
            except (ValueError, TypeError):
              if pd.notna(value):  # Исключаем NaN значения
                    non_numeric_values.append(value)
        
        # Уникальные нечисловые значения
        unique_non_numeric = set(non_numeric_values)
        print(f"Найдено {len(non_numeric_values)} нечисловых значений")
        print(f"Уникальные нечисловые значения: {unique_non_numeric}")
        
        return non_numeric_values
    

def interactive_price_corrector_direct(df, price_column):
    """
    Интерактивная система исправления цен. Изменяет исходный DataFrame напрямую.
    """
    print("Ручное исправление цен")
    print("=" * 55)
    
    # Сохраняем оригинальные значения для статистики
    original_dtypes = df[price_column].dtype
    original_non_numeric = 0
    
    # Функция для проверки, можно ли преобразовать в число
    def is_convertible_to_float(value):
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False
    
    # Находим все уникальные проблемные значения
    unique_prices = df[price_column].astype(str).unique()
    problematic_values = []
    
    for price in unique_prices:
        price_str = str(price).strip()
        # Пропускаем пустые и уже числовые значения
        if not price_str or price_str.lower() in ['nan', 'none', 'null']:
            continue
        if is_convertible_to_float(price_str):
            continue
        problematic_values.append(price_str)
        original_non_numeric += (df[price_column].astype(str) == price_str).sum()
    
    print(f"Найдено {len(problematic_values)} проблемных форматов цен")
    print(f"Всего {original_non_numeric} нечисловых значений")
    
    # Обрабатываем каждое проблемное значение
    changed_count = 0
    skipped_count = 0
    
    for i, original_value in enumerate(problematic_values, 1):
        print(f"\nПроблемное значение {i}/{len(problematic_values)}: '{original_value}'")
        print(f"Встречается {len(df[df[price_column].astype(str) == original_value])} раз")
        
        # Показываем все столбцы для контекста
        examples = df[df[price_column].astype(str) == original_value].head(2)
        if not examples.empty:
            display(examples)
        
        # ПРЕДЛАГАЕМ ВВЕСТИ ЦЕНУ
        
        user_input = input("Введите правильное числовое значение (или Enter чтобы пропустить): ").strip()
        
        if user_input and is_convertible_to_float(user_input):
            # Ввели корректное число - заменяем напрямую в DataFrame
            new_value = float(user_input)
            
            # Находим индексы строк с этим значением
            mask = df[price_column].astype(str) == original_value
            indices_to_change = df[mask].index
            
            # Заменяем значения напрямую
            df.loc[mask, price_column] = new_value
            changed_count += len(indices_to_change)
            
            print(f"Заменено {len(indices_to_change)} значений: '{original_value}' → {new_value}")
        else:
            # Пропускаем
            skipped_count += len(df[df[price_column].astype(str) == original_value])
            print("Пропущено")
    
    # Преобразуем весь столбец в числовой формат
    print("\nПреобразуем столбец в числовой формат...")
    df[price_column] = pd.to_numeric(df[price_column], errors='coerce')
    
    # Статистика
    final_numeric = df[price_column].notna().sum()
    total = len(df)
    
    print(f"\nРЕЗУЛЬТАТЫ:")
    print(f"Исходно нечисловых значений: {original_non_numeric}")
    print(f"Изменено значений: {changed_count}")
    print(f"Пропущено значений: {skipped_count}")
    print(f"Успешно преобразовано: {final_numeric}/{total}")
    print(f"Тип данных столбца теперь: {df[price_column].dtype}")
    
    return df, changed_count

def quick_direct_correction(df, price_column):
    """
    Быстрая коррекция цен с прямым изменением DataFrame.
    """
    print(f"Быстрое исправление столбца '{price_column}'")
    
    # Сразу преобразуем в строку для сравнения
    original_values = df[price_column].copy()
    
    changed_count = 0
    problematic_found = False
    
    # Проходим по всем строкам
    for idx, value in enumerate(original_values):
        try:
            # Пытаемся преобразовать в число
            float(value)
        except (ValueError, TypeError):
            # Не получилось - показываем контекст
            if pd.notna(value):
                problematic_found = True
                print(f"\nПроблемное значение в строке {idx}: '{value}'")
                print("Контекст:")
                display(df.iloc[[idx]])  # Показываем всю строку
                
                # Предлагаем ввести цену
                user_input = input("Введите правильное числовое значение (или Enter чтобы пропустить): ").strip()
                
                if user_input and user_input.replace('.', '').isdigit():
                    # Заменяем напрямую
                    df.at[idx, price_column] = float(user_input)
                    changed_count += 1
                    print(f"Исправлено: {user_input}")
                else:
                    print("Пропущено")
    
    if not problematic_found:
        print("Нечисловых значений не найдено!")
    
    # Финальное преобразование
    df[price_column] = pd.to_numeric(df[price_column], errors='coerce')
    print(f"Изменено значений: {changed_count}")
    
    return df, changed_count

def correct_prices_directly(file_path, price_column):
    """
    Загружает данные и сразу исправляет цены напрямую.
    """
    # Загрузка данных
    df = pd.read_csv(file_path)
    print(f"Загружено данных: {len(df)} строк")
    
    # Исправляем цены напрямую
    df, changed = interactive_price_corrector_direct(df, price_column)


    
    # Сохраняем результат (перезаписываем исходный)
    df.to_csv(file_path, index=False)
    print(f"Данные сохранены обратно в {file_path}")
    
    return df, changed