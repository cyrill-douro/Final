import pandas as pd
from collections import Counter
import re

def finder(df):
  def find_repeated_words(text):
    if not isinstance(text, str) or pd.isna(text):
     return
    
    # Очистка текста и разделение на слова
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())  # только слова длиной 3+ символа
    
    # Подсчет частоты слов
    word_counts = Counter(words)
    
    # Находим слова, которые встречаются более 1 раза
    repeated_words = [word for word, count in word_counts.items() if count > 1]
    
    return ', '.join(repeated_words) if repeated_words else ""

  # Применяем функцию к каждому описанию
  df['repeated_words'] = df['description.1.2247.'].apply(find_repeated_words)

   # Выводим первые несколько строк для проверки
  print("Первые 5 строк с повторяющимися словами:")
  for i in range(5):
    if i < len(df):
        print(f"ID {df.iloc[i]['id']}: {df.iloc[i]['repeated_words']}")

  # Статистика по повторяющимся словам
  repeated_words_stats = df[df['repeated_words'] != ''].shape[0]
  print(f"\nКоличество записей с повторяющимися словами: {repeated_words_stats} из {len(df)}")
  print(f"Процент записей с повторяющимися словами: {repeated_words_stats/len(df)*100:.1f}%")

  # Показываем наиболее частые повторяющиеся слова во всем наборе данных
  all_repeated = []
  for words_str in df['repeated_words']:
    if words_str:
        all_repeated.extend(words_str.split(', '))

  common_repeats = Counter(all_repeated).most_common(100)
  print(f"\n100 самых частых повторяющихся слов:")
  for word, count in common_repeats:
    print(f"  {word}: {count} раз")