import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# === Настройки ===
file_path = 'train.csv'  # путь к исходному файлу
output_dir = r'B:\\RFKT\\result'
output_file_name = 'train_normalized.csv'
output_path = os.path.join(output_dir, output_file_name)

# Метод нормализации: "minmax" или "zscore"
scaling_method = "minmax"

try:
    # === Загрузка данных ===
    df = pd.read_csv(file_path)
    print("Данные 'Space Titanic' успешно загружены!")
    print("-" * 40)

    # === Проверка пропусков ===
    print("Количество пропущенных значений (до заполнения):")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    print("-" * 40)

    # === Заполнение пропусков ===
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns

    # Числовые — средним
    for col in numeric_cols:
        mean_value = df[col].mean()
        df[col].fillna(mean_value, inplace=True)
        print(f"-> Числовая колонка '{col}' заполнена средним: {mean_value:.2f}")

    # Категориальные — модой, безопасно
    for col in categorical_cols:
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)
        df[col] = df[col].infer_objects(copy=False)
        print(f"-> Категориальная колонка '{col}' заполнена модой: {mode_value}")

    print("-" * 40)
    print("Количество пропущенных значений (после заполнения):")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print("-" * 40)



    # === Нормализация числовых данных ===
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    print(f"Числовые колонки для нормализации: {numeric_cols}")

    if scaling_method == "minmax":
        scaler = MinMaxScaler()
        print("Применяется Min-Max нормализация...")
    else:
        scaler = StandardScaler()
        print("Применяется Z-score нормализация...")

    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print("Нормализация завершена.")
    print("-" * 40)

    # === Преобразование только колонки HomePlanet ===
    if 'HomePlanet' in df.columns:
        print("Преобразуем колонку 'HomePlanet'")
        df = pd.get_dummies(df, columns=['HomePlanet'], drop_first=True)
        print("Колонка 'HomePlanet' успешно преобразована.")
    else:
        print("Колонка 'HomePlanet' не найдена, пропускаем преобразование.")
    print("-" * 40)


    # === Удаляем ненужные колонки ===
    for col in ['ShoppingMall', 'FoodCourt', 'Transported']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
            print(f"Колонка '{col}' удалена из итоговой таблицы.")
    print("-" * 40)


    # === Сохранение ===
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Файл успешно сохранён: {output_path}")

except FileNotFoundError:
    print(f"Ошибка: файл '{file_path}' не найден!")
except Exception as e:
    print("Произошла ошибка:", e)
