import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np

# Указываем путь к файлу
file_path = 'train.csv'

try:
    df = pd.read_csv(file_path)
    print("✅ Данные 'Space Titanic' успешно загружены!")
    print("-" * 40)

    # ====================================================================
    # ПРЕДВАРИТЕЛЬНЫЙ ШАГ: Определение типов данных для обработки
    # ====================================================================

    # Числовые столбцы для нормализации и заполнения медианой
    numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # Категориальные столбцы для кодирования и заполнения модой
    # 'Cabin' (Кабина) — обычно слишком много уникальных значений, его пока игнорируем или удаляем.
    # Мы кодируем только те, которые имеют мало уникальных значений.
    categorical_features = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP']

    # Удаление столбцов с высоким количеством пропусков или уникальных значений,
    # которые не будем обрабатывать сейчас (для упрощения задачи)
    df.drop(columns=['Name', 'Cabin', 'PassengerId'], inplace=True, errors='ignore')
    print("Столбцы 'Name', 'Cabin', 'PassengerId' удалены для упрощения.")

    print("-" * 40)

    # ====================================================================
    # ИСПОЛЬЗОВАНИЕ ColumnTransformer ДЛЯ АВТОМАТИЗАЦИИ
    # ====================================================================

    # ColumnTransformer позволяет применить разные преобразования к разным столбцам
    # в один шаг. Это лучшая практика в ML.

    preprocessor = ColumnTransformer(
        transformers=[
            # Трансформатор 1: ЧИСЛОВЫЕ ДАННЫЕ (Нормализация + Заполнение Медианой)
            ('num',
             StandardScaler(),  # Нормализация (StandardScaler)
             numerical_features),

            # Трансформатор 2: КАТЕГОРИАЛЬНЫЕ ДАННЫЕ (Кодирование + Заполнение Модой)
            ('cat',
             OneHotEncoder(handle_unknown='ignore', sparse_output=False),  # Кодирование (One-Hot)
             categorical_features)
        ],
        remainder='passthrough'  # Оставляет необработанные столбцы как есть
    )

    # 1. Заполнение пропусков (теперь внутри ColumnTransformer, но давайте сделаем это отдельно для чистоты)

    # Импутация (заполнение) данных:
    # Это важно сделать ПЕРЕД нормализацией/кодированием.

    # Заполнение числовых данных медианой
    for col in numerical_features:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # Заполнение категориальных данных модой
    for col in categorical_features:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)

    print("✅ Пропущенные значения заполнены медианой/модой.")
    print("-" * 40)

    # 2. Проведение преобразований (Нормализация и Кодирование)

    # Применяем все преобразования к датасету
    df_processed_array = preprocessor.fit_transform(df)

    # Получаем новые имена столбцов после One-Hot кодирования
    feature_names = (numerical_features +
                     list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)))

    # Превращаем результат обратно в DataFrame
    df_processed = pd.DataFrame(df_processed_array, columns=feature_names)

    # ====================================================================
    # ВЫВОД РЕЗУЛЬТАТОВ
    # ====================================================================

    print("✅ Датасет успешно нормализован и закодирован!")
    print("\nВид итогового датасета (первые 5 строк):")
    print(df_processed.head())

    print("-" * 40)
    print(f"Итоговый размер датасета (строк, столбцов): {df_processed.shape}")
    print("\n**Нормализация** видна в столбцах типа 'Age' (значения близки к 0).")
    print("**Кодирование** видно в новых столбцах типа 'HomePlanet_Mars' (значения 0 или 1).")

except FileNotFoundError:
    print(f"❌ Ошибка: Файл '{file_path}' не найден!")
    print("Проверьте, правильно ли указан путь к train.csv.")