import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np

# Указываем путь к файлу
file_path = 'train.csv'

try:
    df = pd.read_csv(file_path)
    print(" Данные успешно загружены!")
    print("-" * 40)

    numerical_features = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    categorical_features = ['HomePlanet', 'Destination', 'CryoSleep', 'VIP']


    df.drop(columns=['Name', 'Cabin', 'PassengerId'], inplace=True, errors='ignore')
    print("Столбцы 'Name', 'Cabin', 'PassengerId' удалены для упрощения.")

    print("-" * 40)


    preprocessor = ColumnTransformer(
        transformers=[
            ('num',
             StandardScaler(),  # Нормализация (StandardScaler)
             numerical_features),

            ('cat',
             OneHotEncoder(handle_unknown='ignore', sparse_output=False),  # Кодирование (One-Hot)
             categorical_features)
        ],
        remainder='passthrough'  # Оставляет необработанные столбцы как есть
    )

    for col in numerical_features:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)

    # Заполнение категориальных данных модой
    for col in categorical_features:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)

    print("Пропущенные значения заполнены медианой/модой.")
    print("-" * 40)


    target = df['Transported']
    df_features = df.drop(columns=['Transported'], errors='ignore')  # Работаем только с признаками


    df_processed_array = preprocessor.fit_transform(df_features)


    feature_names = (
            numerical_features +
            list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    )


    df_processed = pd.DataFrame(df_processed_array, columns=feature_names)


    df_processed['Transported'] = target.values

    print("Датасет успешно нормализован и закодирован!")
    print("\nВид итогового датасета (первые 5 строк):")
    print(df_processed.head())

    print("-" * 40)
    print(f"Итоговый размер датасета (строк, столбцов): {df_processed.shape}")
    print("\nТеперь количество столбцов корректно: 16 признаков + 1 целевая переменная ('Transported').")

except FileNotFoundError:
    print(f"Ошибка: Файл '{file_path}' не найден!")
    print("Проверьте, правильно ли указан путь к train.csv.")