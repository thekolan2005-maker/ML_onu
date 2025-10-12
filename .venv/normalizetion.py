from statistics import median_low

import pandas as pd

file_path = 'train.csv'

try:

    df = pd.read_csv(file_path)
    print("Данные 'Space Titanic' успешно загружены!")
    print("-" * 40)

    print("Количество пропущенных значений (до заполнения):")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])  # Выводим только столбцы с пропусками
    print("-" * 40)


    median_age = df['Age'].median()
    print(f"-> Медиана для столбца 'Age': {median_age:.2f}")

    df['Age'].fillna(median_age, inplace=True)
    print("   Столбец 'Age' заполнен медианой.")

    mode_home_planet = df['HomePlanet'].mode()[0]
    print(f"-> Мода (самое частое значение) для столбца 'HomePlanet': {mode_home_planet}")

    df['HomePlanet'].fillna(mode_home_planet, inplace=True)
    print("   Столбец 'HomePlanet' заполнен модой.")

    mode_Spa= df['Spa'].mode()[0]
    print(f"-> Мода для столбца 'Spa': {mode_Spa}")
    df['Spa'].fillna(mode_Spa, inplace=True)
    print(" Столбец 'Spa' заполнен модой")

    mode_destination = df['Destination'].mode()[0]
    print(f"-> Мода (самое частое значение) для столбца 'Destination': {mode_destination}")

    df['Destination'].fillna(mode_destination, inplace=True)
    print("   Столбец 'Destination' заполнен модой.")

    print("-" * 40)


    print(" Количество пропущенных значений (ПОСЛЕ заполнения):")

    missing_values_after = df.isnull().sum()


    print(missing_values_after[['Age', 'HomePlanet', 'Destination', 'Spa']])



except FileNotFoundError:
    print(f" Ошибка: Файл '{file_path}' не найден!")
    print("Проверьте, правильно ли указан путь к train.csv.")