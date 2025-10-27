import pandas as pd


file_path = 'train.csv'


try:
    df = pd.read_csv(file_path)
    print("Данные 'Титаник' успешно загружены!")
    print("-" * 40)

    print("Первые 5 строк датасета (df.head()):")
    print(df.head())
    print("-" * 40)

    print("Информация о датасете (df.info()):")
    df.info()
    print("-" * 40)

    print(f"Размер датасета (строк, столбцов): {df.shape}")

except FileNotFoundError:
    print(f"Ошибка: Файл '{file_path}' не найден!")
    print("Убедитесь, что 'train.csv' находится в папке проекта PyCharm.")