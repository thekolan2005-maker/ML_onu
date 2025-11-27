import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# school - student's school (binary: 'GP' - Gabriel Pereira or 'MS' - Mousinho da Silveira)
# sex - student's sex (binary: 'F' - female or 'M' - male)
# age - student's age (numeric: from 15 to 22)
# address - student's home address type (binary: 'U' - urban or 'R' - rural)
# famsize - family size (binary: 'LE3' - less or equal to 3 or 'GT3' - greater than 3)
# Pstatus - parent's cohabitation status (binary: 'T' - living together or 'A' - apart)
# Medu - mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
# Fedu - father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 – 5th to 9th grade, 3 – secondary education or 4 – higher education)
# Mjob - mother's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
# Fjob - father's job (nominal: 'teacher', 'health' care related, civil 'services' (e.g. administrative or police), 'at_home' or 'other')
# reason - reason to choose this school (nominal: close to 'home', school 'reputation', 'course' preference or 'other')
# guardian - student's guardian (nominal: 'mother', 'father' or 'other')
# traveltime - home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
# studytime - weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
# failures - number of past class failures (numeric: n if 1<=n<3, else 4)
# schoolsup - extra educational support (binary: yes or no)
# famsup - family educational support (binary: yes or no)
# paid - extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
# activities - extra-curricular activities (binary: yes or no)
# nursery - attended nursery school (binary: yes or no)
# higher - wants to take higher education (binary: yes or no)
# internet - Internet access at home (binary: yes or no)
# romantic - with a romantic relationship (binary: yes or no)
# famrel - quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
# freetime - free time after school (numeric: from 1 - very low to 5 - very high)
# goout - going out with friends (numeric: from 1 - very low to 5 - very high)
# Dalc - workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
# Walc - weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
# health - current health status (numeric: from 1 - very bad to 5 - very good)
# absences - number of school absences (numeric: from 0 to 93)

# === Настройки ===
file_path = 'student-mat.csv'  # путь к исходному файлу
output_dir = r'B:\ML\lab2'  # исправленный путь для выгрузки
output_file_name = 'student-mat-normalization.csv'
output_path = os.path.join(output_dir, output_file_name)

# Метод нормализации: "minmax" или "zscore"
fill_method = "median"
try:
    # === Загрузка данных ===
    df = pd.read_csv(file_path)
    print(f"Данные '{file_path}' успешно загружены!")
    print("-" * 40)

    # === Удаляем ненужные колонки ===
    for col in ['school','address','famsize','Medu','Fedu','Mjob','Fjob','traveltime','internet','reason']:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
            print(f"Колонка '{col}' удалена из итоговой таблицы.")
    print("-" * 40)

    print("Информация о датасете (df.info()):")
    df.info()
    print("-" * 40)


    # === Проверка пропусков ===
    print("Количество пропущенных значений (до заполнения):")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])
    print("-" * 40)
    #
    # # === Заполнение пропусков ===
    #
    # numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    # categorical_cols = df.select_dtypes(exclude=['int64', 'float64']).columns
    #
    # for col in numeric_cols:
    #     if fill_method == "median":
    #         fill_value = df[col].median()
    #         method_text = "медианой"
    #     else:  # mean
    #         fill_value = df[col].mean()
    #         method_text = "средним"
    #
    #     df[col].fillna(fill_value, inplace=True)
    #     print(f"-> Числовая колонка '{col}' заполнена {method_text}: {fill_value:.2f}")
    #
    # # Категориальные — модой, безопасно
    # for col in categorical_cols:
    #     mode_value = df[col].mode()[0]
    #     df[col] = df[col].fillna(mode_value)
    #     df[col] = df[col].infer_objects(copy=False)
    #     print(f"-> Категориальная колонка '{col}' заполнена модой: {mode_value}")
    #
    # print("-" * 40)
    # print("Количество пропущенных значений (после заполнения):")
    # print(df.isnull().sum()[df.isnull().sum() > 0])
    # print("-" * 40)



    # === Нормализация числовых данных ===
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # 2. ИСКЛЮЧЕНИЕ: Убираем 'Walc' из списка для нормализации
   # COLS_TO_EXCLUDE_FROM_SCALING = ['Dalc','Walc','failures','famrel','freetime','goout','health', 'absences', 'G1', 'G2', 'G3','studytime']
    cols_to_scale = [col for col in numeric_cols if col not in COLS_TO_EXCLUDE_FROM_SCALING]

    print(f"Числовые колонки для нормализации: {cols_to_scale}")

    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print("Нормализация завершена.")
    print("-" * 40)

    # 1. Определите список колонок, которые вы хотите преобразовать
    # Вы можете менять этот список вручную, чтобы выбрать нужные колонки.
    columns_to_encode = df.select_dtypes(include=['object']).columns.tolist()

    print("--- Начинаем выборочное One-Hot кодирование ---")

    # 2. Итерация по списку и преобразование каждой колонки
    for col in columns_to_encode:
        if col in df.columns:
            print(f" Преобразуем колонку: '{col}'")
            # Выполняем One-Hot Encoding
            # drop_first=0 (False) оставляет все созданные фиктивные переменные
            df = pd.get_dummies(df, columns=[col], drop_first=0)
            print(f"   Колонка '{col}' успешно преобразована.")
        else:
            print(f"Колонка '{col}' не найдена в датафрейме. Пропускаем преобразование.")

    print("--- Кодирование завершено ---")

    # === Сохранение ===
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Файл успешно сохранён: {output_path}")

except FileNotFoundError:
    print(f"Ошибка: файл '{file_path}' не найден!")
except Exception as e:
    print("Произошла ошибка:", e)





