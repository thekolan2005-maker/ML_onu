import pandas as pd # Эта штука умеет читать наши таблички с данными.
import numpy as np # Эта штука умеет делать всякую сложную математику, чтобы Python не сгорел.
from sklearn.model_selection import train_test_split # Машинка для резки данных на части. Очень острая!
from sklearn.linear_model import LinearRegression, LogisticRegression # Наши главные игрушки: Линейная и Логистическая регрессия. Звучит умно, но это просто линии.
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error # Три способа сказать "Ой, как сильно ты ошибся!"
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # Как оценить, угадал ли ты или нет.
import matplotlib.pyplot as plt # Умеет рисовать картинки.
import seaborn as sns # Делает картинки красивыми, чтобы было что показать.

# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ

# Загружаем нашу волшебную табличку. Она лежит тут!
df = pd.read_csv("student-mat-normalization.csv")
# Смотрим на первые строчки, чтобы понять, что мы вообще загрузили.
print("Голова 10")
print(df.head(10))

# --- ФИКС: УДАЛЯЕМ ЛИШНЮЮ ПРЕОБРАБОТКУ ---
# Примечание: Этот блок закомментирован, потому что ваш новый файл уже полностью
# предобработан (нормализован и закодирован) и не содержит старых колонок.

# # У нас там есть столбец 'Destination' с названиями планет (типа 'TRAPPIST-1e').
# # Компьютеры ненавидят слова, поэтому мы должны превратить их в числа (0 или 1).
# # Это называется One-Hot Encoding — типа, включаем и выключаем лампочки.
# # df = pd.get_dummies(df, columns=['Destination'], drop_first=True)

# # Столбцы типа 'CryoSleep' и 'VIP' - это True/False.
# # Модели этого тоже не понимают, поэтому мы говорим: True = 1, False = 0. Все просто.
# # bool_cols = ['CryoSleep', 'VIP', 'HomePlanet_Europa', 'HomePlanet_Mars']
# # df[bool_cols] = df[bool_cols].astype(int)

# Список всех столбцов, которые абсолютно бесполезны для предсказания (ID и имена).
# В этом наборе данных таких явных столбцов нет, поэтому оставляем пустой список.
COLS_TO_EXCLUDE = []


# 2. ЗАДАЧА РЕГРЕССИИ (Предсказываем возраст 'age' - это число, поэтому Регрессия)


target_reg = "age" # Цель №1: Алкоголь по выходным

# X_reg - это все, что мы знаем (все признаки).
cols_to_drop_reg = COLS_TO_EXCLUDE + [target_reg]
X_reg = df.drop(columns=cols_to_drop_reg)
y_reg = df[target_reg] # y_reg - это то, что мы хотим угадать (возраст).

# Режем данные! 60% для обучения (Train), 40% для проверки (Test).
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.4, random_state=42
)

print(f"\n--- Размеры кусочков для Регрессии ---")
print(f"Кусок для обучения: {X_train_reg.shape[0]} строк. На нем модель учится.")
print(f"Кусок для теста: {X_test_reg.shape[0]} строк. На нем модель сдаёт экзамен.")

# Берем простую Линейную Регрессию (это просто рисование линии).
linear_model = LinearRegression()
linear_model.fit(X_train_reg, y_train_reg) # Команда "Учись!"

# Говорим модели: "А теперь угадай возраст для тестовых данных!"
y_pred_reg_test = linear_model.predict(X_test_reg)

# Оценка: Считаем, насколько сильно мы ошиблись.

# MSE - это типа "сильно поругать за большие ошибки".
mse = mean_squared_error(y_test_reg, y_pred_reg_test)
print(f"\nСреднеквадратичная ошибка (MSE): {mse:.4f}")

# RMSE - то же самое, что MSE, но легче понять, потому что числа похожи на возраст.
rmse = root_mean_squared_error(y_test_reg, y_pred_reg_test)
print(f"Корень среднеквадратичной ошибки (RMSE): {rmse:.4f}")

# MAE - это "ругаем одинаково за все ошибки".
mae = mean_absolute_error(y_test_reg, y_pred_reg_test)
print(f"Средняя абсолютная ошибка (MAE): {mae:.4f}")
# lasso = Lasso(alpha=0.5)
# lasso.fit(X_train, y_train)
#
# ridge = Ridge(alpha=2.0)
# ridge.fit(X_train, y_train)
#
# elastic_model = ElasticNet(alpha=2.0, l1_ratio=0.5)
# elastic_model.fit(X_train, y_train)
#
# lasso_pred = lasso.predict(X_test)
# ridge_pred = ridge.predict(X_test)
# elastic_pred = elastic_model.predict(X_test)

# 3. ЗАДАЧА КЛАССИФИКАЦИИ (Предсказываем 'sex_F' - это ДА или НЕТ, поэтому Классификация)


target_clf = "sex_F" # Цель №2: Угадать, пол студента - женский (0 или 1).

# X_clf - снова все наши признаки.
cols_to_drop_clf = COLS_TO_EXCLUDE + [target_clf]
X_clf = df.drop(columns=cols_to_drop_clf)
y_clf = df[target_clf] # y_clf - это то, что мы хотим угадать (0 или 1).

# Снова режем, но для другой задачи.
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.4, random_state=42
)

# Берем Логистическую Регрессию (звучит как линия, но она угадывает "да" или "нет").
logreg_model = LogisticRegression(solver='liblinear', random_state=42)
logreg_model.fit(X_train_clf, y_train_clf) # Команда "Учись, угадывай 0 или 1!"

# Угадываем, кто спал в капсуле, а кто нет.
y_pred_clf_test = logreg_model.predict(X_test_clf)

# Оценка: Насколько часто мы угадали правильно.
accuracy = accuracy_score(y_test_clf, y_pred_clf_test)
print(f"\nAccuracy score (Процент правильных ответов): {accuracy:.4f}")

# Матрица ошибок! Это табличка, где видно, как именно мы ошиблись (сделали 0, когда надо 1, и наоборот).
cm = confusion_matrix(y_test_clf, y_pred_clf_test)
print("\nМатрица ошибок (Confusion Matrix):")
print(cm)

# Комплексный отчет - тут все остальные сложные метрики (Precision, Recall и F1-Score из методички).
print("\nОтчет по классификации:")
print(classification_report(y_test_clf, y_pred_clf_test))


# 4. РИСУЕМ КАРТИНКУ


# Строим красивую цветную табличку, чтобы матрица ошибок была понятнее.
plt.figure(figsize=(4, 3))
sns.heatmap(cm, annot=True, fmt='d', cmap='bwr',
            xticklabels=['Предсказал 0 (Муж.)', 'Предсказал 1 (Жен.)'],
            yticklabels=['Истинный 0 (Муж.)', 'Истинный 1 (Жен.)'])
plt.title('Матрица ошибок ')
plt.ylabel('Настоящий ответ')
plt.xlabel('Ответ модели')
plt.savefig('confusion_matrix.png')
plt.show()