import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Загрузите данные
data = pd.read_csv("synthetic_data.csv")

# Замените категориальные переменные с помощью one-hot encoding
data = pd.get_dummies(data, columns=['Shelf_Location', 'Urban', 'US'])

# Разделите данные на признаки (X) и целевую переменную (y)
X = data.drop('Sales', axis=1)
y = data['Sales']

# Разделите данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создайте модель случайного леса
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Обучите модель
rf.fit(X_train, y_train)

# Сделайте прогнозы
y_pred = rf.predict(X_test)

# Вычислите среднеквадратичную ошибку
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Random Forest RMSE:", rmse)

# Важность признаков
importances = rf.feature_importances_
feature_importances = pd.DataFrame({"Feature": X.columns, "Importance": importances})
print(feature_importances.sort_values("Importance", ascending=False))

# Удаление наименее важных признаков
selected_features = feature_importances[feature_importances["Importance"] > 0.01]["Feature"]
X_selected = X[selected_features]

# Разделите данные на обучающую и тестовую выборки с использованием отфильтрованных признаков
X_train_selected, X_test_selected, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Создайте модель XGBoost
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Обучите модель XGBoost
xgb.fit(X_train, y_train)

# Сделайте прогнозы для XGBoost
y_pred = xgb.predict(X_test)

# Вычислите среднеквадратичную ошибку для XGBoost
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred))
print("XGBoost RMSE:", rmse_xgb)

# Обучите модель случайного леса и XGBoost с использованием отфильтрованных признаков и сравните результаты
rf.fit(X_train_selected, y_train)
y_pred = rf.predict(X_test_selected)
rmse_rf_selected = np.sqrt(mean_squared_error(y_test, y_pred))
print("Random Forest RMSE (selected features):", rmse_rf_selected)

xgb.fit(X_train_selected, y_train)
y_pred = xgb.predict(X_test_selected)
rmse_xgb_selected = np.sqrt(mean_squared_error(y_test, y_pred))
print("XGBoost RMSE (selected features):", rmse_xgb_selected)
data_df = data

# Датафрейм с важностью признаков
feature_importances_df = feature_importances.sort_values("Importance", ascending=False)

# Датафрейм с результатами
results = {"Model": ["Random Forest", "XGBoost", "Random Forest (selected features)", "XGBoost (selected features)"],
           "RMSE": [rmse, rmse_xgb, rmse_rf_selected, rmse_xgb_selected]}
results_df = pd.DataFrame(results)

# Создайте объект ExcelWriter
writer = pd.ExcelWriter("results_and_data.xlsx", engine='xlsxwriter')

# Запишите каждый датафрейм в разные листы Excel-файла
data_df.to_excel(writer, sheet_name='Data', index=False)
feature_importances_df.to_excel(writer, sheet_name='Feature Importances', index=False)
results_df.to_excel(writer, sheet_name='Results', index=False)

# Сохраните файл Excel
writer._save()