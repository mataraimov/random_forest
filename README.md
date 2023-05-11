# Sales Analysis Report

## Introduction

The data: In this assignment, we used a synthetic dataset with 11 variables and over 400 records. The dataset provided was modified to better fit the problem at hand.


(Results and data sheets link:https://docs.google.com/spreadsheets/d/1spOYn63o7c7-aohXtQzgb0ZkvspgIX1X0m_EU8g2Cwo/edit?usp=sharing)

The attributes are as follows:

- Sales: Unit sales (in thousands) at each location
- Competitor Price: Price charged by competitor at each location
- Income: Community income level (in thousands of dollars)
- Advertising: Local advertising budget for the company at each location (in thousands of dollars)
- Population: Population size in the region (in thousands)
- Price: Price the company charges for car seats at each site
- Shelf Location: A factor with levels Bad, Good, and Medium indicating the quality of the shelving location for the car seats at each site
- Age: Average age of the local population
- Education: Education level at each location
- Urban: A factor with levels No and Yes to indicate whether the store is in an urban or rural location
- US: A factor with levels No and Yes to indicate whether the store is in the US or not

## Data preprocessing

We started by loading the data using pandas and then proceeded to preprocess the data. First, we converted the categorical variables (Shelf Location, Urban, and US) using one-hot encoding:

data = pd.get_dummies(data, columns=['Shelf_Location', 'Urban', 'US'])


Next, we split the data into input features (X) and target variable (y) and then into training and testing sets:

X = data.drop('Sales', axis=1)
y = data['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


## Model building and evaluation

We built two models to predict sales based on the given attributes: a Random Forest model and an XGBoost model. We trained each model and evaluated their performance using the root mean squared error (RMSE) metric.

### Random Forest model

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


### XGBoost model

xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))


## Results

- Random Forest RMSE: 28.842365575659702
- XGBoost RMSE: 31.151933391751058

## Feature importance analysis

We analyzed the importance of each feature in the Random Forest model to understand which factors have the most significant impact on sales. The top 6 features were:

- Competitor_Price
- Population
- Income
- Price
- Advertising
- Age

## Model building and evaluation with selected features

We built new models using only the top 6 features to see if the performance could be improved.

### Random Forest model (selected features)

X_selected = X[['Competitor_Price', 'Population', 'Income', 'Price', 'Advertising', 'Age']]
X_train_selected, X_test_selected, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
rf.fit(X_train_selected, y_train)
y_pred_selected = rf.predict(X_test_selected)
rmse_selected = np.sqrt(mean_squared_error(y_test, y_pred_selected))
XGBoost model (selected features)
xgb.fit(X_train_selected, y_train)
y_pred_selected = xgb.predict(X_test_selected)
rmse_selected = np.sqrt(mean_squared_error(y_test, y_pred_selected))
Results (selected features)

- Random Forest RMSE (selected features): 27.684614354206435
- XGBoost RMSE (selected features): 30.5385707938815
Conclusion

By using only the top 6 features, we were able to slightly improve the performance of both the Random Forest and XGBoost models. The Random Forest model, in particular, showed the best performance with a lower RMSE value. The most important factors that influenced sales were Competitor_Price, Population, Income, Price, Advertising, and Age.) had the most significant impact on sales.
In the absence of visual graphs and charts, we can still discuss the findings and their implications for the company. The results suggest that the company should pay close attention to competitor prices, the population in the area, income levels, and their own product pricing. Additionally, the impact of advertising and the age of the local population should not be underestimated.
By understanding the importance of these factors, the company can make more informed decisions and develop strategies that will help them increase sales in different locations.
