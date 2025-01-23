from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Fetching the dataset
wine_quality = fetch_ucirepo(id=186)

#Extract the x and y from data set
X = wine_quality.data.features
y = wine_quality.data.targets

#Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the random forest regressor algo
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

#Print the results to console
print("   Random Forest Regressor    ")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, rf_preds):.4f}")
print(f"Mean Squared Error (MSE): {mean_squared_error(y_test, rf_preds):.4f}")
print(f"R² Score: {r2_score(y_test, rf_preds):.4f}")
