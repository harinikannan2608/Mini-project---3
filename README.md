import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from scipy.stats import zscore, chi2_contingency
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV

sns.set(style="whitegrid")

# ------------------------------------
# Data Collection
# ------------------------------------
print("Loading dataset...")
file_id = "1VUb9ucTsroGDBOPcwpOfXwzDi-rd4wqQ"
download_url = f"https://drive.google.com/uc?id={file_id}"
df = pd.read_csv(download_url)
print("Dataset loaded\n")

# ------------------------------------
# Data Understanding
# ------------------------------------
print(f"Shape: {df.shape}")
print("\nColumns & Data Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
print("\nSummary Statistics:\n", df.describe())

# ------------------------------------
# Feature Engineering
# ------------------------------------
pickup_col = 'tpep_pickup_datetime'
dropoff_col = 'tpep_dropoff_datetime'

df[pickup_col] = pd.to_datetime(df[pickup_col], errors='coerce')
df[dropoff_col] = pd.to_datetime(df[dropoff_col], errors='coerce')

df['pickup_hour'] = df[pickup_col].dt.hour
df['pickup_day'] = df[pickup_col].dt.day_name()
df['is_weekend'] = df['pickup_day'].isin(['Saturday', 'Sunday']).astype(int)
df['is_night'] = df['pickup_hour'].apply(lambda x: 1 if (x >= 22 or x < 5) else 0)
df['trip_duration_min'] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

df['trip_distance_km'] = haversine(
    df['pickup_longitude'], df['pickup_latitude'],
    df['dropoff_longitude'], df['dropoff_latitude']
)

df['fare_per_km'] = df['total_amount'] / df['trip_distance_km']
df['fare_per_min'] = df['total_amount'] / df['trip_duration_min']

print("Feature engineering completed\n")

# ------------------------------------
# Data Transformation
# ------------------------------------
def remove_outliers_z(df, cols, threshold=3):
    df_clean = df.copy()
    for col in cols:
        z = np.abs(zscore(df_clean[col].dropna()))
        df_clean = df_clean[(z < threshold) | df_clean[col].isnull()]
    return df_clean

print("Handling outliers")
num_cols = ['total_amount', 'trip_distance_km', 'trip_duration_min']
df = remove_outliers_z(df, num_cols)
print("Outliers removed")

for col in num_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)
    if abs(df[col].skew()) > 1:
        if (df[col] >= 0).all():
            df[f'{col}_log'] = np.log1p(df[col])

print("Skewness fixed")

le = LabelEncoder()
df['pickup_day_enc'] = le.fit_transform(df['pickup_day'])
print("Categorical encoding completed\n")

# ------------------------------------
# Feature Selection
# ------------------------------------
print("Performing Feature Selection")

target = 'total_amount'
features = [
    'trip_distance_km', 'trip_duration_min', 'pickup_hour',
    'is_weekend', 'is_night', 'fare_per_km', 'fare_per_min', 'pickup_day_enc'
]

X = df[features]
y = df[target]

X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(0)

print("Pearson Correlation with Target")
print(df[features + [target]].corr()[target].sort_values(ascending=False))

print("Chi-Square Test for Categorical Variables")
df['target_binned'] = pd.qcut(y, q=4, labels=False)
cat_cols = ['is_weekend', 'is_night', 'pickup_day_enc']
for col in cat_cols:
    contingency = pd.crosstab(df[col], df['target_binned'])
    chi2, p, dof, ex = chi2_contingency(contingency)
    print(f"{col}: Chi2 = {chi2:.2f}, p-value = {p:.4f}")

print("Feature Importance from Random Forest")
rf = RandomForestRegressor(random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
print(importances)

plt.figure(figsize=(8,6))
importances.plot(kind='bar')
plt.title("Feature Importance from Random Forest")
plt.ylabel("Importance")
plt.show()

print("Feature Selection completed\n")

# ------------------------------------
# Model Building
# ------------------------------------
print("Building and evaluating regression models")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'RandomForest': RandomForestRegressor(random_state=42),
    'GradientBoosting': GradientBoostingRegressor(random_state=42)
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, preds)
    results.append([name, r2, mse, rmse, mae])

results_df = pd.DataFrame(results, columns=['Model', 'R2', 'MSE', 'RMSE', 'MAE'])
print("\nModel Comparison:\n", results_df.sort_values(by='R2', ascending=False))

best_model_name = results_df.sort_values(by='R2', ascending=False).iloc[0]['Model']
print(f"\nBest model based on R2: {best_model_name}")

# ------------------------------------
# Hyperparameter Tuning (faster)
# ------------------------------------
print("\nHyperparameter tuning for best model (RandomizedSearchCV)")

param_dist = {
    'n_estimators': [10, 50, 100],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4]
}

# Optional: sample smaller set just for tuning
X_tune, _, y_tune, _ = train_test_split(X_train, y_train, train_size=5000, random_state=42)

random_search = RandomizedSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='r2',
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_tune, y_tune)

print("Best parameters:", random_search.best_params_)
best_model = random_search.best_estimator_
preds = best_model.predict(X_test)

print("\nPerformance of Tuned RandomForest:")
print(f"R2: {r2_score(y_test, preds):.4f}")
print(f"MSE: {mean_squared_error(y_test, preds):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, preds)):.4f}")
print(f"MAE: {mean_absolute_error(y_test, preds):.4f}")

# ------------------------------------
# Save Best Model
# ------------------------------------
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("\n✅ Best model saved as 'best_model.pkl'")
