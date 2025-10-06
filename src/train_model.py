import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

# Dummy dataset for CI/CD demonstration
data = {
    "Upvote Ratio": [0.8, 0.9, 0.75, 0.6, 0.95],
    "Number of Comments": [12, 45, 8, 15, 25],
    "Post Length": [120, 220, 150, 180, 130],
    "Text Length (chars)": [350, 500, 410, 390, 420],
    "Text Word Count": [60, 80, 70, 65, 75],
    "Subreddit": ["tech", "finance", "sports", "news", "tech"],
    "Flair": ["Discussion", "News", "Tips", "Update", "Discussion"],
    "Domain": ["reddit.com"]*5,
    "Title_Cleaned": ["sample post one", "sample post two", "sample post three", "sample post four", "sample post five"],
    "Score": [200, 350, 180, 150, 400]
}

df = pd.DataFrame(data)

target = "Score"
numeric_features = ["Upvote Ratio", "Number of Comments", "Post Length", "Text Length (chars)", "Text Word Count"]
categorical_features = ["Subreddit", "Flair", "Domain"]
text_feature = "Title_Cleaned"

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ("text", TfidfVectorizer(max_features=100), text_feature)
])

X = df[numeric_features + categorical_features + [text_feature]]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(n_estimators=50, random_state=42))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}, R2: {r2:.2f}")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/best_model.pkl")
print("✅ Model trained and saved as models/best_model.pkl")
