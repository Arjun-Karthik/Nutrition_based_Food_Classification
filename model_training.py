import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Load dataset
df = pd.read_csv("synthetic_food_dataset_imbalanced.csv")
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# Encode binary flags
df['Is_Vegan'] = df['Is_Vegan'].astype(int)
df['Is_Gluten_Free'] = df['Is_Gluten_Free'].astype(int)
df = pd.get_dummies(df, columns=['Meal_Type', 'Preparation_Method'], drop_first=True)

# Encode target
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Food_Name'])

X = df.drop(['Food_Name', 'Label'], axis=1)
y = df['Label']

# Split and preprocess
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Apply PCA
pca = PCA(n_components=0.95, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Save preprocessing steps
os.makedirs('Models')
joblib.dump(scaler, "Models/scaler.pkl")
joblib.dump(pca, "Models/pca.pkl")
joblib.dump(le, "Models/label_encoder.pkl")

# Models and hyperparameters
model_param_grid = {
    "Logistic Regression": (LogisticRegression(), {"C": [0.01, 0.1, 1, 10]}),
    "Decision Tree": (DecisionTreeClassifier(), {"max_depth": [5, 10, 20], "criterion": ["gini", "entropy"]}),
    "Random Forest": (RandomForestClassifier(), {"n_estimators": [50, 100], "max_depth": [10, 20, None]}),
    "K-Nearest Neighbors": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7]}),
    "Support Vector Machine": (SVC(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
    "XGBoost": (XGBClassifier(eval_metric='mlogloss', use_label_encoder=False), {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1, 0.2]}),
    "Gradient Boosting": (GradientBoostingClassifier(), {"n_estimators": [50, 100], "learning_rate": [0.01, 0.1]})
}

# Train Model
results = []
best_model = None
best_score = -1
best_model_name = ""

for name, (model, param_grid) in model_param_grid.items():
    search = RandomizedSearchCV(model, param_grid, cv=3, scoring="accuracy", n_iter=5, random_state=42, n_jobs=-1)
    search.fit(X_train, y_train)

    # Predict
    best_est = search.best_estimator_
    y_pred = best_est.predict(X_test)


    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1 Score": f1})
    joblib.dump(best_est, f"Models/model_{name.replace(' ', '_')}.pkl")

    if f1 > best_score:
        best_score = f1
        best_model = best_est
        best_model_name = name

# Save Best Model
joblib.dump(best_model, "Models/best_model.pkl")

# Save results
results_df = pd.DataFrame(results).sort_values(by= 'Accuracy', ascending=False)
results_df.to_csv("Models/model_results.csv", index=False)
print(f"Best model: {best_model_name} saved as best_model.pkl")
