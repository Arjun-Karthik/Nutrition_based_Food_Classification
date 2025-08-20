import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import re
import os
import requests

# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(page_title="NutriClass Visualizer", 
                   page_icon='🍱',
                   layout="wide")

st.markdown("""<h1 style = 'text-align:center;'>🍵 NutriClass - Food Classification Using Nutritional Data</h1>
                <hr style='border-top: 3px solid #bbb;'>""", 
                unsafe_allow_html=True)

# -------------------- MODEL DOWNLOAD HELPER --------------------
def load_model_from_url(url, local_path):
    """
    Download model from URL if not already cached locally.
    """
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with st.spinner(f"Downloading {os.path.basename(local_path)}..."):
            response = requests.get(url)
            response.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(response.content)
    return joblib.load(local_path)

# Replace these URLs with your Hugging Face / Google Drive / S3 links
MODEL_URLS = {
    "K-Nearest Neighbors": "https://huggingface.co/yourname/nutriclass-models/resolve/main/K-Nearest_Neighbors.pkl",
    "Random Forest": "https://huggingface.co/yourname/nutriclass-models/resolve/main/Random_Forest.pkl",
    "Decision Tree": "https://huggingface.co/yourname/nutriclass-models/resolve/main/Decision_Tree.pkl",
}

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_food_dataset_imbalanced.csv")

df_raw = load_data()

# -------------------- DATASET DIAGNOSTICS --------------------
st.markdown("<h2 style = text-align:center;>📋 Dataset Diagnostics</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"<h5 style = text-align:center;>🔢 Original Shape: {df_raw.shape}</h5>", unsafe_allow_html=True)

with col2:
    duplicate_count = df_raw.duplicated().sum()
    st.markdown(f"<h5 style = text-align:center;>♻️ Duplicate Rows: {duplicate_count}</h5>", unsafe_allow_html=True)

with col3:
    df_raw = df_raw.drop_duplicates()
    st.markdown(f"<h5 style = text-align:center;>🧹 Shape After Removing Duplicates: {df_raw.shape}</h5>", unsafe_allow_html=True)

st.markdown("""<hr style = 'width:95%; border-top: 3px solid #bbb;'>""", unsafe_allow_html=True)

# -------------------- MISSING VALUES --------------------
missing_values = df_raw.isnull().sum()
st.subheader("❗ Missing Values per Column")
missing_df = missing_values[missing_values > 0].reset_index()
missing_df.columns = ['Column', 'No. of Missing Values']
missing_df.index = range(1, len(missing_df) + 1)
st.dataframe(missing_df)

# -------------------- SAMPLE DATA --------------------
st.subheader("📊 Data Overview")
df_raw.index = range(1, len(df_raw) + 1)
st.dataframe(df_raw.head(10))

# -------------------- EDA ANALYSIS --------------------
st.header("🔍 Exploratory Data Analysis (EDA)")

# Correlation Heatmap
numeric_cols = df_raw.select_dtypes(include=['float64', 'int64']).columns
corr = df_raw[numeric_cols].corr()
fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="📌 Correlation Heatmap")
st.plotly_chart(fig_corr, use_container_width=True)

# Prepare features and target for SMOTE
df_smote = df_raw.copy()
df_smote.dropna(inplace=True)
X = df_smote.drop(columns=['Food_Name'])
y = df_smote['Food_Name']

# Encode categorical columns before SMOTE
X_encoded = pd.get_dummies(X, drop_first=True)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

# Before SMOTE
fig_before = px.bar(
    y.value_counts().reset_index(name="count"),
    x="Food_Name", y="count",
    labels={"Food_Name": "Food Name", "count": "Count"},
    title="🍽️ Class Distribution (Imbalanced)"
)

# After SMOTE
fig_after = px.bar(
    y_resampled.value_counts().reset_index(name="count"),
    x="Food_Name", y="count",
    labels={"Food_Name": "Food Name", "count": "Count"},
    title="🍽️ Class Distribution (Balanced)"
)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_before, use_container_width=True)
with col2:
    st.plotly_chart(fig_after, use_container_width=True)

# 4. Boxplot - Protein vs Meal_Type
fig_box = px.box(df_raw, x="Meal_Type", y="Protein", title="🥩 Protein Distribution by Meal Type")
st.plotly_chart(fig_box, use_container_width=True)

# -------------------- MODEL RESULTS --------------------
results_df = pd.read_csv("model_results.csv")
results_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']] *= 100
results_df = results_df.round(2)

st.subheader("📋 Model Comparison Table")
sorted_results = results_df.sort_values(by="Accuracy", ascending=False)
st.dataframe(
    sorted_results.set_index("Model")
        .style.format("{:.1f}%")
        .background_gradient(cmap="Blues", axis=0)
)

# BarChart for model metrics
fig_bar = px.bar(sorted_results.melt(id_vars='Model'), x='Model', y='value', color='variable',
                 barmode='group', title='📈 Model Metrics Comparison',
                 labels={'value': 'Percentage (%)', 'variable': 'Metric'})
st.plotly_chart(fig_bar, use_container_width=True)

# -------------------- PCA VISUALIZATION --------------------
df_example = load_data().dropna().drop_duplicates()
df_example = pd.get_dummies(df_example, columns=['Meal_Type', 'Preparation_Method'], drop_first=True)
df_example = df_example.drop(['Food_Name'], axis=1)
df_example['Is_Vegan'] = df_example['Is_Vegan'].astype(int)
df_example['Is_Gluten_Free'] = df_example['Is_Gluten_Free'].astype(int)

scaler_for_pca = StandardScaler()
X_scaled = scaler_for_pca.fit_transform(df_example)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
le_cm = joblib.load("Models/label_encoder.pkl")
pca_df['Label'] = le_cm.fit_transform(load_data().dropna().drop_duplicates()['Food_Name'])

fig_pca = px.scatter(pca_df, x='PC1', y='PC2', color=pca_df['Label'].astype(str),
                     title="🔵 PCA - Nutritional Feature Reduction (2D)")
st.plotly_chart(fig_pca, use_container_width=True)

# -------------------- CONFUSION MATRIX --------------------
st.header("🧩 Confusion Matrix")
selected_model = st.selectbox("Select Model to View Confusion Matrix", results_df['Model'])
safe_model_name = re.sub(r'[^a-zA-Z0-9]', '_', selected_model)
model_url = MODEL_URLS.get(selected_model)
model_file = f"Models/{safe_model_name}.pkl"

try:
    loaded_model = load_model_from_url(model_url, model_file)
    scaler = joblib.load("Models/scaler.pkl")
    pca = joblib.load("Models/pca.pkl")
    le = joblib.load("Models/label_encoder.pkl")

    df_y = load_data().dropna().drop_duplicates()
    df_y['Is_Vegan'] = df_y['Is_Vegan'].astype(int)
    df_y['Is_Gluten_Free'] = df_y['Is_Gluten_Free'].astype(int)
    df_y = pd.get_dummies(df_y, columns=['Meal_Type', 'Preparation_Method'], drop_first=True)
    X = df_y.drop(columns=['Food_Name']).reindex(columns=scaler.feature_names_in_, fill_value=0)

    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    y = le_cm.transform(load_data().dropna().drop_duplicates()['Food_Name'])
    y_pred = loaded_model.predict(X_pca)

    cm = confusion_matrix(y, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=le_cm.classes_, y=le_cm.classes_,
                       title=f"Confusion Matrix - {selected_model}")
    st.plotly_chart(fig_cm, use_container_width=True)

except Exception as e:
    st.error(f"Model load failed for {selected_model}: {e}")

# -------------------- PREDICTION SECTION --------------------
st.header("🎯 Try Prediction")

with st.form("prediction_form"):
    calories = st.number_input("Calories", min_value=0, step=1)
    protein = st.number_input("Protein", min_value=0, step=1)
    fat = st.number_input("Fat", min_value=0, step=1)
    carbs = st.number_input("Carbohydrates", min_value=0, step=1)
    sugar = st.number_input("Sugar", min_value=0, step=1)
    fiber = st.number_input("Fiber", min_value=0, step=1)
    sodium = st.number_input("Sodium", min_value=0, step=1)
    cholesterol = st.number_input("Cholesterol", min_value=0, step=1)
    glycemic_index = st.number_input("Glycemic_Index", min_value=0, step=1)
    water_content = st.number_input("Water_Content", min_value=0, step=1)
    serving_size = st.number_input("Serving_Size", min_value=0, step=1)
    meal_type = st.selectbox("Meal Type", ["Breakfast", "Lunch", "Dinner", "Snack"])
    prep_method = st.selectbox("Preparation Method", ["Boiled", "Fried", "Baked", "Raw", "Grilled"])
    is_vegan = st.checkbox("Is Vegan?")
    is_gluten_free = st.checkbox("Is Gluten Free?")

    submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            input_df = pd.DataFrame([{
                "Calories": calories, "Protein": protein, "Fat": fat, "Carbohydrates": carbs,
                "Sugar": sugar, "Fiber": fiber, "Sodium": sodium, "Cholesterol": cholesterol,
                "Glycemic_Index": glycemic_index, "Water_Content": water_content,
                "Serving_Size": serving_size, "Is_Vegan": int(is_vegan), 
                "Is_Gluten_Free": int(is_gluten_free), "Meal_Type": meal_type, 
                "Preparation_Method": prep_method
            }])

            # Match preprocessing
            input_df = pd.get_dummies(input_df, columns=['Meal_Type', 'Preparation_Method'], drop_first=True)
            input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

            input_scaled = scaler.transform(input_df)
            input_pca = pca.transform(input_scaled)

            all_predictions = []
            model_list = results_df["Model"].tolist()

            for model_name in model_list:
                safe_name = re.sub(r'[^a-zA-Z0-9]', '_', model_name)
                model_url = MODEL_URLS.get(model_name)
                model_file = f"Models/{safe_name}.pkl"

                try:
                    clf = load_model_from_url(model_url, model_file)
                    pred_class = clf.predict(input_pca)[0]

                    confidence = None
                    pred_label = None
                    probs_dict = {}

                    if hasattr(clf, "predict_proba"):
                        proba = clf.predict_proba(input_pca)[0]
                        confidence = max(proba) * 100
                        class_labels = le.inverse_transform(range(len(proba)))
                        probs_dict = {cls: round(p * 100, 2) for cls, p in zip(class_labels, proba)}
                        pred_label = le.inverse_transform([pred_class])[0]
                    elif hasattr(clf, "decision_function"):
                        decision = clf.decision_function(input_pca)
                        if decision.ndim > 1:
                            score = decision[0]
                            max_idx = score.argmax()
                            confidence = (score[max_idx] / (1 + abs(score[max_idx]))) * 100
                        else:
                            confidence = (abs(decision[0]) / (1 + abs(decision[0]))) * 100
                        pred_label = le.inverse_transform([pred_class])[0]
                    else:
                        pred_label = le.inverse_transform([pred_class])[0]

                    all_predictions.append({
                        "Model": model_name,
                        "Prediction": pred_label,
                        "Confidence (%)": round(confidence, 2) if confidence is not None else None
                    })

                except Exception as e:
                    all_predictions.append({
                        "Model": model_name,
                        "Prediction": f"⚠️ Error ({type(e).__name__})",
                        "Confidence (%)": None
                    })

            pred_df = pd.DataFrame(all_predictions).set_index("Model")
            pred_df = pred_df.sort_values(by="Confidence (%)", ascending=False)
            st.subheader("📊 Predictions from All Models")
            st.dataframe(
                pred_df.style.format({"Confidence (%)": "{:.2f}%"}).background_gradient(
                    cmap="Greens", subset=["Confidence (%)"])
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
