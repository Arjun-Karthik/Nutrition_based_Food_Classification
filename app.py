import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import pickle
import os
import re

# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(page_title="NutriClass Visualizer",
                   page_icon='🍱',
                   layout="wide")

st.markdown("""<h1 style='text-align:center;'>🍵 NutriClass - Food Classification Using Nutritional Data</h1>
               <hr style='border-top: 3px solid #bbb;'>""", unsafe_allow_html=True)

# -------------------- PATHS --------------------
MODEL_DIR = "Models"
DATA_FILE = "synthetic_food_dataset_imbalanced.csv"
RESULTS_FILE = "model_results.csv"

# -------------------- UTILITY FUNCTIONS --------------------
@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

@st.cache_resource
def load_model(file_name):
    path = os.path.join(MODEL_DIR, file_name)
    if not os.path.exists(path):
        st.error(f"Model file not found: {file_name}")
        return None
    try:
        # Prefer joblib for sklearn models, fallback to pickle
        try:
            return joblib.load(path)
        except Exception:
            with open(path, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load {file_name}: {type(e).__name__} - {e}")
        return None

def get_classifier(model_name):
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', model_name)
    model_file = "Models/K-Nearest_Neighbors.pkl" if model_name.strip() == "K-Nearest Neighbors" else f"Models/{safe_name}.pkl"
    return load_model(model_file)

# -------------------- LOAD DATA --------------------
df_raw = load_data(DATA_FILE)

# -------------------- DATASET DIAGNOSTICS --------------------
st.markdown("<h2 style=text-align:center;>📋 Dataset Diagnostics</h2>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<h5 style=text-align:center;>🔢 Original Shape: {df_raw.shape}</h5>", unsafe_allow_html=True)
with col2:
    duplicate_count = df_raw.duplicated().sum()
    st.markdown(f"<h5 style=text-align:center;>♻️ Duplicate Rows: {duplicate_count}</h5>", unsafe_allow_html=True)
with col3:
    df_raw = df_raw.drop_duplicates()
    st.markdown(f"<h5 style=text-align:center;>🧹 Shape After Removing Duplicates: {df_raw.shape}</h5>", unsafe_allow_html=True)
st.markdown("<hr style='width:95%; border-top: 3px solid #bbb;'>", unsafe_allow_html=True)

# -------------------- MISSING VALUES --------------------
missing_values = df_raw.isnull().sum()
st.subheader("❗ Missing Values per Column")
missing_df = missing_values[missing_values > 0].reset_index()
missing_df.columns = ['Column', 'No. of Missing Values']
missing_df.index = range(1, len(missing_df)+1)
st.dataframe(missing_df)

# -------------------- SAMPLE DATA --------------------
st.subheader("📊 Data Overview")
df_raw.index = range(1, len(df_raw)+1)
st.dataframe(df_raw.head(10))

# -------------------- EDA --------------------
st.header("🔍 Exploratory Data Analysis (EDA)")
numeric_cols = df_raw.select_dtypes(include=['float64', 'int64']).columns
corr = df_raw[numeric_cols].corr()
fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="📌 Correlation Heatmap")
st.plotly_chart(fig_corr, use_container_width=True)

# Class distribution before & after SMOTE
df_smote = df_raw.dropna()
X = df_smote.drop(columns=['Food_Name'])
y = df_smote['Food_Name']
X_encoded = pd.get_dummies(X, drop_first=True)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_encoded, y)

fig_before = px.bar(y.value_counts().reset_index(name="count"), x="Food_Name", y="count",
                    labels={"Food_Name":"Food Name", "count":"Count"}, title="🍽️ Class Distribution (Imbalanced)")
fig_after = px.bar(y_resampled.value_counts().reset_index(name="count"), x="Food_Name", y="count",
                   labels={"Food_Name":"Food Name", "count":"Count"}, title="🍽️ Class Distribution (Balanced)")

col1, col2 = st.columns(2)
col1.plotly_chart(fig_before, use_container_width=True)
col2.plotly_chart(fig_after, use_container_width=True)

# -------------------- Boxplot --------------------
fig_box = px.box(df_raw, x="Meal_Type", y="Protein", title="🥩 Protein Distribution by Meal Type")
st.plotly_chart(fig_box, use_container_width=True)

# -------------------- MODEL RESULTS --------------------
results_df = load_data(RESULTS_FILE)
results_df[['Accuracy','Precision','Recall','F1 Score']] *= 100
results_df = results_df.round(2)

st.subheader("📋 Model Comparison Table")
sorted_results = results_df.sort_values(by="Accuracy", ascending=False)
st.dataframe(
    sorted_results.set_index("Model")
        .style.format("{:.1f}%")
        .background_gradient(cmap="Blues", axis=0)
)
fig_bar = px.bar(sorted_results.melt(id_vars='Model'), x='Model', y='value', color='variable',
                 barmode='group', title='📈 Model Metrics Comparison',
                 labels={'value':'Percentage (%)','variable':'Metric'})
st.plotly_chart(fig_bar, use_container_width=True)

# -------------------- PCA --------------------
df_example = df_raw.dropna().drop_duplicates()
df_example = pd.get_dummies(df_example, columns=['Meal_Type','Preparation_Method'], drop_first=True)
df_example = df_example.drop(['Food_Name'], axis=1)
df_example['Is_Vegan'] = df_example['Is_Vegan'].astype(int)
df_example['Is_Gluten_Free'] = df_example['Is_Gluten_Free'].astype(int)

scaler = load_model("scaler.pkl")
pca = load_model("pca.pkl")
le = load_model("label_encoder.pkl")

X_scaled = scaler.transform(df_example)
X_pca = pca.transform(X_scaled)
pca_plot_df = pd.DataFrame(X_pca[:, :2], columns=['PC1', 'PC2'])
pca_plot_df['Label'] = le.transform(df_raw.dropna().drop_duplicates()['Food_Name'])

fig_pca = px.scatter(pca_plot_df, x='PC1', y='PC2', color=pca_plot_df['Label'].astype(str),
                     title="🔵 PCA - Nutritional Feature Reduction (2D)")
st.plotly_chart(fig_pca, use_container_width=True)

# -------------------- CONFUSION MATRIX --------------------
st.header("🧩 Confusion Matrix")
selected_model = st.selectbox("Select Model to View Confusion Matrix", results_df['Model'])

clf = get_classifier(selected_model)
if clf:
    df_y = df_raw.dropna().drop_duplicates()
    df_y['Is_Vegan'] = df_y['Is_Vegan'].astype(int)
    df_y['Is_Gluten_Free'] = df_y['Is_Gluten_Free'].astype(int)
    df_y = pd.get_dummies(df_y, columns=['Meal_Type','Preparation_Method'], drop_first=True)
    X_cm = df_y.reindex(columns=scaler.feature_names_in_, fill_value=0)
    X_scaled_cm = scaler.transform(X_cm)
    X_pca_cm = pca.transform(X_scaled_cm)
    y_true = le.transform(df_raw.dropna().drop_duplicates()['Food_Name'])
    y_pred = clf.predict(X_pca_cm)

    cm = confusion_matrix(y_true, y_pred)
    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=le.classes_, y=le.classes_,
                       title=f"Confusion Matrix - {selected_model}")
    st.plotly_chart(fig_cm, use_container_width=True)

# -------------------- PREDICTION --------------------
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
        input_df = pd.DataFrame([{
            "Calories": calories, "Protein": protein, "Fat": fat, "Carbohydrates": carbs,
            "Sugar": sugar, "Fiber": fiber, "Sodium": sodium, "Cholesterol": cholesterol,
            "Glycemic_Index": glycemic_index, "Water_Content": water_content,
            "Serving_Size": serving_size, "Is_Vegan": int(is_vegan), 
            "Is_Gluten_Free": int(is_gluten_free), "Meal_Type": meal_type, 
            "Preparation_Method": prep_method
        }])
        input_df = pd.get_dummies(input_df, columns=['Meal_Type','Preparation_Method'], drop_first=True)
        input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)

        input_scaled = scaler.transform(input_df)
        input_pca = pca.transform(input_scaled)

        all_predictions = []
        for model_name in results_df['Model']:
            clf = get_classifier(model_name)
            if clf is None:
                all_predictions.append({"Model": model_name, "Prediction": "⚠️ Load Error", "Confidence (%)": None})
                continue

            pred_class = clf.predict(input_pca)[0]
            confidence = None
            pred_label = le.inverse_transform([pred_class])[0]

            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(input_pca)[0]
                confidence = proba[pred_class] * 100 if pred_class < len(proba) else None

            all_predictions.append({"Model": model_name, "Prediction": pred_label,
                                    "Confidence (%)": round(confidence,2) if confidence else None})

        pred_df = pd.DataFrame(all_predictions).set_index("Model")
        st.subheader("📊 Predictions from All Models")
        st.dataframe(pred_df.style.format({"Confidence (%)":"{:.2f}%"}).background_gradient(cmap="Greens"))

