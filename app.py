import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="NutriClass Visualizer", 
                   page_icon='🍱',
                   layout="wide"
                )

st.markdown("""<h1 style = 'text-align:center;'>🍵 NutriClass - Food Classification Using Nutritional Data</h1>
                <hr style='border-top: 3px solid #bbb;'>""", 
                unsafe_allow_html=True
            )

def load_data():
    df = pd.read_csv("synthetic_food_dataset_imbalanced.csv")
    return df

df_raw = load_data()

st.markdown("<h2 style = text-align:center;>📋 Dataset Diagnostics</h2>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    # Original Shape
    original_shape = df_raw.shape
    st.markdown(f"<h5 style = text-align:center;>🔢 Original Shape: {original_shape}</h5>", unsafe_allow_html=True)

with col2:
    # Count Duplicate rows
    duplicate_count = df_raw.duplicated().sum()
    st.markdown(f"<h5 style = text-align:center;>♻️ Duplicate Rows: {duplicate_count}</h5>", unsafe_allow_html=True)

with col3:
    # Remove duplicates
    df_raw = df_raw.drop_duplicates()
    post_dup_shape = df_raw.shape
    st.markdown(f"<h5 style = text-align:center;>🧹 Shape After Removing Duplicates: {post_dup_shape}</h5>", unsafe_allow_html=True)

st.markdown("""<hr style = 'width:95%; display:block; margin-left:auto; margin-right:auto; border-top: 3px solid #bbb;'>""", unsafe_allow_html=True)

# Count Missing Values
missing_values = df_raw.isnull().sum()
st.subheader("❗Missing Values per Column")
missing_df = missing_values[missing_values > 0].reset_index()
missing_df.columns = ['Column', 'No. of missing values']
missing_df.index = range(1, len(missing_df) + 1)
st.dataframe(missing_df)

# Show Sample Data
st.subheader("📊 Data Overview")
df_raw.index = range(1, len(df_raw) + 1)
st.dataframe(df_raw.head(10))

# Load model results
results_df = pd.read_csv("Models/model_results.csv")
results_df[['Accuracy', 'Precision', 'Recall', 'F1 Score']] *= 100
results_df = results_df.round(2)

# Model Comparison Table
st.subheader("📋 Model Comparison Table")
st.dataframe(
    results_df.set_index("Model")
        .style.format("{:.1f}%")
        .background_gradient(cmap="Blues", axis=0)
)

# Model Metrics Comparison - Bar Chart
fig_bar = px.bar(results_df.melt(id_vars='Model'), x='Model', y='value', color='variable',
                 barmode='group', title='📈 Model Metrics Comparison',
                 labels={'value': 'Percentage (%)', 'variable': 'Metric'})
st.plotly_chart(fig_bar, use_container_width=True)

# Confusion Matrix
st.header("🧩 Confusion Matrix")
selected_model = st.selectbox("Select Model to View Confusion Matrix", results_df['Model'])
import re
safe_model_name = re.sub(r'[^a-zA-Z0-9]', '_', selected_model)
if selected_model.strip() == "K-Nearest Neighbors":
    model_file = "Models/model_K-Nearest_Neighbors.pkl"
else:
    model_file = f"Models/model_{safe_model_name}.pkl"


try:
    model = joblib.load(model_file)
    scaler_cm = joblib.load("Models/scaler.pkl")
    pca_cm = joblib.load("Models/pca.pkl")
    le_cm = joblib.load("Models/label_encoder.pkl")

    # Load and preprocess dataset
    df_raw = pd.read_csv("synthetic_food_dataset_imbalanced.csv").dropna().drop_duplicates()
    df_raw['Is_Vegan'] = df_raw['Is_Vegan'].astype(int)
    df_raw['Is_Gluten_Free'] = df_raw['Is_Gluten_Free'].astype(int)
    df_raw = pd.get_dummies(df_raw, columns=['Meal_Type', 'Preparation_Method'], drop_first=True)
    df_raw = df_raw.drop(columns=['Food_Name'])

    # Align feature columns to training set
    X = df_raw.reindex(columns=scaler_cm.feature_names_in_, fill_value=0)
    X_scaled = scaler_cm.transform(X)
    X_pca = pca_cm.transform(X_scaled)

    # Load target
    df_y = pd.read_csv("synthetic_food_dataset_imbalanced.csv").dropna().drop_duplicates()
    y = le_cm.transform(df_y['Food_Name'])

    # Predict
    y_pred = model.predict(X_pca)

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=le_cm.classes_,
        y=le_cm.classes_,
        title=f"Confusion Matrix - {selected_model}"
    )
    st.plotly_chart(fig_cm, use_container_width=True)

except FileNotFoundError:
    st.error(f"Model file not found for {selected_model}. Make sure model_{selected_model.replace(' ', '_')}.pkl exists.")