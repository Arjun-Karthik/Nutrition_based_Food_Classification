# NutriClass: Food Classification Using Nutritional Data

## ⚙️ Workflow
1. Data Understanding
   
   - Loaded a tabular dataset containing food nutrition information.
   - Inspected:
     - Class distribution
     - Sample entries
     - Dataset size and balance
     - Noise and inconsistencies

2. Data Preprocessing
   
     - Handled missing values through imputation or row removal.
     - Removed duplicates to prevent skewed learning.
     - Detected and treated outliers in features like calories or sugar.
     - Normalized/standardized numerical features to ensure model stability.

3. Feature Engineering

     - Applied Label Encoding / One-Hot Encoding for categorical targets.
     - Explored PCA / feature selection techniques to reduce dimensionality.
     - Finalized a cleaned, well-prepared dataset for training.

4. Model Selection & Training
   
     - Trained and compared multiple models:
        - Logistic Regression
        - Decision Tree
        - Random Forest
        - K-Nearest Neighbors
        - Support Vector Machine
        - XGBoost
        - Gradient Boosting Classifier
     - Used cross-validation to ensure robustness.
  
5. Evaluation & Insights
   
      - Evaluated models based on:
        - Accuracy
        - Precision
        - Recall
        - F1-score
        - Confusion Matrix
      - Also compared:
        - Model performances side-by-side
        - Feature importance and contribution
        - Visualization of prediction behaviors

## ▶️ Running the App

Ensure Python 3.8+ is installed.

1. Clone the repo:
   
       https://github.com/Arjun-Karthik/Nutrition_based_Food_Classification
       cd Nutrition_based_Food_Classification

2.Install dependencies

       pip install -r requirements.txt

3. Run Streamlit app

       streamlit run app.py

## 🧩 Features

   - Multi-class classification using nutritional content
   - Automated preprocessing and encoding
   - Support for model comparison and performance benchmarking
   - Visual analytics for:
      - Class distribution
      - Feature importance
      - Confusion matrix, precision-recall, etc.
   - Educational insights into food and health using ML

## ✅ Requirements

   - streamlit
   - pandas
   - scikit-learn
   - xgboost
   - imblearn
   - joblib
   - matplotlib
   - plotly

Install all with:

       pip install -r requirements.txt

## 📸 Screenshots

### 📊 Model Performance

<img src="Screenshots/Model Metrics.png" width="800"/>

### 🔵 PCA Visualization

<img src="Screenshots/PCA Visualization.png" width="800"/>

## 🎥 Demo Video

   <a href="https://www.linkedin.com/posts/arjun-t-a51383200_nutriclass-food-classification-using-nutritional-activity-7358768128262066178-qE4L?utm_source=share&utm_medium=member_desktop&rcm=ACoAADNQBh0BQsEphYCjQb01l17Z8-pUyINZuxs">NutriClass: Food Classification Using Nutritional Data Demo Video</a>

## 📃 License

   This project is licensed under the MIT License – see the LICENSE file for details.
