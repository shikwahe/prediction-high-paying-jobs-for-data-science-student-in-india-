# ==========================================
# 1️⃣ Import Libraries
# ==========================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="DS Salary Predictor", layout="wide")

st.title("💼 Data Science Salary Classification Dashboard")

# ==========================================
# 2️⃣ Load Dataset
# ==========================================
@st.cache_data
def load_data():
    return pd.read_csv("ds_salaries.csv")

df = load_data()

# ==========================================
# 3️⃣ Create Binary Target
# ==========================================
median_salary = df["salary_in_usd"].median()

df["salary_binary"] = df["salary_in_usd"].apply(
    lambda x: "High" if x >= median_salary else "Low"
)

# ==========================================
# 4️⃣ Select Features
# ==========================================
feature_cols = [
    "experience_level",
    "employment_type",
    "job_title",
    "company_location",
    "company_size"
]

X = df[feature_cols]
y = df["salary_binary"]

# ==========================================
# 5️⃣ One-Hot Encoding
# ==========================================
X = pd.get_dummies(X, drop_first=True)
model_columns = X.columns

# ==========================================
# 6️⃣ Train-Test Split
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================
# 7️⃣ Train Random Forest
# ==========================================
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

rf_model.fit(X_train, y_train)

# ==========================================
# 8️⃣ Model Performance
# ==========================================
st.subheader("📊 Model Performance")

accuracy = accuracy_score(y_test, rf_model.predict(X_test))

colA, colB = st.columns(2)
colA.metric("Model Accuracy", f"{round(accuracy*100,2)} %")
colB.metric("Dataset Size", len(df))

# ==========================================
# 📊 Side-by-Side Visualizations
# ==========================================
st.subheader("📊 Salary Insights Dashboard")

col1, col2 = st.columns(2)

# -----------------------------
# 📈 Salary Distribution
# -----------------------------
with col1:
    st.markdown("### 📈 Salary Distribution")

    salary_counts = df["salary_binary"].value_counts()

    fig1, ax1 = plt.subplots(figsize=(4,3))
    salary_counts.plot(kind='bar', ax=ax1)
    ax1.set_xlabel("Category")
    ax1.set_ylabel("Count")
    ax1.set_title("High vs Low")
    plt.xticks(rotation=0)

    st.pyplot(fig1)

# -----------------------------
# 📊 Feature Importance
# -----------------------------
with col2:
    st.markdown("### 🌟 Top 10 Important Features")

    importances = rf_model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": model_columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig2, ax2 = plt.subplots(figsize=(4,3))

    feature_importance_df.head(10).plot(
        kind='barh',
        x="Feature",
        y="Importance",
        ax=ax2
    )

    ax2.invert_yaxis()
    ax2.set_title("Top 10 Features")
    ax2.set_xlabel("Score")

    st.pyplot(fig2)

# ==========================================
# 🔮 Prediction Section
# ==========================================
st.subheader("🔮 Predict Salary Category")

st.sidebar.header("Enter Job Details")

experience_level = st.sidebar.selectbox(
    "Experience Level",
    sorted(df["experience_level"].unique())
)

employment_type = st.sidebar.selectbox(
    "Employment Type",
    sorted(df["employment_type"].unique())
)

job_title = st.sidebar.selectbox(
    "Job Title",
    sorted(df["job_title"].unique())
)

company_location = st.sidebar.selectbox(
    "Company Location",
    sorted(df["company_location"].unique())
)

company_size = st.sidebar.selectbox(
    "Company Size",
    sorted(df["company_size"].unique())
)

# Predict button
predict_button = st.sidebar.button("Predict Salary")

if predict_button:

    input_data = pd.DataFrame({
        "experience_level": [experience_level],
        "employment_type": [employment_type],
        "job_title": [job_title],
        "company_location": [company_location],
        "company_size": [company_size]
    })

    # Encode
    input_encoded = pd.get_dummies(input_data)

    # Align with training columns
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = rf_model.predict(input_encoded)[0]
    confidence = rf_model.predict_proba(input_encoded).max()

    st.subheader("🎯 Prediction Result")

    if prediction == "High":
        st.success("💰 Predicted Salary: HIGH")
    else:
        st.warning("📉 Predicted Salary: LOW")

    st.info(f"Model Confidence: {round(confidence*100,2)} %")

# ==========================================
# 📌 Insights
# ==========================================
st.markdown("""
---
### 📌 Key Insights:
- Experience level significantly impacts salary.
- Certain job roles dominate high salary category.
- Company size and location influence prediction outcome.
""")
