# ==========================================
# 💼 Data Science Salary Dashboard (Premium UI)
# ==========================================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="DS Salary Predictor", layout="wide")

# ==========================================
# 🎨 Custom Styling
# ==========================================
st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
.big-font {
    font-size:22px !important;
    font-weight:600;
}
.card {
    padding:20px;
    border-radius:15px;
    background-color:white;
    box-shadow:0 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.title("💼 Data Science Salary Classification Dashboard")
st.markdown("### Predict whether a Data Science job belongs to High or Low Salary category")

# ==========================================
# 📂 Load Dataset
# ==========================================
@st.cache_data
def load_data():
    return pd.read_csv("ds_salaries.csv")

df = load_data()

# Create binary target
median_salary = df["salary_in_usd"].median()
df["salary_binary"] = df["salary_in_usd"].apply(
    lambda x: "High" if x >= median_salary else "Low"
)

# ==========================================
# 🧠 Model Preparation
# ==========================================
feature_cols = [
    "experience_level",
    "employment_type",
    "job_title",
    "company_location",
    "company_size"
]

X = pd.get_dummies(df[feature_cols], drop_first=True)
y = df["salary_binary"]

model_columns = X.columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

rf_model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, rf_model.predict(X_test))

# ==========================================
# 📊 Metrics Cards
# ==========================================
col1, col2, col3 = st.columns(3)

col1.metric("📈 Model Accuracy", f"{round(accuracy*100,2)} %")
col2.metric("📊 Total Records", len(df))
col3.metric("💰 Median Salary", f"${int(median_salary)}")

st.markdown("---")

# ==========================================
# 📊 Visual Dashboard
# ==========================================
colA, colB = st.columns(2)

# Salary Distribution
with colA:
    st.markdown("### 📈 Salary Distribution")
    salary_counts = df["salary_binary"].value_counts()

    fig1, ax1 = plt.subplots(figsize=(5,4))
    salary_counts.plot(kind='bar', ax=ax1)
    ax1.set_title("High vs Low Salary")
    ax1.set_xlabel("")
    ax1.set_ylabel("Count")
    plt.xticks(rotation=0)

    st.pyplot(fig1)

# Feature Importance
with colB:
    st.markdown("### 🌟 Top 10 Important Features")
    importances = rf_model.feature_importances_

    fi_df = pd.DataFrame({
        "Feature": model_columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    fig2, ax2 = plt.subplots(figsize=(5,4))
    fi_df.head(10).plot(
        kind='barh',
        x="Feature",
        y="Importance",
        ax=ax2
    )
    ax2.invert_yaxis()
    ax2.set_title("Feature Importance")
    st.pyplot(fig2)

st.markdown("---")

# ==========================================
# 🔮 Prediction Section
# ==========================================
st.markdown("## 🔮 Predict Salary Category")

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

if st.sidebar.button("🚀 Predict Salary"):

    input_data = pd.DataFrame({
        "experience_level": [experience_level],
        "employment_type": [employment_type],
        "job_title": [job_title],
        "company_location": [company_location],
        "company_size": [company_size]
    })

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    prediction = rf_model.predict(input_encoded)[0]
    confidence = rf_model.predict_proba(input_encoded).max()

    st.markdown("### 🎯 Prediction Result")

    if prediction == "High":
        st.success(f"💰 HIGH Salary Job")
    else:
        st.warning(f"📉 LOW Salary Job")

    st.info(f"Model Confidence: {round(confidence*100,2)} %")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")
