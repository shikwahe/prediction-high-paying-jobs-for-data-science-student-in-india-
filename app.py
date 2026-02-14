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

st.title("💼 Data Science Salary Classification App")

# ==========================================
# 2️⃣ Load Dataset
# ==========================================
@st.cache_data
def load_data():
    return pd.read_csv("ds_salaries.csv")

df = load_data()

# ==========================================
# 3️⃣ Create Binary Target (High / Low)
# ==========================================
median_salary = df["salary_in_usd"].median()

df["salary_binary"] = df["salary_in_usd"].apply(
    lambda x: "High" if x >= median_salary else "Low"
)

# ==========================================
# 📊 High vs Low Salary Distribution
# ==========================================
st.subheader("📈 Salary Category Distribution")

salary_counts = df["salary_binary"].value_counts()

fig_dist, ax_dist = plt.subplots()
salary_counts.plot(kind='bar', ax=ax_dist)
ax_dist.set_xlabel("Salary Category")
ax_dist.set_ylabel("Count")
ax_dist.set_title("High vs Low Salary Distribution")
st.pyplot(fig_dist)

# ==========================================
# 4️⃣ Select 5 Features
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

# Save column names for prediction alignment
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
# 7️⃣ Train Random Forest Model
# ==========================================
rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

rf_model.fit(X_train, y_train)

# ==========================================
# 8️⃣ Show Model Accuracy
# ==========================================
st.subheader("📊 Model Accuracy")

accuracy = accuracy_score(y_test, rf_model.predict(X_test))
st.write("Random Forest Accuracy:", round(accuracy, 3))

# ==========================================
# 🌟 Feature Importance
# ==========================================
st.subheader("🌟 Feature Importance (Top 15)")

importances = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": model_columns,
    "Importance": importances
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

st.dataframe(feature_importance_df.head(15))

# ==========================================
# 📊 Feature Importance Visualization
# ==========================================
fig_imp, ax_imp = plt.subplots(figsize=(8,6))

feature_importance_df.head(15).plot(
    kind='barh',
    x="Feature",
    y="Importance",
    ax=ax_imp
)

ax_imp.invert_yaxis()
ax_imp.set_title("Top 15 Feature Importances")
ax_imp.set_xlabel("Importance Score")

st.pyplot(fig_imp)

# ==========================================
# 🔮 Prediction Section
# ==========================================
st.subheader("🔮 Predict Salary Category")

st.sidebar.header("Enter Job Details")

experience_level = st.sidebar.selectbox(
    "Experience Level",
    df["experience_level"].unique()
)

employment_type = st.sidebar.selectbox(
    "Employment Type",
    df["employment_type"].unique()
)

job_title = st.sidebar.selectbox(
    "Job Title",
    df["job_title"].unique()
)

company_location = st.sidebar.selectbox(
    "Company Location",
    df["company_location"].unique()
)

company_size = st.sidebar.selectbox(
    "Company Size",
    df["company_size"].unique()
)

# ==========================================
# 🔮 Make Prediction
# ==========================================
if st.sidebar.button("Predict Salary"):

    input_dict = {
        "experience_level": experience_level,
        "employment_type": employment_type,
        "job_title": job_title,
        "company_location": company_location,
        "company_size": company_size
    }

    input_df = pd.DataFrame([input_dict])

    # One-hot encode input
    input_encoded = pd.get_dummies(input_df)

    # Align with training columns
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = rf_model.predict(input_encoded)[0]

    if prediction == "High":
        st.success("💰 Predicted Salary: HIGH")
    else:
        st.warning("📉 Predicted Salary: LOW")

# ==========================================
# 📌 Insights Section
# ==========================================
st.markdown("""
### 📌 Key Insights:
- Experience level strongly impacts salary classification.
- Certain job titles are associated with higher salary groups.
- Company size and location influence salary prediction.
""")
