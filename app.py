# ==========================================
# 1️⃣ Import Libraries
# ==========================================
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="DS Salary Predictor")

st.title("💼 Data Science Salary Classification")

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
# 4️⃣ Select ONLY 5 Features (Important Fix)
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
# 9️⃣ Prediction Section
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
  
    # Show Result
    if prediction == "High":
        st.success("💰 Predicted Salary: HIGH")
    else:
        st.warning("📉 Predicted Salary: LOW")
