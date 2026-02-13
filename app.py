import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="DS Salary Predictor")

st.title("💼 Data Science Salary Classification")

@st.cache_data
def load_data():
    return pd.read_csv("ds_salaries.csv")

df = load_data()

# Create Binary Target
median_salary = df["salary_in_usd"].median()
df["salary_binary"] = df["salary_in_usd"].apply(
    lambda x: "High" if x >= median_salary else "Low"
)

y = df["salary_binary"]

X = df.drop(columns=[
    "salary",
    "salary_currency",
    "salary_in_usd",
    "salary_binary"
])

X = pd.get_dummies(X, drop_first=True)

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

rf_model.fit(X_train, y_train)

st.subheader("📊 Model Accuracy")
st.write("Random Forest Accuracy:",
         round(accuracy_score(y_test, rf_model.predict(X_test)), 3))

# ---------------- Prediction Section ----------------

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

if st.sidebar.button("Predict Salary"):

    input_dict = {
        "experience_level": experience_level,
        "employment_type": employment_type,
        "job_title": job_title,
        "company_location": company_location,
        "company_size": company_size
    }

    input_df = pd.DataFrame([input_dict])
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    prediction = rf_model.predict(input_encoded)[0]

    if prediction == "High":
        st.success("💰 Predicted Salary: HIGH")
    else:
        st.warning("📉 Predicted Salary: LOW")
