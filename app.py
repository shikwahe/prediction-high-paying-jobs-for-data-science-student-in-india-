import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

st.set_page_config(page_title="DS Salary Predictor")

st.title("💼 Data Science Salary Classification App")

@st.cache_data
def load_data():
    return pd.read_csv("ds_salaries.csv")

df = load_data()

st.write("Dataset Shape:", df.shape)
st.dataframe(df.head())

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

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)

# Random Forest
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Accuracy Display
st.subheader("📊 Model Accuracy Comparison")

st.write("Logistic Regression Accuracy:",
         round(accuracy_score(y_test, y_pred_log), 3))

st.write("Random Forest Accuracy:",
         round(accuracy_score(y_test, y_pred_rf), 3))

# Confusion Matrix Plot
st.subheader("📈 Random Forest Confusion Matrix")

cm_rf = confusion_matrix(y_test, y_pred_rf)

fig, ax = plt.subplots()
sns.heatmap(cm_rf, annot=True, fmt='d', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)
