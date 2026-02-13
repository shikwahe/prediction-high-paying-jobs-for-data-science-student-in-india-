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

    # Apply same preprocessing
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    prediction = rf_model.predict(input_encoded)[0]

    if prediction == "High":
        st.success("💰 Predicted Salary Category: HIGH")
    else:
        st.warning("📉 Predicted Salary Category: LOW")
