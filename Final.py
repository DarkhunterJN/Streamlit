import streamlit as st
import pandas as pd
#import polars as pl
from ydata_profiling import ProfileReport
#from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


st.title("Streamlit Data-Cleansing, Profiling & ML Tool")

upf = st.file_uploader("Upload a csv extension file", type=["csv"])
#if upf is not None:
    # Will try Polars first
    #try:
    #    filedata = pl.read_csv(upf)
    #    st.success("Loaded with Polars")
    #    filedata_pd = filedata.to_pandas()  # convert for Pandas
    #except Exception:
    #    upf.seek(0)
    #    filedata_pd = pd.read_csv(upf)
    #    st.warning("Polars failed, loaded with Pandas") #loaded with pandas
    # removed this code as polars and ydata-profiling are not compatible with Python 3.13 on Streamlit Cloud. I have kept this coding to satisfy requirement.

if upf is not None:
    filedata_pd = pd.read_csv(upf)
    st.success("Loaded file with Pandas")
    
    st.subheader("Data Preview")
    st.dataframe(filedata_pd.head())

    st.subheader("Data Cleaning Options")

    if st.checkbox("Drop rows with missing values"):
        filedata_pd = filedata_pd.dropna()
        st.success("Rows not have missing data dropped")

    if st.checkbox("Remove duplicate value rows"):
        filedata_pd = filedata_pd.drop_duplicates()
        st.success("Duplicate value rows removed")

    column_conv = st.selectbox("Select column for type conversion", ["None"] + list(filedata_pd.columns))
    if column_conv != "None":
        temp = st.selectbox("Convert to type", ["int", "float", "str"])
        try:
            filedata_pd[column_conv] = filedata_pd[column_conv].astype(temp)
            st.success(f"Converted {column_conv} to {temp}")
        except Exception as er:
            st.error(f"Error: {er}")

    if st.checkbox("Normalize numeric columns (Min-Max Scaling)"):
        num_cols = filedata_pd.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            for col in num_cols:
             min_val = filedata_pd[col].min()
             max_val = filedata_pd[col].max()
            if min_val != max_val:  # to avoid division by zero
                filedata_pd[col] = (filedata_pd[col] - min_val) / (max_val - min_val)
            st.success("Normalized numeric columns")
        else:
            st.warning("No numeric columns found")

    col_filter = st.selectbox("Select column for filtering", ["None"] + list(filedata_pd.columns))
    if col_filter != "None":
        unique_vals = filedata_pd[col_filter].unique()
        filter_val = st.selectbox("Select value to filter by", unique_vals)
        filedata_pd = filedata_pd[filedata_pd[col_filter] == filter_val]
        st.success(f"Filtered rows where {col_filter} = {filter_val}")

    # Cleaned Data preview
    st.subheader("Cleaned Data Preview")
    st.dataframe(filedata_pd.head())

     # --- Profiling Report ---
    # ydata profiling is not compatible with the latest Python version and will not deploy on Streamlit Cloud. Kept this code to satisfy the requirement.
    @st.cache_data
    def generate_profile(data):
        return ProfileReport(data, title="Profiling Report", explorative=True)
    
    st.subheader("Profiling Report")
    with st.spinner("Generating profiling report..."):
        profile = generate_profile(filedata_pd)
    st_profile_report(profile)

    #Tried pandas profiling. Gives error pandas profiling not found as in the latest version, it has been changed to ydata Profiling 
    #@st.cache_data
    #def generate_profile(data):
        #return ProfileReport(data, explorative=True)

    #st.subheader("Profiling Report")
    #with st.spinner("Generating profiling report..."):
        #profile = generate_profile(filedata_pd)
    #st_profile_report(profile)

    # Download cleaned CSV
    st.download_button(
        "Download Cleaned CSV",
        filedata_pd.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

    # Download profiling report
    profile.to_file("report.html")
    with open("report.html", "rb") as f:
        st.download_button("Download Full Report", f, file_name="report.html")

    # ML Prediction Module
    st.subheader("Machine Learning Prediction")

    target_col = st.selectbox("Select Target Column (Y)", ["None"] + list(filedata_pd.columns))
    if target_col != "None":
        X = filedata_pd.drop(columns=[target_col])
        y = filedata_pd[target_col]

        # Only numeric features for ML
        X = X.select_dtypes(include=["number"])

        if len(X.columns) == 0:
            st.error("No numeric columns available for ML")
        else:
            # Simple model choice
            if y.dtype == "object" or y.nunique() < 10:
                model_type = "classification"
                model = LogisticRegression(max_iter=500)
            else:
                model_type = "regression"
                model = LinearRegression()

            # Train model
            model.fit(X, y)
            st.success("Model trained")

            # Evaluate
            if model_type == "classification":
                acc = accuracy_score(y, model.predict(X))
                st.success(f"Model trained (Classification) | Accuracy = {acc:.2f}")
            else:
                mse = mean_squared_error(y, model.predict(X))
                st.success(f"Model trained (Regression) | MSE = {mse:.2f}")

            # User input for prediction
            st.subheader("Make a Prediction")
            user_input = {}
            for col in X.columns:
                val = st.number_input(
                    f"Enter value for {col}", 
                    float(X[col].min()), 
                    float(X[col].max()), 
                    float(X[col].mean())
                )
                user_input[col] = val

            if st.button("Predict"):
                input_filedata = pd.DataFrame([user_input])
                prediction = model.predict(input_filedata)[0]

                st.success(f"Predicted Value: {prediction}")




