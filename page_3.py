# import streamlit as st
# import pandas as pd
# from model_streamlit import preprocess_data, train_model, evaluate_model, predict

# st.title("Upload your sample")
# st.write("In this page you can upload your samples data.")

# st.subheader("Upload the excel file with the sample data.")
# st.write("Upload an Excel file containing your data. This file will be used for evaluation in the next page.")

# # File uploader widget
# uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

# # Check if a file is uploaded
# if uploaded_file is not None:
#     # Read the uploaded Excel file
#     df = pd.read_excel(uploaded_file, sheet_name=0)
    
#     # Display the DataFrame
#     st.write("Data preview:")
#     st.dataframe(df)
    
#     # Get the number of rows and columns
#     num_rows, num_columns = df.shape
    
#     # Display the number of rows and columns
#     st.write(f"The dataset contains {num_rows} rows and {num_columns} columns.")

#     # Save the uploaded file to session state for use in the next page
#     st.session_state.df = df

# # Load raw data for model training
# raw_data = pd.read_excel("raw_data.xlsx")

# if 'df' in st.session_state:
#     df = st.session_state.df
    
#     # Preprocess the raw and evaluation data
#     new_raw_data = preprocess_data(raw_data)
#     #new_evaluation_data = preprocess_data(df)

#     # Ensure consistency in dummy variable columns
#     if 'Gender' in df.columns:
#         df = pd.get_dummies(df, columns=['Gender'])
#     else:
#         # Handle the case where the expected 'Gender' column is missing
#         st.write("The 'Gender' column is missing from the uploaded data.")
#         df = pd.get_dummies(df)  # Dummy encoding other categorical columns if needed

#     # Ensure consistency in dummy variable columns
#     df = pd.get_dummies(df, columns=['Gender'])
#     missing_cols = set(new_raw_data.columns) - set(df.columns)
#     for col in missing_cols:
#         df[col] = 0
#     df = df[new_raw_data.columns.drop('Disease')]
    
#     # Prepare the training data
#     X = new_raw_data.drop(['ID', 'Disease'], axis=1).values
#     y = new_raw_data['Disease']

#     # Train the model
#     clf = train_model(X, y)

#     # Evaluate the model
#     accuracy, log_loss_value, precision, recall, f1, report = evaluate_model(clf, X, y)
#     st.write(f'Accuracy: {accuracy}')
#     st.write(f'Log Loss: {log_loss_value}')
#     st.write(f'Precision: {precision}')
#     st.write(f'Recall: {recall}')
#     st.write(f'F1 Score: {f1}')
#     st.text(report)

#     # Prepare the test dataset
#     new_evaluation_data = df
#     new_evaluation_data = pd.get_dummies(new_evaluation_data, columns = ['Gender'])
#     X_test = new_evaluation_data.drop(['ID'], axis=1).values

#     # Predict the Disease column in the evaluation dataset
#     y_test_pred = predict(clf, X_test)
#     df['Predicted_Disease'] = y_test_pred

#     # Show the predicted results
#     st.write("Predicted results:")
#     st.dataframe(df)
#     # Optionally, save the predicted results
#     # df.to_excel('predicted_evaluation_data.xlsx', index=False)

import streamlit as st
import pandas as pd
from model_streamlit import train_model, evaluate_model, predict

# Define preprocess_data to standardize columns and apply get_dummies
def preprocess_data(raw_data):
    # Standardize column names to lowercase and strip whitespace
    raw_data.columns = raw_data.columns.str.strip().str.lower()
    
    # Check for 'gender' in columns and apply pd.get_dummies accordingly
    if 'gender' in raw_data.columns:
        # Convert 'gender' to dummy variables
        processed_data = pd.get_dummies(raw_data, columns=['gender'])
    else:
        # If 'gender' column is missing, apply get_dummies on other columns only
        processed_data = pd.get_dummies(raw_data)
    
    return processed_data

st.title("Upload your sample")
st.write("In this page you can upload your samples data.")

st.subheader("Upload the excel file with the sample data.")
st.write("Upload an Excel file containing your data. The data in the file will be analyzed with our machine learning model. At the end of the analysis, each sample will be predicted to be **diseased** or **healthy**")

st.write("The excel file shold contain 32 columns, each row should represent one sample. You can view and download an example file here:")

# Update the path to point to your Excel file
excel_file = "test_data.xlsx"

# Read the first sheet of the Excel file
df = pd.read_excel(excel_file, sheet_name=0)  

# Display the DataFrame
st.dataframe(df)  

# File uploader widget
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

# Read the example Excel file in binary mode
with open("C:\Users\Utente\Downloads\ML lipedema\APP", "rb") as file:
    example_file_binary = file.read()

# Add a download button to download the existing file
st.download_button(
    label="Download Example Excel File",
    data=example_file_binary,
    file_name="test_data2.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Check if a file is uploaded
if uploaded_file is not None:
    # Read the uploaded Excel file
    df = pd.read_excel(uploaded_file, sheet_name=0)
    
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()
    
    # Display the DataFrame
    st.write("Data preview:")
    st.dataframe(df)
    
    # Get the number of rows and columns
    num_rows, num_columns = df.shape
    
    # Display the number of rows and columns
    st.write(f"The dataset contains {num_rows} rows and {num_columns} columns.")

    # Save the uploaded file to session state for use in the next page
    st.session_state.df = df

# Load raw data for model training
raw_data = pd.read_excel("raw_data.xlsx")

# Standardize column names for raw data
raw_data.columns = raw_data.columns.str.strip().str.lower()

if 'df' in st.session_state:
    df = st.session_state.df
    
    # Preprocess the raw and evaluation data
    new_raw_data = preprocess_data(raw_data)
    
    # Ensure consistency in dummy variable columns
    if 'gender' in df.columns:
        df = pd.get_dummies(df, columns=['gender'])
    else:
        # Handle the case where the expected 'gender' column is missing
        st.write("The 'gender' column is missing from the uploaded data.")
        df = pd.get_dummies(df)  # Dummy encoding other categorical columns if needed

    # Ensure consistency in dummy variable columns
    missing_cols = set(new_raw_data.columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    df = df[new_raw_data.columns.drop('disease')]
    
    # Prepare the training data
    X = new_raw_data.drop(['id', 'disease'], axis=1).values
    y = new_raw_data['disease']

    # Train the model
    clf = train_model(X, y)

    # Evaluate the model
    accuracy, log_loss_value, precision, recall, f1, report = evaluate_model(clf, X, y)
    st.write(f'Accuracy: {accuracy}')
    st.write(f'Log Loss: {log_loss_value}')
    st.write(f'Precision: {precision}')
    st.write(f'Recall: {recall}')
    st.write(f'F1 Score: {f1}')
    st.text(report)

    # Prepare the test dataset
    new_evaluation_data = df
    X_test = new_evaluation_data.drop(['id'], axis=1).values

    # Predict the Disease column in the evaluation dataset
    y_test_pred = predict(clf, X_test)
    df['predicted_disease'] = y_test_pred

    # Show the predicted results
    st.write("Predicted results:")

    for index, prediction in enumerate(df['predicted_disease']):
    # Retrieve the ID directly from the 'id' column in the DataFrame without storing it in a conflicting variable
        if prediction == 1:
            st.write(f"Sample ID {df['id'].iloc[index]}: The adipose tissue is predicted to be **diseased**.")
        else:
            st.write(f"Sample ID {df['id'].iloc[index]}: The adipose tissue is predicted to be **healthy**.")
    
    # Optionally, save the predicted results
    # df.to_excel('predicted_evaluation_data.xlsx', index=False)

    # Add the "Download Report" button at the end of the page
    if st.button("Click here to download your report"):
        st.write("Work in progress...")