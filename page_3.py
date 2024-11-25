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
df = pd.read_excel(excel_file, sheet_name=0)  # Use sheet_name='Sheet1' if you know the sheet name

# Display the DataFrame
st.dataframe(df)  # Use st.table(df) if you prefer a static table

st.write("You can download an example dataframe in xlsx format here:")

# Filepath to the Excel file in the repository
file_path = "test_data2.xlsx"

# Check if the file exists in the repository
try:
    # Read the file into memory
    with open(file_path, "rb") as file:
        file_data = file.read()

    # Create a download button
    st.download_button(
        label="Download test xlsx file",
        data=file_data,
        file_name="test_data2.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
except FileNotFoundError:
    st.error("The file test_data2.xlsx was not found in the repository.")

# Define required columns for validation
REQUIRED_COLUMNS = ['id', 'gender']  # Add other required columns as needed

# File uploader widget
uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")

def validate_excel_file(df):
    """Validates the uploaded Excel file."""
    # Check if required columns are present
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        st.error(f"The uploaded file is missing the following required columns: {', '.join(missing_columns)}")
        return False

    # Ensure no missing values in required columns
    for col in REQUIRED_COLUMNS:
        if df[col].isnull().any():
            st.error(f"The column '{col}' contains missing values. Please clean your data and try again.")
            return False

    # Additional checks (e.g., data types) can be added here
    if not pd.api.types.is_numeric_dtype(df['id']):
        st.error("The 'id' column must contain numeric values.")
        return False
    
    return True


# Check if a file is uploaded
if uploaded_file is not None:
    # Read the uploaded Excel file
    df = pd.read_excel(uploaded_file, sheet_name=0)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # Validate the uploaded file
    if not validate_excel_file(df):
        st.stop()  # Halt execution if validation fails
    
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
