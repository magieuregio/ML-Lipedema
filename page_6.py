import streamlit as st



st.title("Download an example")

# Filepath to the Excel file in the repository
file_path = "test_data2.xlsx"

# Check if the file exists in the repository
try:
    # Read the file into memory
    with open(file_path, "rb") as file:
        file_data = file.read()

    # Create a download button
    st.download_button(
        label="Download test_data2.xlsx",
        data=file_data,
        file_name="test_data2.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
except FileNotFoundError:
    st.error("The file test_data2.xlsx was not found in the repository.")
