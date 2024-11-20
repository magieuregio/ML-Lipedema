import streamlit as st
import pandas as pd
import pickle
from io import StringIO
import subprocess

st.title("The developed machine learning model")
st.subheader("In this page you can read a theoretical explanation of the machine learning algorithm employed.")

st.write("# Support Vector Classifier (SVC)")

st.write("""
**Description:** A Support Vector Classifier (SVC) is a supervised machine learning model that uses a hyperplane to separate data into different classes. It is a type of Support Vector Machine (SVM) specifically used for classification tasks.

**Strengths:**
- **Effective in high-dimensional spaces:** SVC works well when there are a large number of features.
- **Robust to overfitting:** Especially effective with a clear margin of separation and with fewer outliers.
- **Kernel trick:** Can efficiently handle non-linear classification using different kernel functions (e.g., polynomial, radial basis function).

**Weaknesses:**
- **Computationally intensive:** Training can be slow for large datasets due to quadratic programming problems.
- **Choice of kernel and parameters:** Requires careful tuning of parameters like the regularization parameter (C) and the choice of kernel.
- **Less effective with overlapping classes:** Performance drops when classes are not clearly separable.

**Use Cases:**
- Text categorization
- Image recognition
- Bioinformatics
""")