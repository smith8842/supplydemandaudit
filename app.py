
import streamlit as st
import pandas as pd

st.set_page_config(page_title="SD Audit - Supply & Demand Audit", layout="wide")

st.title("📊 SD Audit - Supply & Demand Audit Tool")
st.markdown("Upload your Excel export from Oracle ERP to get started.")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    try:
        xls = pd.ExcelFile(uploaded_file)
        sheet_names = xls.sheet_names
        st.success(f"✅ File uploaded. Sheets detected: {', '.join(sheet_names)}")

        selected_sheet = st.selectbox("Select sheet to preview", sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        st.dataframe(df)

    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
