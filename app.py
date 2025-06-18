
# ----------------------------------------
# SD Audit - Streamlit App Entry Point
# ----------------------------------------

# Import necessary libraries
import streamlit as st
import pandas as pd

# ----------------------------------------
# Section: Page Setup
# ----------------------------------------

# Configure the page layout and title for the web app
st.set_page_config(page_title="SD Audit - Supply & Demand Audit", layout="wide")

# Display the main title of the app
st.title("📊 SD Audit - Supply & Demand Audit Tool")

# Short instruction text
st.markdown("Upload your Excel export from Oracle ERP to get started.")

# ----------------------------------------
# Section: File Upload Interface
# ----------------------------------------

# Widget to allow users to upload an Excel file
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

# ----------------------------------------
# Section: File Preview Interface
# ----------------------------------------

# If a file has been uploaded, try to read and preview it
if uploaded_file:
    try:
        # Load the Excel file into memory using pandas
        xls = pd.ExcelFile(uploaded_file)

        # Extract the sheet names from the Excel file
        sheet_names = xls.sheet_names

        # Show success message listing available sheets
        st.success(f"✅ File uploaded. Sheets detected: {', '.join(sheet_names)}")



# ----------------------------------------
# Section: Combined Metric Scorecard Display
# ----------------------------------------

# Only run if both required sheets exist
if uploaded_file and "Purchase Orders" in sheet_names:

    # Load needed sheets
    po_df = pd.read_excel(uploaded_file, sheet_name="Purchase Orders")
    settings_df = pd.read_excel(uploaded_file, sheet_name="Item Settings") if "Item Settings" in sheet_names else None

    st.header("📈 SD Audit Scorecard")

    col1, col2 = st.columns(2)

    # -------------------------------
    # Metric 1: % Late Purchase Orders
    # -------------------------------
    with col1:
        # Convert dates
        po_df["Required Date"] = pd.to_datetime(po_df["Required Date"])
        po_df["Received Date"] = pd.to_datetime(po_df["Received Date"])
        po_df["Is Late"] = po_df["Received Date"] > po_df["Required Date"]

        total_pos = len(po_df)
        late_pos = po_df["Is Late"].sum()
        late_percent = (late_pos / total_pos) * 100 if total_pos else 0

        st.metric(
            label="📦 % Late Purchase Orders",
            value=f"{late_percent:.1f}%",
            delta=f"{late_pos} of {total_pos} POs",
            delta_color="inverse" if late_percent < 10 else "normal"
        )

    # -------------------------------
    # Metric 2: Lead Time Accuracy
    # -------------------------------
    with col2:
        if settings_df is not None:
            po_df["Order Date"] = pd.to_datetime(po_df["Order Date"])
            po_df["Actual Lead Time"] = (po_df["Received Date"] - po_df["Order Date"]).dt.days

            merged_df = pd.merge(po_df, settings_df[["Item", "Lead Time (Days)"]],
                                 on="Item", how="inner")
            merged_df.dropna(subset=["Actual Lead Time", "Lead Time (Days)"], inplace=True)

            merged_df["Lead Time Accuracy"] = 1 - (
                abs(merged_df["Actual Lead Time"] - merged_df["Lead Time (Days)"]) /
                merged_df["Lead Time (Days)"]
            )
            merged_df = merged_df[merged_df["Lead Time Accuracy"] >= 0]

            avg_accuracy = merged_df["Lead Time Accuracy"].mean() * 100 if not merged_df.empty else 0.0

            st.metric(
                label="📏 Lead Time Accuracy",
                value=f"{avg_accuracy:.1f}%",
                delta="vs ERP Planned",
                delta_color="inverse" if avg_accuracy >= 90 else "normal"
            )

with st.expander("🔍 View Late Purchase Orders"):
    st.dataframe(po_df[po_df["Is Late"] == True])

if settings_df is not None and not merged_df.empty:
    with st.expander("🔍 View Lead Time Accuracy Details"):
        st.dataframe(merged_df[["Item", "Actual Lead Time", "Lead Time (Days)", "Lead Time Accuracy"]])

