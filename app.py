
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
# Section: Late Purchase Order Analysis
# ----------------------------------------

# Check if the user selected the "Purchase Orders" sheet
if uploaded_file and selected_sheet == "Purchase Orders":
    # Load the selected sheet into a DataFrame
    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)

    # Section header in the UI
    st.subheader("🔍 Purchase Order Analysis")

    # ----------------------------------------
    # Step 1: Convert date columns to datetime
    # ----------------------------------------
    df["Required Date"] = pd.to_datetime(df["Required Date"])
    df["Received Date"] = pd.to_datetime(df["Received Date"])

    # ----------------------------------------
    # Step 2: Flag each PO as late or on-time
    # ----------------------------------------
    df["Is Late"] = df["Received Date"] > df["Required Date"]

    # ----------------------------------------
    # Step 3: Calculate late PO statistics
    # ----------------------------------------
    total_pos = len(df)                # Total number of POs in the sheet
    late_pos = df["Is Late"].sum()     # Count of POs where 'Is Late' is True

    # Prevent division by zero if file is empty
    if total_pos > 0:
        late_po_percent = (late_pos / total_pos) * 100
    else:
        late_po_percent = 0.0

    # ----------------------------------------
    # Step 4: Display results in the Streamlit UI
    # ----------------------------------------
    st.metric(
        label="📦 % of Late Purchase Orders",                 # Display title
        value=f"{late_po_percent:.1f}%",                     # Main metric
        delta=f"{late_pos} of {total_pos} POs",              # Extra detail below
        delta_color="inverse" if late_po_percent < 10 else "normal"  # Green if good
    )

    # ----------------------------------------
    # Step 5: Show detailed table of late POs (optional)
    # ----------------------------------------
    with st.expander("View Late Purchase Orders"):
        st.dataframe(df[df["Is Late"] == True])

