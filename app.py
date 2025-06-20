# ----------------------------------------
# SD Audit - All Metrics Grouped by Category
# ----------------------------------------

import streamlit as st
import pandas as pd

# Set up page
st.set_page_config(page_title="SD Audit - All Metrics", layout="wide")
st.title("📊 SD Audit - Full Supply & Demand Health Audit")
st.markdown("Upload your Excel export from your ERP/MES system to analyze.")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    # Load sheets
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    st.success(f"✅ File uploaded. Sheets: {', '.join(sheet_names)}")

    # Load each sheet into a DataFrame
    po_df = pd.read_excel(uploaded_file, sheet_name="Purchase Orders") if "Purchase Orders" in sheet_names else None
    wo_df = pd.read_excel(uploaded_file, sheet_name="Work Orders") if "Work Orders" in sheet_names else None
    forecast_df = pd.read_excel(uploaded_file, sheet_name="Forecast") if "Forecast" in sheet_names else None
    consumption_df = pd.read_excel(uploaded_file, sheet_name="Consumption") if "Consumption" in sheet_names else None
    settings_df = pd.read_excel(uploaded_file, sheet_name="Item Settings") if "Item Settings" in sheet_names else None
    mrp_df = pd.read_excel(uploaded_file, sheet_name="MRP Messages") if "MRP Messages" in sheet_names else None

    # Preview uploaded data in expandable sections
    with st.expander("📄 Purchase Orders"):
        if po_df is not None:
            st.dataframe(po_df)

    with st.expander("📄 Work Orders"):
        if wo_df is not None:
            st.dataframe(wo_df)

    with st.expander("📄 Forecast"):
        if forecast_df is not None:
            st.dataframe(forecast_df)

    with st.expander("📄 Consumption"):
        if consumption_df is not None:
            st.dataframe(consumption_df)

    with st.expander("📄 Item Settings"):
        if settings_df is not None:
            st.dataframe(settings_df)

    with st.expander("📄 MRP Messages"):
        if mrp_df is not None:
            st.dataframe(mrp_df)

    # Dictionary to collect results by category
    results = {
        "Procurement Metrics": {},
        "Production Metrics": {},
        "Forecasting Metrics": {},
        "Planning Parameter Metrics": {},
        "MRP Action Metrics": {}
    }

# --------------------
# Metrics Calculations
# ---------------------
    
    # -------------------------------
    # Procurement Metrics
    # -------------------------------
    if po_df is not None and not po_df.empty:
        # --- % Late Purchase Orders ---
        po_df['Is Late'] = po_df['Received Date'] > po_df['Required Date']
        po_late_percent = po_df['Is Late'].mean() * 100  # Every PO counts, even 1 day late
    
        # --- PO Lead Time Accuracy ---
        po_df['Actual Lead Time'] = (po_df['Received Date'] - po_df['Order Date']).dt.days
        po_avg_lead = po_df.groupby('Item').agg(
            Actual_Lead_Time=('Actual Lead Time', 'mean'),
            PO_Count=('Actual Lead Time', 'count')
        ).reset_index()
        po_avg_lead = po_avg_lead.merge(settings_df[['Item', 'Lead Time (Days)']], on='Item', how='left')
        po_avg_lead = po_avg_lead[po_avg_lead['PO_Count'] >= 3]  # Only parts with 3+ POs
        po_avg_lead['Accurate'] = abs(po_avg_lead['Actual_Lead_Time'] - po_avg_lead['Lead Time (Days)']) / po_avg_lead['Lead Time (Days)'] <= 0.10
        po_lead_time_accuracy = po_avg_lead['Accurate'].mean() * 100 if not po_avg_lead.empty else None
    else:
        po_late_percent = None
        po_lead_time_accuracy = None

    # -------------------------------
    # Production Metrics
    # -------------------------------
    if wo_df is not None and not wo_df.empty:
        # --- % Late Work Orders ---
        wo_df['Is Late'] = wo_df['Completed Date'] > wo_df['Due Date']
        wo_late_percent = wo_df['Is Late'].mean() * 100  # Every WO counts, even 1 day late
    
        # --- WO Lead Time Accuracy ---
        wo_df['Actual Cycle Time'] = (wo_df['Completed Date'] - wo_df['Start Date']).dt.days
        wo_avg_lead = wo_df.groupby('Item').agg(
            Actual_Lead_Time=('Actual Cycle Time', 'mean'),
            WO_Count=('Actual Cycle Time', 'count')
        ).reset_index()
        wo_avg_lead = wo_avg_lead.merge(settings_df[['Item', 'Lead Time (Days)']], on='Item', how='left')
        wo_avg_lead = wo_avg_lead[wo_avg_lead['WO_Count'] >= 3]  # Only parts with 3+ WOs
        wo_avg_lead['Accurate'] = abs(wo_avg_lead['Actual_Lead_Time'] - wo_avg_lead['Lead Time (Days)']) / wo_avg_lead['Lead Time (Days)'] <= 0.10
        wo_lead_time_accuracy = wo_avg_lead['Accurate'].mean() * 100 if not wo_avg_lead.empty else None
    else:
        wo_late_percent = None
        wo_lead_time_accuracy = None


    # ----------------------------------------
    # Forecasting Metrics
    # ----------------------------------------
    if forecast_df is not None and consumption_df is not None:
        f = forecast_df.groupby("Item")["Forecast Qty"].sum()
        c = consumption_df.groupby("Item")["Qty Used"].sum()
        fc = pd.concat([f, c], axis=1).dropna()
        fc["Accuracy"] = 1 - abs(fc["Forecast Qty"] - fc["Qty Used"]) / fc["Qty Used"]

        v = forecast_df.groupby(["Item"])["Forecast Qty"].std()

        results["Forecasting Metrics"]["Forecast Accuracy"] = fc["Accuracy"].mean() * 100 if not fc.empty else 0
        results["Forecasting Metrics"]["Forecast Volatility"] = v.mean() if not v.empty else 0

    # ----------------------------------------
    # Planning Parameter Metrics
    # ----------------------------------------
    if settings_df is not None and consumption_df is not None:
        param_df = settings_df.copy()
        c = consumption_df.groupby("Item")["Qty Used"].mean().rename("Avg Usage")
        merged = pd.merge(param_df, c, on="Item", how="inner")

        # Assume Planning Method column exists with "Reorder Point", "Min/Max", "MRP"
        ro_df = merged[merged["Planning Method"].isin(["Reorder Point", "Min/Max"])].copy()

        # Statistical recommendations
        ro_df["Reco Reorder Point"] = ro_df["Avg Usage"] * 1.5  # simplified formula
        ro_df["Reco Min"] = ro_df["Avg Usage"]
        ro_df["Reco Max"] = ro_df["Avg Usage"] * 2

        def within_threshold(actual, recommended):
            return abs(actual - recommended) / recommended <= 0.1

        reorder_effective = ro_df.apply(lambda r: within_threshold(r["Reorder Point"], r["Reco Reorder Point"]), axis=1)
        min_effective = ro_df.apply(lambda r: within_threshold(r["Min Qty"], r["Reco Min"]), axis=1)
        max_effective = ro_df.apply(lambda r: within_threshold(r["Max Qty"], r["Reco Max"]), axis=1)

        results["Planning Parameter Metrics"]["Safety Stock Coverage"] = ro_df["Safety Stock"].notna().mean() * 100
        results["Planning Parameter Metrics"]["Min/Max Appropriateness"] = min(min_effective.mean(), max_effective.mean()) * 100
        results["Planning Parameter Metrics"]["Reorder Point Effectiveness"] = reorder_effective.mean() * 100

    # ----------------------------------------
    # MRP Action Metrics
    # ----------------------------------------
    if mrp_df is not None:
        df = mrp_df.copy()
        df["Message Date"] = pd.to_datetime(df["Message Date"])
        df["Action Date"] = pd.to_datetime(df["Action Date"])
        df["Lead Time"] = (df["Action Date"] - df["Message Date"]).dt.days

        results["MRP Action Metrics"]["MRP Message Timeliness"] = df["Lead Time"].mean() if not df.empty else 0

# ----------------------------------------
# Scorecard Display
# ----------------------------------------
    st.title("🧾 Supply & Demand Audit Scorecard")
    
    # -------------------------------
    # Procurement Metrics - UI
    # -------------------------------
    with st.container():
        st.subheader("📦 Procurement Metrics")
    
        col1, col2 = st.columns(2)
    
        with col1:
            if po_late_percent is not None:
                st.metric(label="% Late Purchase Orders", value=f"{po_late_percent:.1f}%")
    
        with col2:
            if po_lead_time_accuracy is not None:
                st.metric(label="PO Lead Time Accuracy", value=f"{po_lead_time_accuracy:.1f}%")
    
    # -------------------------------
    # Production Metrics - UI
    # -------------------------------
    with st.container():
        st.subheader("🏭 Production Metrics")
    
        col1, col2 = st.columns(2)
    
        with col1:
            if wo_late_percent is not None:
                st.metric(label="% Late Work Orders", value=f"{wo_late_percent:.1f}%")
    
        with col2:
            if wo_lead_time_accuracy is not None:
                st.metric(label="WO Lead Time Accuracy", value=f"{wo_lead_time_accuracy:.1f}%")
    
    # -------------------------------
    # Other Metrics (Loop for now)
    # -------------------------------
    # Loop through remaining categories (e.g., Forecasting, Planning Parameter, MRP Actions)
    # that are stored in a dictionary called `results`
    for category, metrics in results.items():
        # Skip Procurement and Production because they are rendered above
        if category in ["📦 Procurement Metrics", "🏭 Production Metrics"]:
            continue
    
        # Only display category if it has metrics
        if metrics:
            st.subheader(category)
            cols = st.columns(len(metrics))
    
            for col, (label, value) in zip(cols, metrics.items()):
                col.metric(label, f"{value:.1f}")

