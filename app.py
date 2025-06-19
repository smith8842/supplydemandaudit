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

    
    # ----------------------------------------
    # Procurement Metrics
    # ----------------------------------------
    if po_df is not None and settings_df is not None:
        df = po_df.copy()
        df["Required Date"] = pd.to_datetime(df["Required Date"])
        df["Received Date"] = pd.to_datetime(df["Received Date"])
        df["Order Date"] = pd.to_datetime(df["Order Date"])
        df["Is Late"] = df["Received Date"] > df["Required Date"]
        df["Actual LT"] = (df["Received Date"] - df["Order Date"]).dt.days
        merged = pd.merge(df, settings_df[["Item", "Lead Time (Days)"]], on="Item", how="inner")
        merged["Lead Time Accuracy"] = 1 - (abs(merged["Actual LT"] - merged["Lead Time (Days)"]) / merged["Lead Time (Days)"])
        merged = merged[merged["Lead Time Accuracy"] >= 0]

        results["Procurement Metrics"]["% Late POs"] = df["Is Late"].mean() * 100 if not df.empty else 0
        results["Procurement Metrics"]["Lead Time Accuracy"] = merged["Lead Time Accuracy"].mean() * 100 if not merged.empty else 0

    # ----------------------------------------
    # Production Metrics
    # ----------------------------------------
    if wo_df is not None:
        df = wo_df.copy()
        df["Due Date"] = pd.to_datetime(df["Due Date"])
        df["Completed Date"] = pd.to_datetime(df["Completed Date"])
        df["Start Date"] = pd.to_datetime(df["Start Date"])
        df["Is Late"] = df["Completed Date"] > df["Due Date"]
        df["Actual LT"] = (df["Completed Date"] - df["Start Date"]).dt.days

        results["Production Metrics"]["% Late Work Orders"] = df["Is Late"].mean() * 100 if not df.empty else 0
        results["Production Metrics"]["WO Lead Time Accuracy"] = df["Actual LT"].mean() if not df.empty else 0

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
    for category, metrics in results.items():
        st.subheader(category)
        cols = st.columns(len(metrics))
        for col, (label, value) in zip(cols, metrics.items()):
            col.metric(label, f"{value:.1f}")
