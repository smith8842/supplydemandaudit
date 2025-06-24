# ----------------------------------------
# SD Audit - All Metrics Grouped by Oracle-Aligned Schema
# ----------------------------------------

import streamlit as st
import pandas as pd

# Set up page
st.set_page_config(page_title="SD Audit - All Metrics", layout="wide")
st.title("📊 SD Audit - Full Supply & Demand Health Audit")
st.markdown("Upload your Oracle-exported Excel file to analyze.")

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    # Load sheets
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names
    st.success(f"✅ File uploaded. Sheets: {', '.join(sheet_names)}")

    # Load Oracle-aligned sheets
    part_master_df = pd.read_excel(uploaded_file, sheet_name="PART_MASTER")
    po_df = pd.read_excel(uploaded_file, sheet_name="PURCHASE_ORDER_LINES")
    wo_df = pd.read_excel(uploaded_file, sheet_name="WORK_ORDERS")
    mrp_df = pd.read_excel(uploaded_file, sheet_name="MRP_SUGGESTIONS")
    forecast_df = pd.read_excel(uploaded_file, sheet_name="FORECAST")
    consumption_df = pd.read_excel(uploaded_file, sheet_name="ACTUAL_CONSUMPTION")
    scrap_df = pd.read_excel(uploaded_file, sheet_name="SCRAP")
    inventory_df = pd.read_excel(uploaded_file, sheet_name="INVENTORY_BALANCES")

    # Display sheet previews
    for name, df in {
        "PART_MASTER": part_master_df,
        "PURCHASE_ORDER_LINES": po_df,
        "WORK_ORDERS": wo_df,
        "MRP_SUGGESTIONS": mrp_df,
        "FORECAST": forecast_df,
        "ACTUAL_CONSUMPTION": consumption_df,
        "SCRAP": scrap_df,
        "INVENTORY_BALANCES": inventory_df,
    }.items():
        with st.expander(f"📄 {name}"):
            st.dataframe(df.head())

    # --- WHAT Metrics: Supply Planning Results ---
    st.markdown("---")
    st.header("🔢 WHAT - Supply Planning Results")

    latest_demand = forecast_df.groupby("PART_ID").agg({"QUANTITY": "sum"}).rename(columns={"QUANTITY": "DEMAND"})
    inventory_agg = inventory_df.groupby("PART_ID").agg({"ON_HAND_QUANTITY": "sum"})
    what_df = part_master_df.set_index("PART_ID").join([latest_demand, inventory_agg])
    what_df = what_df.fillna(0)

    what_df["SHORTAGE"] = what_df["ON_HAND_QUANTITY"] < what_df["DEMAND"]
    shortage_percent = (what_df["SHORTAGE"].sum() / len(what_df)) * 100

    what_df["EXCESS"] = (what_df["ON_HAND_QUANTITY"] > (what_df["DEMAND"] + what_df["SAFETY_STOCK"])) | (what_df["ON_HAND_QUANTITY"] > what_df["MAX_QTY"])
    excess_percent = (what_df["EXCESS"].sum() / len(what_df)) * 100

    trailing_consumption = consumption_df.groupby("PART_ID")["QUANTITY"].sum()
    what_df = what_df.join(trailing_consumption.rename("TRAILING_CONSUMPTION"))
    what_df["INVENTORY_TURNS"] = what_df["TRAILING_CONSUMPTION"] / (what_df["ON_HAND_QUANTITY"] + 1)
    avg_turns = what_df["INVENTORY_TURNS"].mean()

    with st.expander("📊 WHAT Metrics Results"):
        col1, col2, col3 = st.columns(3)
        col1.metric("% of Parts with Material Shortages", f"{shortage_percent:.1f}%")
        col2.metric("% of Parts with Excess Inventory", f"{excess_percent:.1f}%")
        col3.metric("Avg Inventory Turns", f"{avg_turns:.1f}")
        st.dataframe(what_df.reset_index())

    # --- WHY Metrics: Root Cause ---
    st.markdown("---")
    st.header("🔎 WHY - Root Cause Metrics")

    # % Late Purchase Orders (closed only)
    closed_po_df = po_df[po_df["STATUS"].str.lower() == "closed"]
    closed_po_df["LATE"] = pd.to_datetime(closed_po_df["RECEIPT_DATE"]) > pd.to_datetime(closed_po_df["NEED_BY_DATE"])
    po_late_percent = (closed_po_df["LATE"].sum() / len(closed_po_df)) * 100 if len(closed_po_df) > 0 else 0

    # % Late Work Orders (closed only)
    closed_wo_df = wo_df[wo_df["STATUS"].str.lower() == "closed"]
    closed_wo_df["LATE"] = pd.to_datetime(closed_wo_df["COMPLETION_DATE"]) > pd.to_datetime(closed_wo_df["DUE_DATE"])
    wo_late_percent = (closed_wo_df["LATE"].sum() / len(closed_wo_df)) * 100 if len(closed_wo_df) > 0 else 0

    with st.expander("📊 WHY Metrics Results"):
        col1, col2 = st.columns(2)
        col1.metric("% Late Purchase Orders", f"{po_late_percent:.1f}%")
        col2.metric("% Late Work Orders", f"{wo_late_percent:.1f}%")
        st.dataframe(pd.concat([
            closed_po_df[["PART_ID", "RECEIPT_DATE", "NEED_BY_DATE", "LATE"]].head(),
            closed_wo_df[["PART_ID", "COMPLETION_DATE", "DUE_DATE", "LATE"]].head()
        ], keys=["PO Sample", "WO Sample"]))

    # --- HOW Placeholder ---
    st.markdown("---")
    st.header("💡 HOW - Optimization Recommendations")
    st.info("Ideal LT, Min/Max, and Planning Method Suggestions will be provided based on root cause findings.")

