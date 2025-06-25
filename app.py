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

    # Convert date columns for filtering
    po_df["NEED_BY_DATE"] = pd.to_datetime(po_df["NEED_BY_DATE"])
    po_df["RECEIPT_DATE"] = pd.to_datetime(po_df["RECEIPT_DATE"])
    wo_df["DUE_DATE"] = pd.to_datetime(wo_df["DUE_DATE"])
    wo_df["COMPLETION_DATE"] = pd.to_datetime(wo_df["COMPLETION_DATE"])
    mrp_df["NEED_BY_DATE"] = pd.to_datetime(mrp_df["NEED_BY_DATE"])

    # Identify planning strategies
    mrp_parts = part_master_df[part_master_df["PLANNING_METHOD"] == "MRP"]["PART_ID"]
    rop_minmax_parts = part_master_df[part_master_df["PLANNING_METHOD"].isin(["ROP", "MIN_MAX"])]

    # Determine late open POs based on RECEIPT_DATE > NEED_BY_DATE
    late_open_pos = po_df[(po_df["STATUS"].str.lower() == "open") & (po_df["RECEIPT_DATE"] > po_df["NEED_BY_DATE"])]

    # Determine late open WOs based on COMPLETION_DATE > DUE_DATE
    late_open_wos = wo_df[(wo_df["STATUS"].str.lower() == "open") & (wo_df["COMPLETION_DATE"] > wo_df["DUE_DATE"])]

    # Identify parts with expected shortage conditions from POs/WOs
    parts_with_late_pos = late_open_pos["PART_ID"].unique()
    parts_with_late_wos = late_open_wos["PART_ID"].unique()

    # For ROP/MinMax parts, check for discrete unfulfilled demand
    today = pd.Timestamp.today()
    discrete_demand = mrp_df[
        (mrp_df["PART_ID"].isin(rop_minmax_parts["PART_ID"])) &
        (~mrp_df["SUGGESTION_TYPE"].str.upper().str.contains("GROSS")) &
        (mrp_df["NEED_BY_DATE"] < today)
    ]
    parts_with_late_demand = discrete_demand["PART_ID"].unique()

    # Combine all parts with shortage indicators
    shortage_part_ids = set(parts_with_late_pos).union(parts_with_late_wos).union(parts_with_late_demand)
    shortage_percent = (len(shortage_part_ids) / len(part_master_df)) * 100

    # Sum total on-hand inventory per part
    inventory_agg = inventory_df.groupby("PART_ID").agg({"ON_HAND_QUANTITY": "sum"})

    # Merge inventory and planning data
    what_df = part_master_df.set_index("PART_ID").join(inventory_agg)
    what_df = what_df.fillna(0)

    # Refactored EXCESS logic (MRP-planned parts only)
    lead_time_buffer = part_master_df.set_index("PART_ID")["LEAD_TIME_DAYS"] * 1.1
    cutoff_dates = pd.to_datetime(today + pd.to_timedelta(lead_time_buffer, unit="D"))
    cutoff_dates.name = "CUTOFF_DATE"

    mrp_window_flags = (
        mrp_df[mrp_df["PART_ID"].isin(mrp_parts)]
        .merge(cutoff_dates, left_on="PART_ID", right_index=True)
        .assign(IN_WINDOW=lambda df: df["NEED_BY_DATE"] <= df["CUTOFF_DATE"])
        .groupby("PART_ID")["IN_WINDOW"].any()
    )

    what_df = what_df.join(mrp_window_flags.rename("HAS_MRP_WITHIN_LT"))
    what_df["HAS_MRP_WITHIN_LT"] = what_df["HAS_MRP_WITHIN_LT"].fillna(False)
    what_df["EXCESS"] = (
        (what_df.index.isin(mrp_parts)) &
        (~what_df["HAS_MRP_WITHIN_LT"]) &
        (what_df["ON_HAND_QUANTITY"] > what_df[["SAFETY_STOCK", "MIN_QTY"]].max(axis=1))
    )
    excess_percent = (what_df["EXCESS"].sum() / len(what_df)) * 100

    # Sum total trailing consumption per part
    trailing_consumption = consumption_df.groupby("PART_ID")["QUANTITY"].sum()
    what_df = what_df.join(trailing_consumption.rename("TRAILING_CONSUMPTION"))
    what_df["INVENTORY_TURNS"] = what_df["TRAILING_CONSUMPTION"] / (what_df["ON_HAND_QUANTITY"] + 1)
    avg_turns = what_df["INVENTORY_TURNS"].mean()

    # Add shortage column back to table
    what_df["SHORTAGE"] = what_df.index.isin(shortage_part_ids)

    # Show results in UI
    with st.expander("📊 WHAT Metrics Results"):
        col1, col2, col3 = st.columns(3)
        col1.metric("% of Parts with Material Shortages", f"{shortage_percent:.1f}%")
        col2.metric("% of Parts with Excess Inventory", f"{excess_percent:.1f}%")
        col3.metric("Avg Inventory Turns", f"{avg_turns:.1f}")
        st.dataframe(what_df.reset_index())

    # --- WHY Metrics: Root Cause ---
    st.markdown("---")
    st.header("🔎 WHY - Root Cause Metrics")

    # Filter PO data to closed orders only
    closed_po_df = po_df[po_df["STATUS"].str.lower() == "closed"]
    closed_po_df["LATE"] = pd.to_datetime(closed_po_df["RECEIPT_DATE"]) > pd.to_datetime(closed_po_df["NEED_BY_DATE"])
    po_late_percent = (closed_po_df["LATE"].sum() / len(closed_po_df)) * 100 if len(closed_po_df) > 0 else 0

    # Filter WO data to closed orders only
    closed_wo_df = wo_df[wo_df["STATUS"].str.lower() == "closed"]
    closed_wo_df["LATE"] = pd.to_datetime(closed_wo_df["COMPLETION_DATE"]) > pd.to_datetime(closed_wo_df["DUE_DATE"])
    wo_late_percent = (closed_wo_df["LATE"].sum() / len(closed_wo_df)) * 100 if len(closed_wo_df) > 0 else 0

    # Show results in UI
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
