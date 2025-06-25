# ----------------------------------------
# SD Audit - All Metrics Grouped by Oracle-Aligned Schema
# ----------------------------------------

import streamlit as st
import pandas as pd
import numpy as np

# Set up page
st.set_page_config(page_title="SD Audit - All Metrics", layout="wide")
st.title("📊 SD Audit - Full Supply & Demand Health Audit")
st.markdown("Upload your Oracle-exported Excel file to analyze.")

# Define analysis parameters
trailing_days = 90
z_score = 1.65  # corresponds to ~95% service level

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    # --- Load All Sheets ---
    xls = pd.ExcelFile(uploaded_file)
    part_master_df = pd.read_excel(xls, sheet_name="PART_MASTER")
    po_df = pd.read_excel(xls, sheet_name="PURCHASE_ORDER_LINES")
    wo_df = pd.read_excel(xls, sheet_name="WORK_ORDERS")
    mrp_df = pd.read_excel(xls, sheet_name="MRP_SUGGESTIONS")
    consumption_df = pd.read_excel(xls, sheet_name="ACTUAL_CONSUMPTION")
    inventory_df = pd.read_excel(xls, sheet_name="INVENTORY_BALANCES")
    
# --------------------------------
# Metrics Calculations
# --------------------------------
    
    # --- WHAT METRICS ---
    inventory_agg = inventory_df.groupby("PART_ID")["ON_HAND_QUANTITY"].sum()
    what_df = part_master_df.set_index("PART_ID").join(inventory_agg)
    what_df = what_df.fillna(0)

    trailing_consumption = consumption_df.groupby("PART_ID")["QUANTITY"].sum()
    trailing_avg_daily = trailing_consumption / trailing_days
    what_df = what_df.join(trailing_consumption.rename("TRAILING_CONSUMPTION"))
    what_df = what_df.join(trailing_avg_daily.rename("AVG_DAILY_CONSUMPTION"))

    # MRP Excess logic
    mrp_df["NEED_BY_DATE"] = pd.to_datetime(mrp_df["NEED_BY_DATE"])
    lead_time_buffer = part_master_df.set_index("PART_ID")["LEAD_TIME"] * 1.1
    cutoff_dates = pd.to_datetime(pd.Timestamp.today() + pd.to_timedelta(lead_time_buffer, unit="D"))
    cutoff_dates.name = "CUTOFF_DATE"
    mrp_parts = part_master_df[part_master_df["PLANNING_METHOD"] == "MRP"]["PART_ID"]

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

    # ROP/MinMax Excess logic
    ropmm_parts = part_master_df[part_master_df["PLANNING_METHOD"].isin(["ROP", "MIN_MAX"])]
    what_df["IDEAL_MINIMUM"] = (
        what_df["AVG_DAILY_CONSUMPTION"] * what_df["LEAD_TIME"] * 1.1 + what_df["SAFETY_STOCK"]
    )
    what_df["IDEAL_MAXIMUM"] = what_df["IDEAL_MINIMUM"] * 1.1
    ropmm_excess_parts = what_df[
        (what_df.index.isin(ropmm_parts["PART_ID"])) &
        (what_df["ON_HAND_QUANTITY"] > what_df["IDEAL_MAXIMUM"])
    ].index.tolist()
    what_df.loc[ropmm_excess_parts, "EXCESS"] = True

    # Inventory turns
    what_df["INVENTORY_TURNS"] = what_df["TRAILING_CONSUMPTION"] / (what_df["ON_HAND_QUANTITY"] + 1)
    avg_turns = what_df["INVENTORY_TURNS"].mean()

    # Shortage logic (simplified for now)
    late_pos = po_df[(po_df["STATUS"].str.lower() == "open") & (pd.to_datetime(po_df["RECEIPT_DATE"]) > pd.to_datetime(po_df["NEED_BY_DATE"]))]["PART_ID"].unique()
    late_wos = wo_df[(wo_df["STATUS"].str.lower() == "open") & (pd.to_datetime(wo_df["COMPLETION_DATE"]) > pd.to_datetime(wo_df["DUE_DATE"]))]["PART_ID"].unique()
    ropmm_shortage_parts = what_df[
        (what_df.index.isin(ropmm_parts["PART_ID"])) &
        (what_df["ON_HAND_QUANTITY"] < what_df["IDEAL_MINIMUM"])
    ].index.tolist()
    shortage_part_ids = set(ropmm_shortage_parts).union(set(late_pos)).union(set(late_wos))
    what_df["SHORTAGE"] = what_df.index.isin(shortage_part_ids)

    shortage_percent = (what_df["SHORTAGE"].sum() / len(what_df)) * 100
    excess_percent = (what_df["EXCESS"].sum() / len(what_df)) * 100

    with st.expander("📊 WHAT Metrics Results"):
        col1, col2, col3 = st.columns(3)
        col1.metric("% of Parts with Material Shortages", f"{shortage_percent:.1f}%")
        col2.metric("% of Parts with Excess Inventory", f"{excess_percent:.1f}%")
        col3.metric("Avg Inventory Turns", f"{avg_turns:.1f}")
        st.dataframe(what_df.reset_index())
        
    # -------- WHY Metrics -----------
    # --- PO and WO Late % and Lead Time Accuracy ---
    po_df["RECEIPT_DATE"] = pd.to_datetime(po_df["RECEIPT_DATE"])
    po_df["NEED_BY_DATE"] = pd.to_datetime(po_df["NEED_BY_DATE"])
    po_df["LT_DAYS"] = (po_df["RECEIPT_DATE"] - po_df["NEED_BY_DATE"]).dt.days
    
    # % Late POs
    total_pos = len(po_df)
    late_pos_count = len(po_df[(po_df["STATUS"].str.lower() == "open") & (po_df["RECEIPT_DATE"] > po_df["NEED_BY_DATE"])])
    po_late_percent = (late_pos_count / total_pos) * 100 if total_pos > 0 else 0
    
    # PO Lead Time Accuracy
    closed_po = po_df[po_df["STATUS"].str.lower() == "closed"]
    lt_accuracy_po = closed_po.groupby("PART_ID").agg(
        actual_lt=("LT_DAYS", "mean"),
        erp_lt=("LEAD_TIME", "first")
    ).dropna()
    lt_accuracy_po["WITHIN_TOLERANCE"] = abs(lt_accuracy_po["actual_lt"] - lt_accuracy_po["erp_lt"]) / lt_accuracy_po["erp_lt"] <= 0.10
    po_lead_time_accuracy = lt_accuracy_po["WITHIN_TOLERANCE"].mean() * 100 if not lt_accuracy_po.empty else 0
    
    # % Late WOs
    wo_df["COMPLETION_DATE"] = pd.to_datetime(wo_df["COMPLETION_DATE"])
    wo_df["DUE_DATE"] = pd.to_datetime(wo_df["DUE_DATE"])
    wo_df["WO_LT_DAYS"] = (wo_df["COMPLETION_DATE"] - wo_df["DUE_DATE"]).dt.days
    total_wos = len(wo_df)
    late_wo_count = len(wo_df[(wo_df["STATUS"].str.lower() == "open") & (wo_df["COMPLETION_DATE"] > wo_df["DUE_DATE"])])
    wo_late_percent = (late_wo_count / total_wos) * 100 if total_wos > 0 else 0
    
    # WO Lead Time Accuracy
    closed_wo = wo_df[wo_df["STATUS"].str.lower() == "closed"]
    lt_accuracy_wo = closed_wo.groupby("PART_ID").agg(
        actual_lt=("WO_LT_DAYS", "mean"),
        erp_lt=("LEAD_TIME", "first")
    ).dropna()
    lt_accuracy_wo["WITHIN_TOLERANCE"] = abs(lt_accuracy_wo["actual_lt"] - lt_accuracy_wo["erp_lt"]) / lt_accuracy_wo["erp_lt"] <= 0.10
    wo_lead_time_accuracy = lt_accuracy_wo["WITHIN_TOLERANCE"].mean() * 100 if not lt_accuracy_wo.empty else 0

    # --- SAFETY STOCK ACCURACY ---
    recent_cutoff = pd.Timestamp.today() - pd.Timedelta(days=trailing_days)
    recent_consumption = consumption_df[consumption_df["CONSUMPTION_DATE"] >= recent_cutoff]

    daily_consumption = (
        recent_consumption.groupby(["PART_ID", "CONSUMPTION_DATE"]).agg({"QUANTITY": "sum"}).reset_index()
    )
    std_dev_daily = daily_consumption.groupby("PART_ID")["QUANTITY"].std().fillna(0)

    ss_df = part_master_df.set_index("PART_ID").copy()
    ss_df = ss_df.join(std_dev_daily.rename("STD_DEV_CONSUMPTION"))
    ss_df["IDEAL_SS"] = z_score * ss_df["STD_DEV_CONSUMPTION"] * np.sqrt(ss_df["LEAD_TIME"])
    ss_df = ss_df[~ss_df["IDEAL_SS"].isnull()]
    ss_df["WITHIN_TOLERANCE"] = (
        abs(ss_df["SAFETY_STOCK"] - ss_df["IDEAL_SS"]) / ss_df["IDEAL_SS"] <= 0.10
    )
    valid_ss_parts = len(ss_df["WITHIN_TOLERANCE"])
    compliant_parts = ss_df["WITHIN_TOLERANCE"].sum()
    ss_coverage_percent = (compliant_parts / valid_ss_parts * 100) if valid_ss_parts > 0 else 0

    # Show WHY metrics in UI
    with st.expander("🧪 WHY Metrics Results"):
        col1, col2 = st.columns(2)
        col1.metric("% Late Purchase Orders", f"{po_late_percent:.1f}%")
        col2.metric("PO Lead Time Accuracy", f"{po_lead_time_accuracy:.1f}%")
        col1.metric("% Late Work Orders", f"{wo_late_percent:.1f}%")
        col2.metric("WO Lead Time Accuracy", f"{wo_lead_time_accuracy:.1f}%")
        col1.metric("% of Parts with Valid Safety Stock", f"{ss_coverage_percent:.1f}%")

        st.markdown("### Late PO Details")
        st.dataframe(po_df[po_df["STATUS"].str.lower() == "open"].loc[
            pd.to_datetime(po_df["RECEIPT_DATE"]) > pd.to_datetime(po_df["NEED_BY_DATE"]),
            ["PART_ID", "PO_NUMBER", "NEED_BY_DATE", "RECEIPT_DATE"]
        ])

        st.markdown("### PO Lead Time Accuracy Detail")
        st.dataframe(po_df[po_df["STATUS"].str.lower() == "closed"][["PART_ID", "PO_NUMBER", "NEED_BY_DATE", "RECEIPT_DATE"]])

        st.markdown("### Safety Stock Accuracy Detail")
        st.dataframe(ss_df.reset_index()[[
            "PART_ID", "LEAD_TIME", "SAFETY_STOCK", "STD_DEV_CONSUMPTION", "IDEAL_SS", "WITHIN_TOLERANCE"
        ]])
