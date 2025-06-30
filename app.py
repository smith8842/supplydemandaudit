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
trailing_days = 90 # number of days analyzed for consumption in the past
z_score = 1.65  # corresponds to ~95% service level
high_scrap_threshold = 0.10  # Set high scrap threshold (parameterized)

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    # --- Load All Sheets ---
    xls = pd.ExcelFile(uploaded_file)
    part_master_df = pd.read_excel(xls, sheet_name="PART_MASTER")
    po_df = pd.read_excel(xls, sheet_name="PURCHASE_ORDER_LINES")
  #  st.write("Columns in PURCHASE_ORDER_LINES:", po_df.columns.tolist())
    wo_df = pd.read_excel(xls, sheet_name="WORK_ORDERS")
    mrp_df = pd.read_excel(xls, sheet_name="MRP_SUGGESTIONS")
    consumption_df = pd.read_excel(xls, sheet_name="WIP_TRANSACTIONS")
    inventory_df = pd.read_excel(xls, sheet_name="INVENTORY_BALANCES")

# --------------------------------
# Metrics Calculations
# --------------------------------

    # --- WHAT METRICS ---

    # Calculate inventory aggregation and trailing consumption
    inventory_agg = inventory_df.groupby("PART_ID")["ON_HAND_QUANTITY"].sum()
    what_df = part_master_df.set_index("PART_ID").join(inventory_agg)
    what_df = what_df.fillna(0)

    # === FINALIZATION PATCH FOR PART-LEVEL AUDIT DF ===
    # Add PART_ID as a visible column (currently only index)
    what_df["PART_ID"] = what_df.index

    # Confirm essential fields exist (already included from initial join)
    # No need to rejoin PART_NUMBER or PLANNING_METHOD

    # Normalize numeric column types for downstream consistency
    numeric_cols = [
        "LEAD_TIME", "SAFETY_STOCK", "MIN_QTY", "MAX_QTY",
        "ON_HAND_QUANTITY", "TRAILING_CONSUMPTION", "AVG_DAILY_CONSUMPTION"
    ]
    for col in numeric_cols:
        if col in what_df.columns:
            what_df[col] = pd.to_numeric(what_df[col], errors="coerce").fillna(0)

    # Normalize planning method as string
    if "PLANNING_METHOD" in what_df.columns:
        what_df["PLANNING_METHOD"] = what_df["PLANNING_METHOD"].astype(str)

    trailing_consumption = consumption_df.groupby("PART_ID")["QUANTITY"].sum()
    trailing_avg_daily = trailing_consumption / trailing_days
    what_df = what_df.join(trailing_consumption.rename("TRAILING_CONSUMPTION"))
    what_df = what_df.join(trailing_avg_daily.rename("AVG_DAILY_CONSUMPTION"))

    # MRP-based Excess Logic
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

    # ROP / Min-Max Excess Logic
    ropmm_parts = part_master_df[part_master_df["PLANNING_METHOD"].isin(["ROP", "MIN_MAX"])]
    what_df["IDEAL_MINIMUM"] = what_df["AVG_DAILY_CONSUMPTION"] * what_df["LEAD_TIME"] * 1.1 + what_df["SAFETY_STOCK"]
    what_df["IDEAL_MAXIMUM"] = what_df["IDEAL_MINIMUM"] * 1.1
    ropmm_excess_parts = what_df[
        (what_df.index.isin(ropmm_parts["PART_ID"])) &
        (what_df["ON_HAND_QUANTITY"] > what_df["IDEAL_MAXIMUM"])
    ].index.tolist()
    what_df.loc[ropmm_excess_parts, "EXCESS"] = True

    # Inventory Turns
    what_df["INVENTORY_TURNS"] = what_df["TRAILING_CONSUMPTION"] / (what_df["ON_HAND_QUANTITY"] + 1)
    avg_turns = what_df["INVENTORY_TURNS"].mean()

    # Shortage Logic
    late_pos = po_df[
        (po_df["STATUS"].str.lower() == "open") &
        (po_df["RECEIPT_DATE"] > po_df["NEED_BY_DATE"])
    ]["PART_ID"].unique()

    late_wos = wo_df[
        (wo_df["STATUS"].str.lower() == "open") &
        (wo_df["COMPLETION_DATE"] > wo_df["DUE_DATE"])
    ]["PART_ID"].unique()

    ropmm_shortage_parts = what_df[
        (what_df.index.isin(ropmm_parts["PART_ID"])) &
        (what_df["ON_HAND_QUANTITY"] < what_df["IDEAL_MINIMUM"])
    ].index.tolist()

    shortage_part_ids = set(ropmm_shortage_parts).union(set(late_pos)).union(set(late_wos))
    what_df["SHORTAGE"] = what_df.index.isin(shortage_part_ids)

    shortage_percent = (what_df["SHORTAGE"].sum() / len(what_df)) * 100
    excess_percent = (what_df["EXCESS"].sum() / len(what_df)) * 100

    # -------- WHY Metrics -----------
    # --- PO and WO Late % and Lead Time Accuracy ---
    po_df["RECEIPT_DATE"] = pd.to_datetime(po_df["RECEIPT_DATE"])
    po_df["NEED_BY_DATE"] = pd.to_datetime(po_df["NEED_BY_DATE"])
    po_df["LT_DAYS"] = (po_df["RECEIPT_DATE"] - po_df["NEED_BY_DATE"]).dt.days

    total_pos = len(po_df)
    late_pos_count = len(po_df[(po_df["STATUS"].str.lower() == "open") & (po_df["RECEIPT_DATE"] > po_df["NEED_BY_DATE"])] )
    po_late_percent = (late_pos_count / total_pos) * 100 if total_pos > 0 else 0

    closed_po = po_df[po_df["STATUS"].str.lower() == "closed"]
    lt_accuracy_po = closed_po.groupby("PART_ID").agg(actual_lt=("LT_DAYS", "mean")).dropna()
    lt_accuracy_po = lt_accuracy_po.join(part_master_df.set_index("PART_ID")["LEAD_TIME"].rename("erp_lt"))
    lt_accuracy_po = lt_accuracy_po.dropna()
    lt_accuracy_po["WITHIN_TOLERANCE"] = abs(lt_accuracy_po["actual_lt"] - lt_accuracy_po["erp_lt"]) / lt_accuracy_po["erp_lt"] <= 0.10
    po_lead_time_accuracy = lt_accuracy_po["WITHIN_TOLERANCE"].mean() * 100 if not lt_accuracy_po.empty else 0

    wo_df["COMPLETION_DATE"] = pd.to_datetime(wo_df["COMPLETION_DATE"])
    wo_df["DUE_DATE"] = pd.to_datetime(wo_df["DUE_DATE"])
    wo_df["WO_LT_DAYS"] = (wo_df["COMPLETION_DATE"] - wo_df["DUE_DATE"]).dt.days
    total_wos = len(wo_df)
    late_wo_count = len(wo_df[(wo_df["STATUS"].str.lower() == "open") & (wo_df["COMPLETION_DATE"] > wo_df["DUE_DATE"])] )
    wo_late_percent = (late_wo_count / total_wos) * 100 if total_wos > 0 else 0

    closed_wo = wo_df[wo_df["STATUS"].str.lower() == "closed"]
    lt_accuracy_wo = closed_wo.groupby("PART_ID").agg(actual_lt=("WO_LT_DAYS", "mean")).dropna()
    lt_accuracy_wo = lt_accuracy_wo.join(part_master_df.set_index("PART_ID")["LEAD_TIME"].rename("erp_lt"))
    lt_accuracy_wo = lt_accuracy_wo.dropna()
    lt_accuracy_wo["WITHIN_TOLERANCE"] = abs(lt_accuracy_wo["actual_lt"] - lt_accuracy_wo["erp_lt"]) / lt_accuracy_wo["erp_lt"] <= 0.10
    wo_lead_time_accuracy = lt_accuracy_wo["WITHIN_TOLERANCE"].mean() * 100 if not lt_accuracy_wo.empty else 0

    # --- Safety Stock Accuracy Calculation ---
    recent_cutoff = pd.Timestamp.today() - pd.Timedelta(days=trailing_days)
    recent_consumption = consumption_df[consumption_df["TRANSACTION_DATE"] >= recent_cutoff]

    daily_consumption = (
        recent_consumption.groupby(["PART_ID", "TRANSACTION_DATE"]).agg({"QUANTITY": "sum"}).reset_index()
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

    # --- Scrap Rate Calculation (from WIP_TRANSACTIONS) ---
    scrap_transactions = consumption_df[consumption_df["TRANSACTION_TYPE"] == "Scrap"]
    consumed_transactions = consumption_df[
        consumption_df["TRANSACTION_TYPE"].isin(["Backflush", "Manual Issue"])
    ]

    total_scrap_by_part = scrap_transactions.groupby("PART_ID")["QUANTITY"].sum()
    total_consumed_by_part = consumed_transactions.groupby("PART_ID")["QUANTITY"].sum()

    scrap_rate_by_part = total_scrap_by_part / (total_scrap_by_part + total_consumed_by_part)
    scrap_rate_by_part = scrap_rate_by_part.fillna(0)

    part_detail_df = part_detail_df.join(scrap_rate_by_part.rename("AVG_SCRAP_RATE"))

    valid_scrap_parts = scrap_rate_by_part.count()
    high_scrap_parts = (scrap_rate_by_part > high_scrap_threshold).sum()
    high_scrap_percent = (high_scrap_parts / valid_scrap_parts * 100) if valid_scrap_parts > 0 else 0
        
    # Summary metric: % of parts with scrap rate > threshold
    valid_scrap_parts = scrap_rate_by_part.count()
    high_scrap_parts = (scrap_rate_by_part > high_scrap_threshold).sum()
    high_scrap_percent = (high_scrap_parts / valid_scrap_parts * 100) if valid_scrap_parts > 0 else 0

#------------------------------------    
# ------- UI for Results -----------
# -----------------------------------

    # --- UI for WHAT Metrics ---
    with st.expander("📊 WHAT Metrics Results"):
        col1, col2, col3 = st.columns(3)
        col1.metric("🔻 % of Parts with Material Shortages", f"{shortage_percent:.1f}%")
        col2.metric("📦 % of Parts with Excess Inventory", f"{excess_percent:.1f}%")
        col3.metric("🔁 Avg Inventory Turns", f"{avg_turns:.1f}")

        st.markdown("### Detailed WHAT Metrics Table")
        st.dataframe(what_df[[
            "PART_NUMBER", "SHORTAGE", "EXCESS", "INVENTORY_TURNS", 
            "ON_HAND_QUANTITY", "TRAILING_CONSUMPTION", 
            "AVG_DAILY_CONSUMPTION", "SAFETY_STOCK", "MIN_QTY", "MAX_QTY", "LEAD_TIME"
        ]])

    # --- UI for WHY Metrics ---
    with st.expander("🔍 WHY Metrics Results"):
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("📦 % Late Purchase Orders", f"{po_late_percent:.1f}%")
        col2.metric("🏭 % Late Work Orders", f"{wo_late_percent:.1f}%")
        col3.metric("📈 PO Lead Time Accuracy", f"{po_lead_time_accuracy:.1f}%")
        col4.metric("🛠️ WO Lead Time Accuracy", f"{wo_lead_time_accuracy:.1f}%")
        col5.metric("🛡️ SS Coverage Accuracy", f"{ss_coverage_percent:.1f}%")
        col6.metric("🧯 % of Parts with High Scrap", f"{high_scrap_percent:.1f}%")

        st.markdown("### Detailed WHY Metrics Table — by Part")
        part_detail_df = part_master_df.copy()
        part_detail_df = part_detail_df.set_index("PART_ID")
        part_detail_df = part_detail_df.join(trailing_avg_daily.rename("AVG_DAILY_CONSUMPTION"))
        part_detail_df = part_detail_df.join(ss_df[["IDEAL_SS", "WITHIN_TOLERANCE"]].rename(columns={"WITHIN_TOLERANCE": "SS_COMPLIANT_PART"}))
        part_detail_df = part_detail_df.join(lt_accuracy_po[["actual_lt", "WITHIN_TOLERANCE"]].rename(columns={"actual_lt": "AVG_PO_LEAD_TIME", "WITHIN_TOLERANCE": "PO_LEAD_TIME_ACCURATE"}))
        part_detail_df = part_detail_df.join(lt_accuracy_wo[["actual_lt", "WITHIN_TOLERANCE"]].rename(columns={"actual_lt": "AVG_WO_LEAD_TIME", "WITHIN_TOLERANCE": "WO_LEAD_TIME_ACCURATE"}))

        st.dataframe(part_detail_df.reset_index()[[
            "PART_ID", "PART_NUMBER", "LEAD_TIME", "SAFETY_STOCK", "AVG_DAILY_CONSUMPTION",
            "IDEAL_SS", "SS_COMPLIANT_PART", "PO_LEAD_TIME_ACCURATE", "WO_LEAD_TIME_ACCURATE",
            "AVG_WO_LEAD_TIME", "AVG_PO_LEAD_TIME", "AVG_SCRAP_RATE"
        ]])

        st.markdown("### Detailed WHY Metrics Table — by Order")
        po_order_df = po_df[po_df["STATUS"].str.lower().isin(["open", "closed"])]
        po_order_df["ERP_LEAD_TIME"] = po_order_df["PART_ID"].map(part_master_df.set_index("PART_ID")["LEAD_TIME"])        
        po_order_df = po_order_df.assign(
            ORDER_TYPE="PO",
            ORDER_ID=po_order_df["PO_LINE_ID"],
            NEED_BY_DATE=po_order_df["NEED_BY_DATE"],
            RECEIPT_DATE=po_order_df["RECEIPT_DATE"],
            IS_LATE=po_order_df["RECEIPT_DATE"] > po_order_df["NEED_BY_DATE"],
            LT_DAYS=po_order_df["LT_DAYS"],
            ERP_LEAD_TIME=po_order_df["PART_ID"].map(part_master_df.set_index("PART_ID")["LEAD_TIME"]),
            WITHIN_10_PERCENT=(abs(po_order_df["LT_DAYS"] - po_order_df["ERP_LEAD_TIME"]) / po_order_df["ERP_LEAD_TIME"]) <= 0.10
        )

        wo_order_df = wo_df[wo_df["STATUS"].str.lower().isin(["open", "closed"])]
        wo_order_df["ERP_LEAD_TIME"] = wo_order_df["PART_ID"].map(part_master_df.set_index("PART_ID")["LEAD_TIME"])
        wo_order_df = wo_order_df.assign(
            ORDER_TYPE="WO",
            ORDER_ID=wo_order_df["WO_ID"],
            NEED_BY_DATE=wo_order_df["DUE_DATE"],
            RECEIPT_DATE=wo_order_df["COMPLETION_DATE"],
            IS_LATE=wo_order_df["COMPLETION_DATE"] > wo_order_df["DUE_DATE"],
            LT_DAYS=wo_order_df["WO_LT_DAYS"],
            ERP_LEAD_TIME=wo_order_df["PART_ID"].map(part_master_df.set_index("PART_ID")["LEAD_TIME"]),
            WITHIN_10_PERCENT=(abs(wo_order_df["WO_LT_DAYS"] - wo_order_df["ERP_LEAD_TIME"]) / wo_order_df["ERP_LEAD_TIME"]) <= 0.10
        )

        all_orders_df = pd.concat([po_order_df, wo_order_df], ignore_index=True)

        st.dataframe(all_orders_df[[
            "ORDER_TYPE", "ORDER_ID", "PART_ID", "NEED_BY_DATE", "RECEIPT_DATE", "STATUS", "IS_LATE",
            "ERP_LEAD_TIME", "LT_DAYS", "WITHIN_10_PERCENT"
        ]])
