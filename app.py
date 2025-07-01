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
lt_buffer_multiplier = 1.1  # % buffer for lead time calculations (used in MRP suggestion window)
inventory_buffer_multiplier = 1.1  # % buffer for inventory Min/Max logic
eoq_holding_cost_per_unit = 2  # assumed holding cost per unit over the trailing days period
eoq_order_cost = 100  # assumed cost to place an order
ss_tolerance_pct = 0.10  # % tolerance allowed for SS accuracy checks
lt_tolerance_pct = 0.10  # % tolerance allowed for lead time accuracy checks
valid_consumption_types = ["Backflush", "Manual Issue"]  # WIP transaction types used as valid consumption
scrap_transaction_type = "Scrap"  # WIP transaction type used to identify scrap

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

    @st.cache_data
    def calculate_what_metrics(part_master_df, inventory_df, consumption_df, mrp_df, po_df, wo_df):
        inventory_agg = inventory_df.groupby("PART_ID")["ON_HAND_QUANTITY"].sum()
        what_df = part_master_df.set_index("PART_ID").join(inventory_agg).fillna(0)
        what_df["PART_ID"] = what_df.index
    
        # Normalize types
        numeric_cols = ["LEAD_TIME", "SAFETY_STOCK", "MIN_QTY", "MAX_QTY", "ON_HAND_QUANTITY", "TRAILING_CONSUMPTION", "AVG_DAILY_CONSUMPTION"]
        for col in numeric_cols:
            if col in what_df.columns:
                what_df[col] = pd.to_numeric(what_df[col], errors="coerce").fillna(0)
    
        if "PLANNING_METHOD" in what_df.columns:
            what_df["PLANNING_METHOD"] = what_df["PLANNING_METHOD"].astype(str)
    
        # Consumption stats
        trailing_consumption = consumption_df.groupby("PART_ID")["QUANTITY"].sum()
        trailing_avg_daily = trailing_consumption / trailing_days
        what_df = what_df.join(trailing_consumption.rename("TRAILING_CONSUMPTION"))
        what_df = what_df.join(trailing_avg_daily.rename("AVG_DAILY_CONSUMPTION"))
    
        # Ideal SS
        recent_cutoff = pd.Timestamp.today() - pd.Timedelta(days=trailing_days)
        recent = consumption_df[consumption_df["TRANSACTION_DATE"] >= recent_cutoff]
        daily = recent.groupby(["PART_ID", "TRANSACTION_DATE"]).agg({"QUANTITY": "sum"}).reset_index()
        std_dev_daily = daily.groupby("PART_ID")["QUANTITY"].std().fillna(0)
        ss_temp = part_master_df.set_index("PART_ID").copy()
        ss_temp = ss_temp.join(std_dev_daily.rename("STD_DEV_CONSUMPTION"))
        ss_temp["IDEAL_SS"] = z_score * ss_temp["STD_DEV_CONSUMPTION"] * np.sqrt(ss_temp["LEAD_TIME"])
        what_df = what_df.join(ss_temp["IDEAL_SS"])
    
        # MRP logic
        mrp_df["NEED_BY_DATE"] = pd.to_datetime(mrp_df["NEED_BY_DATE"])
        lt_buffer = part_master_df.set_index("PART_ID")["LEAD_TIME"] * lt_buffer_multiplier
        cutoff = pd.to_datetime(pd.Timestamp.today() + pd.to_timedelta(lt_buffer, unit="D"))
        cutoff.name = "CUTOFF_DATE"
        mrp_parts = part_master_df[part_master_df["PLANNING_METHOD"] == "MRP"]["PART_ID"]
    
        in_window = (
            mrp_df[mrp_df["PART_ID"].isin(mrp_parts)]
            .merge(cutoff, left_on="PART_ID", right_index=True)
            .assign(IN_WINDOW=lambda df: df["NEED_BY_DATE"] <= df["CUTOFF_DATE"])
            .groupby("PART_ID")["IN_WINDOW"].any()
        )
    
        what_df = what_df.join(in_window.rename("HAS_MRP_WITHIN_LT")).fillna({"HAS_MRP_WITHIN_LT": False})
        what_df["EXCESS"] = (
            what_df.index.isin(mrp_parts) &
            ~what_df["HAS_MRP_WITHIN_LT"] &
            (what_df["ON_HAND_QUANTITY"] > what_df[["SAFETY_STOCK", "MIN_QTY"]].max(axis=1))
        )
    
        # ROP/Min-Max Excess
        ropmm_parts = part_master_df[part_master_df["PLANNING_METHOD"].isin(["ROP", "MIN_MAX"])]
        what_df["IDEAL_MINIMUM"] = (
            what_df["AVG_DAILY_CONSUMPTION"] * what_df["LEAD_TIME"] * inventory_buffer_multiplier +
            what_df.get("IDEAL_SS", 0)
        )
        eoq = np.sqrt((2 * what_df["TRAILING_CONSUMPTION"] * eoq_order_cost) / eoq_holding_cost_per_unit)
        what_df["EOQ"] = eoq.fillna(0)
        what_df["IDEAL_MAXIMUM"] = what_df["IDEAL_MINIMUM"] + what_df["EOQ"]
    
        ropmm_excess_parts = what_df[
            what_df.index.isin(ropmm_parts["PART_ID"]) &
            (what_df["ON_HAND_QUANTITY"] > what_df["IDEAL_MAXIMUM"])
        ].index
        what_df.loc[ropmm_excess_parts, "EXCESS"] = True
    
        # Inventory turns
        what_df["INVENTORY_TURNS"] = what_df["TRAILING_CONSUMPTION"] / (what_df["ON_HAND_QUANTITY"] + 1)
        avg_turns = what_df["INVENTORY_TURNS"].mean()
    
        # Shortages
        late_pos = po_df[(po_df["STATUS"].str.lower() == "open") & (po_df["RECEIPT_DATE"] > po_df["NEED_BY_DATE"])]["PART_ID"].unique()
        late_wos = wo_df[(wo_df["STATUS"].str.lower() == "open") & (wo_df["COMPLETION_DATE"] > wo_df["DUE_DATE"])]["PART_ID"].unique()
        ropmm_shortages = what_df[
            what_df.index.isin(ropmm_parts["PART_ID"]) &
            (what_df["ON_HAND_QUANTITY"] < what_df["IDEAL_MINIMUM"])
        ].index
        shortage_ids = set(ropmm_shortages).union(late_pos).union(late_wos)
        what_df["SHORTAGE"] = what_df.index.isin(shortage_ids)
    
        shortage_pct = (what_df["SHORTAGE"].sum() / len(what_df)) * 100
        excess_pct = (what_df["EXCESS"].sum() / len(what_df)) * 100
    
        return what_df, shortage_pct, excess_pct, avg_turns

       
    # -------- WHY Metrics -----------
    
    @st.cache_data 
    def calculate_why_metrics(part_master_df, consumption_df, po_df, wo_df):
      
        # PO metrics
        po_df["RECEIPT_DATE"] = pd.to_datetime(po_df["RECEIPT_DATE"])
        po_df["NEED_BY_DATE"] = pd.to_datetime(po_df["NEED_BY_DATE"])
        po_df["LT_DAYS"] = (po_df["RECEIPT_DATE"] - po_df["NEED_BY_DATE"]).dt.days
    
        total_pos = len(po_df)
        late_pos_count = len(po_df[(po_df["STATUS"].str.lower() == "open") & (po_df["RECEIPT_DATE"] > po_df["NEED_BY_DATE"])])
        po_late_percent = (late_pos_count / total_pos) * 100 if total_pos > 0 else 0
    
        closed_po = po_df[po_df["STATUS"].str.lower() == "closed"]
        lt_accuracy_po = closed_po.groupby("PART_ID").agg(actual_lt=("LT_DAYS", "mean")).dropna()
        lt_accuracy_po = lt_accuracy_po.join(part_master_df.set_index("PART_ID")["LEAD_TIME"].rename("erp_lt")).dropna()
        lt_accuracy_po["WITHIN_TOLERANCE"] = (
            abs(lt_accuracy_po["actual_lt"] - lt_accuracy_po["erp_lt"]) / lt_accuracy_po["erp_lt"]
        ) <= lt_tolerance_pct
        po_lead_time_accuracy = lt_accuracy_po["WITHIN_TOLERANCE"].mean() * 100 if not lt_accuracy_po.empty else 0
    
        # WO metrics
        wo_df["COMPLETION_DATE"] = pd.to_datetime(wo_df["COMPLETION_DATE"])
        wo_df["DUE_DATE"] = pd.to_datetime(wo_df["DUE_DATE"])
        wo_df["WO_LT_DAYS"] = (wo_df["COMPLETION_DATE"] - wo_df["DUE_DATE"]).dt.days
    
        total_wos = len(wo_df)
        late_wo_count = len(wo_df[(wo_df["STATUS"].str.lower() == "open") & (wo_df["COMPLETION_DATE"] > wo_df["DUE_DATE"])])
        wo_late_percent = (late_wo_count / total_wos) * 100 if total_wos > 0 else 0
    
        closed_wo = wo_df[wo_df["STATUS"].str.lower() == "closed"]
        lt_accuracy_wo = closed_wo.groupby("PART_ID").agg(actual_lt=("WO_LT_DAYS", "mean")).dropna()
        lt_accuracy_wo = lt_accuracy_wo.join(part_master_df.set_index("PART_ID")["LEAD_TIME"].rename("erp_lt")).dropna()
        lt_accuracy_wo["WITHIN_TOLERANCE"] = (
            abs(lt_accuracy_wo["actual_lt"] - lt_accuracy_wo["erp_lt"]) / lt_accuracy_wo["erp_lt"]
        ) <= lt_tolerance_pct
        wo_lead_time_accuracy = lt_accuracy_wo["WITHIN_TOLERANCE"].mean() * 100 if not lt_accuracy_wo.empty else 0
    
        # Safety stock accuracy
        recent_cutoff = pd.Timestamp.today() - pd.Timedelta(days=trailing_days)
        recent_consumption = consumption_df[consumption_df["TRANSACTION_DATE"] >= recent_cutoff]
        daily_consumption = recent_consumption.groupby(["PART_ID", "TRANSACTION_DATE"])["QUANTITY"].sum().reset_index()
        trailing_avg_daily = daily_consumption.groupby("PART_ID")["QUANTITY"].mean().fillna(0)
    
        std_dev_daily = daily_consumption.groupby("PART_ID")["QUANTITY"].std().fillna(0)
        ss_df = part_master_df.set_index("PART_ID").copy()
        ss_df = ss_df.join(std_dev_daily.rename("STD_DEV_CONSUMPTION"))
        ss_df["IDEAL_SS"] = z_score * ss_df["STD_DEV_CONSUMPTION"] * np.sqrt(ss_df["LEAD_TIME"])
        ss_df = ss_df[~ss_df["IDEAL_SS"].isnull()]
        ss_df["WITHIN_TOLERANCE"] = (
            abs(ss_df["SAFETY_STOCK"] - ss_df["IDEAL_SS"]) / ss_df["IDEAL_SS"] <= ss_tolerance_pct
        )
        valid_ss_parts = len(ss_df["WITHIN_TOLERANCE"])
        compliant_parts = ss_df["WITHIN_TOLERANCE"].sum()
        ss_coverage_percent = (compliant_parts / valid_ss_parts * 100) if valid_ss_parts > 0 else 0
    
        # Scrap rates
        scrap_transactions = consumption_df[consumption_df["TRANSACTION_TYPE"] == scrap_transaction_type]
        consumed_transactions = consumption_df[consumption_df["TRANSACTION_TYPE"].isin(valid_consumption_types)]
    
        total_scrap_by_part = scrap_transactions.groupby("PART_ID")["QUANTITY"].sum()
        total_consumed_by_part = consumed_transactions.groupby("PART_ID")["QUANTITY"].sum()
        scrap_rate_by_part = total_scrap_by_part / (total_scrap_by_part + total_consumed_by_part)
        scrap_rate_by_part = scrap_rate_by_part.fillna(0)
    
        scrap_rate_df = scrap_rate_by_part.rename("AVG_SCRAP_RATE").to_frame()
        valid_scrap_parts = scrap_rate_by_part.count()
        high_scrap_parts = (scrap_rate_by_part > high_scrap_threshold).sum()
        high_scrap_percent = (high_scrap_parts / valid_scrap_parts * 100) if valid_scrap_parts > 0 else 0
    
        # Final part-level WHY detail DataFrame
        why_df = part_master_df.copy().set_index("PART_ID")
        why_df = why_df.join(trailing_avg_daily.rename("AVG_DAILY_CONSUMPTION"))
        why_df = why_df.join(ss_df[["IDEAL_SS", "WITHIN_TOLERANCE"]].rename(columns={"WITHIN_TOLERANCE": "SS_COMPLIANT_PART"}))
        why_df = why_df.join(lt_accuracy_po[["actual_lt", "WITHIN_TOLERANCE"]].rename(columns={"actual_lt": "AVG_PO_LEAD_TIME", "WITHIN_TOLERANCE": "PO_LEAD_TIME_ACCURATE"}))
        why_df = why_df.join(lt_accuracy_wo[["actual_lt", "WITHIN_TOLERANCE"]].rename(columns={"actual_lt": "AVG_WO_LEAD_TIME", "WITHIN_TOLERANCE": "WO_LEAD_TIME_ACCURATE"}))
    
        scrap_rate_df.index = scrap_rate_df.index.astype(why_df.index.dtype)
        why_df = why_df.join(scrap_rate_df)
    
        return why_df, po_late_percent, wo_late_percent, po_lead_time_accuracy, wo_lead_time_accuracy, ss_coverage_percent, high_scrap_percent

    # --- Combine WHAT and WHY part-level data ---

    @st.cache_data
    def build_combined_part_df(what_df, why_df):
        combined_df = what_df.set_index("PART_ID").join(
            why_df, how="outer", lsuffix="_what", rsuffix="_why"
        )
        # Drop duplicate columns if any (e.g., PART_NUMBER or LEAD_TIME)
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
        return combined_df.reset_index()
    
    # --- Column Definitions for AI Interpretation ---
    def define_column_dictionary():
        return {
            "PART_ID": "Unique system identifier for each part",
            "PART_NUMBER": "Visual identifier for each part in ERP and other systems",
            "PLANNING_METHOD": "The method used for planning this part (e.g., MRP, ROP, Min/Max)",
            "LEAD_TIME": "ERP-planned lead time in days",
            "SAFETY_STOCK": "ERP-defined safety stock quantity",
            "MIN_QTY": "ERP-defined Minimum inventory quantity allowed before reorder is triggered",
            "MAX_QTY": "ERP-defined Maximum inventory quantity allowed that constrains order quantity",
            "ON_HAND_QUANTITY": "Current total inventory on hand",
            "TRAILING_CONSUMPTION": "Total consumption over a defined past period of time",
            "AVG_DAILY_CONSUMPTION": "Average daily consumption over a defined past period of time",
            "SHORTAGE": "Flag indicating if the part is in material shortage per MRP",
            "EXCESS": "Flag indicating if the part has excess inventory over needed demand",
            "INVENTORY_TURNS": "How often current inventory turns over, or gets consumed, based on trailing consumption",
            "IDEAL_MINIMUM": "Recommended minimum inventory based on consumption, lead time and statistical calculation",
            "IDEAL_MAXIMUM": "Recommended maximum inventory based on consumption, lead time and statistical calculation",
            "HAS_MRP_WITHIN_LT": "Indicates whether MRP has generated a suggested planned order within the part’s lead time",
            "IDEAL_SS": "Recommended ideal safety stock based on consumption, lead time and statistical calculation",
            "SS_COMPLIANT_PART": "Whether ERP safety stock is within defined buffer % of the ideal value",
            "AVG_PO_LEAD_TIME": "Average actual lead time for closed purchase orders",
            "PO_LEAD_TIME_ACCURATE": "Flag if PO lead time is within defined buffer % of ERP value",
            "AVG_WO_LEAD_TIME": "Average actual lead time for closed work orders",
            "WO_LEAD_TIME_ACCURATE": "Flag if WO lead time is within defined buffer % of ERP value",
            "AVG_SCRAP_RATE": "Average component scrap rate from WIP transactions",
            "ORDER_TYPE": "Indicates whether the row is a PO or WO",
            "ORDER_ID": "Unique identifier for each order (PO or WO)",
            "NEED_BY_DATE": "Required fulfillment date for the order based on MRP",
            "RECEIPT_DATE": "Actual fulfillment/completion date",
            "STATUS": "Current order status, typically 'open' or 'closed'",
            "IS_LATE": "Whether the order was fulfilled after the need date",
            "ERP_LEAD_TIME": "ERP-defined lead time at the time of order",
            "LT_DAYS": "Actual number of days the order took to fulfill",
            "LT_ACCURACY_FLAG": "Whether actual lead time was within the defined tolerance % of ERP lead time",
            "Z_SCORE": "Statistical value used to calculate ideal safety stock. Changes based on the service level you want. Currently set to 1.65 for 95% service level",
            "TRAILING_DAYS": "Number of trailing days used to calculate consumption patterns. Currently set to 90 days",
            "HIGH_SCRAP_THRESHOLD": "Scrap rate threshold used to flag high-scrap parts. Currently set to 10%",
            "EOQ": "Statistical value that is considered the most economical quantity of a part to order each time based on trailing demand, ordering cost, and holding cost",
            "lt_buffer_multiplier": "Multiplier used to extend ERP lead time for MRP suggestion cutoff checks",
            "inventory_buffer_multiplier": "Multiplier used to adjust Min/Max inventory buffer logic",
            "lt_tolerance_pct": "Tolerance percent used to evaluate lead time accuracy (e.g., 10%)",
            "ss_tolerance_pct": "Tolerance percent used to evaluate ERP safety stock accuracy",
            "eoq_order_cost": "Assumed fixed cost per order used to calculate EOQ",
            "eoq_holding_cost_per_unit": "Assumed holding cost per unit used to calculate EOQ",
            "valid_consumption_types": "Transaction types counted as valid consumption when calculating scrap",
            "scrap_transaction_type": "Transaction type used to identify scrap in WIP transactions"
        }

    # Cache dictionary for AI agent access
    column_definitions = define_column_dictionary()
    st.session_state["column_definitions"] = column_definitions

    # NOW insert the function calls here:
    what_part_detail_df, shortage_percent, excess_percent, avg_turns = calculate_what_metrics(
        part_master_df, inventory_df, consumption_df, mrp_df, po_df, wo_df
    )

    why_part_detail_df, po_late_percent, wo_late_percent, po_lead_time_accuracy, wo_lead_time_accuracy, ss_coverage_percent, high_scrap_percent = calculate_why_metrics(
        part_master_df, consumption_df, po_df, wo_df
    )

    combined_part_detail_df = build_combined_part_df(what_part_detail_df, why_part_detail_df)
    st.session_state["combined_part_detail_df"] = combined_part_detail_df

#------------------------------------    
# ------- UI for Results -----------
# -----------------------------------

    # --- UI for WHAT Metrics ---
    st.markdown("📊 WHAT Metrics Results")
    col1, col2, col3 = st.columns(3)
    col1.metric("🔻 % of Parts with Material Shortages", f"{shortage_percent:.1f}%")
    col2.metric("📦 % of Parts with Excess Inventory", f"{excess_percent:.1f}%")
    col3.metric("🔁 Avg Inventory Turns", f"{avg_turns:.1f}")

    with st.expander("📄 Show detailed WHAT part-level table"):
        st.markdown("### Detailed WHAT Metrics Table")
        st.dataframe(what_part_detail_df[[
            "PART_NUMBER", "SHORTAGE", "EXCESS", "INVENTORY_TURNS", 
            "ON_HAND_QUANTITY", "TRAILING_CONSUMPTION", 
            "AVG_DAILY_CONSUMPTION", "SAFETY_STOCK", "MIN_QTY", "MAX_QTY", "LEAD_TIME"
        ]])


    # --- UI for WHY Metrics ---
    st.markdown("🔍 WHY Metrics Results")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("📦 % Late Purchase Orders", f"{po_late_percent:.1f}%")
    col2.metric("🏭 % Late Work Orders", f"{wo_late_percent:.1f}%")
    col3.metric("📈 PO Lead Time Accuracy", f"{po_lead_time_accuracy:.1f}%")
    col4.metric("🛠️ WO Lead Time Accuracy", f"{wo_lead_time_accuracy:.1f}%")
    col5.metric("🛡️ SS Coverage Accuracy", f"{ss_coverage_percent:.1f}%")
    col6.metric("🧯 % of Parts with High Scrap", f"{high_scrap_percent:.1f}%")

    with st.expander("📄 Show detailed WHY part-level table"):
        st.markdown("### Detailed WHY Metrics Table — by Part")    
        st.dataframe(why_part_detail_df.reset_index()[[
            "PART_ID", "PART_NUMBER", "LEAD_TIME", "SAFETY_STOCK", "AVG_DAILY_CONSUMPTION",
            "IDEAL_SS", "SS_COMPLIANT_PART", "PO_LEAD_TIME_ACCURATE", "WO_LEAD_TIME_ACCURATE",
            "AVG_WO_LEAD_TIME", "AVG_PO_LEAD_TIME", "AVG_SCRAP_RATE"
        ]])

    with st.expander("📄 Show detailed WHY order-level table"):
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
            LT_ACCURACY_FLAG=(abs(po_order_df["LT_DAYS"] - po_order_df["ERP_LEAD_TIME"]) / po_order_df["ERP_LEAD_TIME"]) <= lt_tolerance_pct
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
            LT_ACCURACY_FLAG=(abs(wo_order_df["WO_LT_DAYS"] - wo_order_df["ERP_LEAD_TIME"]) / wo_order_df["ERP_LEAD_TIME"]) <= lt_tolerance_pct
        )
    
        all_orders_df = pd.concat([po_order_df, wo_order_df], ignore_index=True)
        all_orders_df = all_orders_df.dropna(subset=["ORDER_ID", "PART_ID", "NEED_BY_DATE", "RECEIPT_DATE"])
        all_orders_df["ORDER_ID"] = all_orders_df["ORDER_ID"].astype(str)
        all_orders_df["PART_ID"] = all_orders_df["PART_ID"].astype(str)
        
        st.session_state["all_orders_df"] = all_orders_df.copy()
    
        st.dataframe(all_orders_df[[
            "ORDER_TYPE", "ORDER_ID", "PART_ID", "NEED_BY_DATE", "RECEIPT_DATE", "STATUS", "IS_LATE",
            "ERP_LEAD_TIME", "LT_DAYS", "LT_ACCURACY_FLAG"
        ]])

