# ----------------------------------------
# SD Audit - All Metrics Grouped by Oracle-Aligned Schema
# ----------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import openai
import json


from gpt_functions import (
    get_material_shortage_summary,
    get_excess_inventory_summary,
    get_inventory_turns_summary,
    get_scrap_rate_summary,
    get_lead_time_accuracy_summary,
    get_ss_accuracy_summary,
    get_late_orders_summary,
    route_gpt_function_call,
    detect_functions_from_prompt,
    # extract_common_parameters,
    apply_universal_filters,
    smart_filter_rank_summary,
    shortage_function_spec,
    excess_function_spec,
    inventory_turns_function_spec,
    scrap_rate_function_spec,
    lead_time_accuracy_function_spec,
    ss_accuracy_function_spec,
    late_orders_function_spec,
    smart_filter_rank_function_spec,
    all_function_specs,
)

openai_api_key = st.secrets["OPENAI_API_KEY"]

# st.write("✅ App loaded successfully. Waiting for file upload.")


def normalize_numeric_columns(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")  # No fillna!
    return df


# Set up page
st.set_page_config(page_title="SD Audit - All Metrics", layout="wide")
st.title("📊 SD Audit - Full Supply & Demand Health Audit")
st.markdown("Upload your Oracle-exported Excel file to analyze.")
#  st.sidebar.markdown("## App Status")
# st.sidebar.write("Waiting for file upload...")


# --- OpenAI API Test Block ---
if openai_api_key:
    from openai import OpenAI as OpenAIClient

    st.markdown("---")
    st.subheader("🧪 OpenAI API Test")

    if st.button("Run Test Query"):
        try:
            client = OpenAIClient(
                api_key=openai_api_key, organization="org-3Va0Uv9V3lCF4EWsBURKlCAG"
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": "What is a safety stock and why is it important?",
                    },
                ],
                temperature=0.5,
            )
            st.success("API call succeeded!")
            st.write(response.choices[0].message.content)
        except Exception as e:
            st.error(f"OpenAI API call failed: {e}")

# Define analysis parameters
trailing_days = 90  # number of days analyzed for consumption in the past
z_score = 1.65  # corresponds to ~95% service level
high_scrap_threshold = 0.10  # Set high scrap threshold (parameterized)
lt_buffer_multiplier = (
    1.1  # % buffer for lead time calculations (used in MRP suggestion window)
)
inventory_buffer_multiplier = 1.1  # % buffer for inventory Min/Max logic
eoq_holding_cost_per_unit = (
    2  # assumed holding cost per unit over the trailing days period
)
eoq_order_cost = 100  # assumed cost to place an order
ss_tolerance_pct = 0.10  # % tolerance allowed for SS accuracy checks
lt_tolerance_pct = 0.10  # % tolerance allowed for lead time accuracy checks
valid_consumption_types = [
    "Backflush",
    "Manual Issue",
]  # WIP transaction types used as valid consumption
scrap_transaction_type = "Scrap"  # WIP transaction type used to identify scrap

# Upload Excel file
uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    st.success("✅ File uploaded successfully")
    # st.write("⏳ Processing...")
    # --- Load All Sheets ---
    xls = pd.ExcelFile(uploaded_file)
    part_master_df = pd.read_excel(xls, sheet_name="PART_MASTER")
    numeric_cols_master = ["LEAD_TIME", "SAFETY_STOCK", "MIN_QTY", "MAX_QTY"]
    part_master_df = normalize_numeric_columns(part_master_df, numeric_cols_master)
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
    def calculate_what_metrics(
        part_master_df, inventory_df, consumption_df, mrp_df, po_df, wo_df
    ):
        inventory_agg = inventory_df.groupby("PART_ID")["ON_HAND_QUANTITY"].sum()
        what_df = part_master_df.set_index("PART_ID").join(inventory_agg).fillna(0)
        what_df["PART_ID"] = what_df.index

        numeric_cols_what = [
            "LEAD_TIME",
            "SAFETY_STOCK",
            "MIN_QTY",
            "MAX_QTY",
            "ON_HAND_QUANTITY",
            "TRAILING_CONSUMPTION",
            "AVG_DAILY_CONSUMPTION",
            "LATE_MRP_NEED_QTY",
            "IDEAL_MINIMUM",
            "IDEAL_MAXIMUM",
            "EOQ",
            "INVENTORY_TURNS",
            "SHORTAGE_QTY",
            "EXCESS_QTY",
        ]
        what_df = normalize_numeric_columns(what_df, numeric_cols_what)

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
        daily = (
            recent.groupby(["PART_ID", "TRANSACTION_DATE"])
            .agg({"QUANTITY": "sum"})
            .reset_index()
        )
        std_dev_daily = daily.groupby("PART_ID")["QUANTITY"].std().fillna(0)
        ss_temp = part_master_df.set_index("PART_ID").copy()
        ss_temp = ss_temp.join(std_dev_daily.rename("STD_DEV_CONSUMPTION"))
        ss_temp["IDEAL_SS"] = (
            z_score * ss_temp["STD_DEV_CONSUMPTION"] * np.sqrt(ss_temp["LEAD_TIME"])
        )
        what_df = what_df.join(ss_temp["IDEAL_SS"])

        # Cutoff Date Calculations
        mrp_df["NEED_BY_DATE"] = pd.to_datetime(mrp_df["NEED_BY_DATE"])
        lt_buffer = (
            part_master_df.set_index("PART_ID")["LEAD_TIME"] * lt_buffer_multiplier
        )
        cutoff = pd.to_datetime(
            pd.Timestamp.today() + pd.to_timedelta(lt_buffer, unit="D")
        )
        cutoff.name = "CUTOFF_DATE"
        mrp_parts = part_master_df[part_master_df["PLANNING_METHOD"] == "MRP"][
            "PART_ID"
        ]

        in_window = (
            mrp_df[mrp_df["PART_ID"].isin(mrp_parts)]
            .merge(cutoff, left_on="PART_ID", right_index=True)
            .assign(IN_WINDOW=lambda df: df["NEED_BY_DATE"] <= df["CUTOFF_DATE"])
            .groupby("PART_ID")["IN_WINDOW"]
            .any()
        )

        # Ideal Min, Max and EOQ Calculations
        ropmm_parts = part_master_df[
            part_master_df["PLANNING_METHOD"].isin(["ROP", "MIN_MAX"])
        ]
        what_df["IDEAL_MINIMUM"] = what_df["AVG_DAILY_CONSUMPTION"] * what_df[
            "LEAD_TIME"
        ] * inventory_buffer_multiplier + what_df.get("IDEAL_SS", 0)
        eoq = np.sqrt(
            (2 * what_df["TRAILING_CONSUMPTION"] * eoq_order_cost)
            / eoq_holding_cost_per_unit
        )
        what_df["EOQ"] = eoq.fillna(0)
        what_df["IDEAL_MAXIMUM"] = what_df["IDEAL_MINIMUM"] + what_df["EOQ"]

        ropmm_excess_parts = what_df[
            what_df.index.isin(ropmm_parts["PART_ID"])
            & (what_df["ON_HAND_QUANTITY"] > what_df["IDEAL_MAXIMUM"])
        ].index
        what_df.loc[ropmm_excess_parts, "EXCESS_YN"] = True

        what_df["INVENTORY_TURNS"] = np.where(
            what_df["ON_HAND_QUANTITY"] > 0,
            what_df["TRAILING_CONSUMPTION"] / what_df["ON_HAND_QUANTITY"],
            np.nan,
        )

        avg_turns = what_df["INVENTORY_TURNS"].mean()

        # --- Create new field: LATE_MRP_NEED_QUANTITY ---
        # Total sum of all late POs for each part
        late_po_qty = (
            po_df[
                (po_df["STATUS"].str.lower() == "open")
                & (po_df["RECEIPT_DATE"] > po_df["NEED_BY_DATE"])
            ]
            .groupby("PART_ID")["ORDER_QUANTITY"]
            .sum()
        )
        # Total sum of all late WOs for each part
        late_wo_qty = (
            wo_df[
                (wo_df["STATUS"].str.lower() == "open")
                & (wo_df["COMPLETION_DATE"] > wo_df["DUE_DATE"])
            ]
            .groupby("PART_ID")["QUANTITY"]
            .sum()
        )
        # Total sum of all late MRP messages within LT for each part
        late_mrp_qty = (
            mrp_df.merge(cutoff, left_on="PART_ID", right_index=True)
            .query("NEED_BY_DATE <= CUTOFF_DATE")
            .groupby("PART_ID")["QUANTITY"]
            .sum()
        )
        # Combine all quantities
        late_total = late_po_qty.add(late_wo_qty, fill_value=0).add(
            late_mrp_qty, fill_value=0
        )
        # Add to what_df
        what_df["LATE_MRP_NEED_QTY"] = (
            what_df.index.to_series().map(late_total).fillna(0).round(2)
        )

        # --- Unified SHORTAGE_QTY calculation (after late MRP need is computed) ---
        def compute_shortage_amount(row):
            if row["PLANNING_METHOD"] == "MRP":
                return max(
                    round(row["LATE_MRP_NEED_QTY"] - row["ON_HAND_QUANTITY"], 2), 0
                )
            elif row["PLANNING_METHOD"] in ["ROP", "MIN_MAX"]:
                return max(round(row["IDEAL_MINIMUM"] - row["ON_HAND_QUANTITY"], 2), 0)
            else:
                return 0

        # --- Defines a Shortage T/F based on Shortage amount
        what_df["SHORTAGE_QTY"] = what_df.apply(compute_shortage_amount, axis=1)
        what_df["SHORTAGE_YN"] = what_df["SHORTAGE_QTY"] > 0

        # --- Unified EXCESS_QTY calculation ---
        def compute_excess_amount(row):
            if row["PLANNING_METHOD"] == "MRP":
                return max(
                    round(row["ON_HAND_QUANTITY"] - row["LATE_MRP_NEED_QTY"], 2), 0
                )
            elif row["PLANNING_METHOD"] in ["ROP", "MIN_MAX"]:
                return max(round(row["ON_HAND_QUANTITY"] - row["IDEAL_MAXIMUM"], 2), 0)
            else:
                return 0

        what_df["EXCESS_QTY"] = what_df.apply(compute_excess_amount, axis=1)
        what_df["EXCESS_YN"] = what_df["EXCESS_QTY"] > 0

        # Calculate final summary metrics
        shortage_pct = (what_df["SHORTAGE_YN"].sum() / len(what_df)) * 100
        excess_pct = (what_df["EXCESS_YN"].sum() / len(what_df)) * 100

        return what_df, shortage_pct, excess_pct, avg_turns

    # -------- WHY Metrics -----------

    @st.cache_data
    def calculate_why_metrics(part_master_df, consumption_df, po_df, wo_df):

        # ✅ Only include closed orders from the start
        closed_po = po_df[po_df["STATUS"].str.lower() == "closed"].copy()
        closed_wo = wo_df[wo_df["STATUS"].str.lower() == "closed"].copy()

        # Convert relevant dates
        closed_po["RECEIPT_DATE"] = pd.to_datetime(closed_po["RECEIPT_DATE"])
        closed_po["NEED_BY_DATE"] = pd.to_datetime(closed_po["NEED_BY_DATE"])
        closed_po["START_DATE"] = pd.to_datetime(closed_po["START_DATE"])
        closed_wo["COMPLETION_DATE"] = pd.to_datetime(closed_wo["COMPLETION_DATE"])
        closed_wo["DUE_DATE"] = pd.to_datetime(closed_wo["DUE_DATE"])
        closed_wo["START_DATE"] = pd.to_datetime(closed_wo["START_DATE"])

        # Step 1–2: Actual Lead Time and Days Late
        closed_po["ACTUAL_LT_DAYS"] = (
            closed_po["RECEIPT_DATE"] - closed_po["START_DATE"]
        ).dt.days
        closed_wo["ACTUAL_LT_DAYS"] = (
            closed_wo["COMPLETION_DATE"] - closed_wo["START_DATE"]
        ).dt.days

        # Step 3: Avg PO/WO/Combined LT
        avg_po_lt = (
            closed_po.groupby("PART_ID")["ACTUAL_LT_DAYS"]
            .mean()
            .rename("AVG_PO_LEAD_TIME")
        )
        avg_wo_lt = (
            closed_wo.groupby("PART_ID")["ACTUAL_LT_DAYS"]
            .mean()
            .rename("AVG_WO_LEAD_TIME")
        )

        lt_counts = pd.concat(
            [
                closed_po.groupby("PART_ID").size().rename("PO_COUNT"),
                closed_wo.groupby("PART_ID").size().rename("WO_COUNT"),
            ],
            axis=1,
        ).fillna(0)
        lt_counts["TOTAL_COUNT"] = lt_counts["PO_COUNT"] + lt_counts["WO_COUNT"]

        combined_lt_df = pd.DataFrame(index=part_master_df["PART_ID"])
        combined_lt_df["ERP_LEAD_TIME"] = part_master_df.set_index("PART_ID")[
            "LEAD_TIME"
        ]
        combined_lt_df = combined_lt_df.join(avg_po_lt).join(avg_wo_lt).join(lt_counts)

        combined_lt_df["COMBINED_LT"] = (
            (combined_lt_df["AVG_PO_LEAD_TIME"] * combined_lt_df["PO_COUNT"])
            + (combined_lt_df["AVG_WO_LEAD_TIME"] * combined_lt_df["WO_COUNT"])
        ) / combined_lt_df["TOTAL_COUNT"].replace(0, np.nan)

        # Step 4: Accuracy % vs ERP
        combined_lt_df["PO_LT_ACCURACY_PCT"] = np.where(
            combined_lt_df["AVG_PO_LEAD_TIME"].notna(),
            (
                1
                - abs(
                    combined_lt_df["AVG_PO_LEAD_TIME"] - combined_lt_df["ERP_LEAD_TIME"]
                )
                / combined_lt_df["ERP_LEAD_TIME"]
            ),
            np.nan,
        )

        combined_lt_df["WO_LT_ACCURACY_PCT"] = np.where(
            combined_lt_df["AVG_WO_LEAD_TIME"].notna(),
            (
                1
                - abs(
                    combined_lt_df["AVG_WO_LEAD_TIME"] - combined_lt_df["ERP_LEAD_TIME"]
                )
                / combined_lt_df["ERP_LEAD_TIME"]
            ),
            np.nan,
        )

        combined_lt_df["COMBINED_LT_ACCURACY_PCT"] = np.where(
            combined_lt_df["COMBINED_LT"].notna(),
            (
                1
                - abs(combined_lt_df["COMBINED_LT"] - combined_lt_df["ERP_LEAD_TIME"])
                / combined_lt_df["ERP_LEAD_TIME"]
            ),
            np.nan,
        )

        # Step 5: Accuracy Flags
        combined_lt_df["PO_LEAD_TIME_ACCURATE"] = (
            combined_lt_df["PO_LT_ACCURACY_PCT"] >= 1 - lt_tolerance_pct
        ).mask(combined_lt_df["PO_LT_ACCURACY_PCT"].isna())

        combined_lt_df["WO_LEAD_TIME_ACCURATE"] = (
            combined_lt_df["WO_LT_ACCURACY_PCT"] >= 1 - lt_tolerance_pct
        ).mask(combined_lt_df["WO_LT_ACCURACY_PCT"].isna())

        combined_lt_df["COMBINED_LT_ACCURATE"] = (
            combined_lt_df["COMBINED_LT_ACCURACY_PCT"] >= 1 - lt_tolerance_pct
        ).mask(combined_lt_df["COMBINED_LT_ACCURACY_PCT"].isna())

        # Step 6: Push part-level metrics to WHY table
        why_df = part_master_df.copy().set_index("PART_ID")
        why_df = why_df.join(
            combined_lt_df[
                [
                    "AVG_PO_LEAD_TIME",
                    "AVG_WO_LEAD_TIME",
                    "COMBINED_LT",
                    "PO_LT_ACCURACY_PCT",
                    "WO_LT_ACCURACY_PCT",
                    "COMBINED_LT_ACCURACY_PCT",
                    "PO_LEAD_TIME_ACCURATE",
                    "WO_LEAD_TIME_ACCURATE",
                    "COMBINED_LT_ACCURATE",
                ]
            ]
        )

        # Safety stock accuracy
        recent_cutoff = pd.Timestamp.today() - pd.Timedelta(days=trailing_days)
        recent_consumption = consumption_df[
            consumption_df["TRANSACTION_DATE"] >= recent_cutoff
        ]
        daily_consumption = (
            recent_consumption.groupby(["PART_ID", "TRANSACTION_DATE"])["QUANTITY"]
            .sum()
            .reset_index()
        )
        trailing_avg_daily = (
            daily_consumption.groupby("PART_ID")["QUANTITY"].mean().fillna(0)
        )

        std_dev_daily = daily_consumption.groupby("PART_ID")["QUANTITY"].std().fillna(0)
        ss_df = part_master_df.set_index("PART_ID").copy()
        ss_df = ss_df.join(std_dev_daily.rename("STD_DEV_CONSUMPTION"))
        ss_df["IDEAL_SS"] = (
            z_score * ss_df["STD_DEV_CONSUMPTION"] * np.sqrt(ss_df["LEAD_TIME"])
        )
        ss_df["SS_DEVIATION_QTY"] = ss_df["SAFETY_STOCK"] - ss_df["IDEAL_SS"]
        ss_df["SS_DEVIATION_PCT"] = ss_df["SS_DEVIATION_QTY"] / ss_df["IDEAL_SS"]
        ss_df["WITHIN_TOLERANCE"] = abs(ss_df["SS_DEVIATION_PCT"]) <= ss_tolerance_pct

        valid_ss_parts = ss_df["IDEAL_SS"].notnull().sum()
        compliant_parts = ss_df["WITHIN_TOLERANCE"].sum()
        ss_accuracy_percent = (
            (compliant_parts / valid_ss_parts * 100) if valid_ss_parts > 0 else 0
        )

        # Scrap rates
        scrap_transactions = consumption_df[
            consumption_df["TRANSACTION_TYPE"] == scrap_transaction_type
        ]
        consumed_transactions = consumption_df[
            consumption_df["TRANSACTION_TYPE"].isin(valid_consumption_types)
        ]

        total_scrap_by_part = scrap_transactions.groupby("PART_ID")["QUANTITY"].sum()
        total_consumed_by_part = consumed_transactions.groupby("PART_ID")[
            "QUANTITY"
        ].sum()
        scrap_rate_by_part = total_scrap_by_part / (
            total_scrap_by_part + total_consumed_by_part
        )
        total_scrap_denominator = (total_scrap_by_part + total_consumed_by_part).rename(
            "SCRAP_DENOMINATOR"
        )
        scrap_rate_by_part = scrap_rate_by_part.replace([np.inf, -np.inf], np.nan)

        scrap_rate_df = scrap_rate_by_part.rename("AVG_SCRAP_RATE").to_frame()
        scrap_rate_df = scrap_rate_df.join(total_scrap_denominator)
        scrap_rate_df["HIGH_SCRAP_PART"] = (
            scrap_rate_df["AVG_SCRAP_RATE"] > high_scrap_threshold
        )

        valid_scrap_parts = scrap_rate_by_part.count()
        high_scrap_parts = (scrap_rate_by_part > high_scrap_threshold).sum()
        high_scrap_percent = (
            (high_scrap_parts / valid_scrap_parts * 100) if valid_scrap_parts > 0 else 0
        )

        # Final part-level WHY detail DataFrame
        why_df = part_master_df.copy().set_index("PART_ID")
        why_df = why_df.join(trailing_avg_daily.rename("AVG_DAILY_CONSUMPTION"))
        why_df = why_df.join(
            ss_df[
                ["IDEAL_SS", "SS_DEVIATION_QTY", "SS_DEVIATION_PCT", "WITHIN_TOLERANCE"]
            ].rename(columns={"WITHIN_TOLERANCE": "SS_COMPLIANT_PART"})
        )
        why_df = why_df.join(
            combined_lt_df[
                [
                    "AVG_PO_LEAD_TIME",
                    "AVG_WO_LEAD_TIME",
                    "COMBINED_LT",
                    "PO_LT_ACCURACY_PCT",
                    "WO_LT_ACCURACY_PCT",
                    "COMBINED_LT_ACCURACY_PCT",
                    "PO_LEAD_TIME_ACCURATE",
                    "WO_LEAD_TIME_ACCURATE",
                    "COMBINED_LT_ACCURATE",
                ]
            ]
        )

        scrap_rate_df.index = scrap_rate_df.index.astype(why_df.index.dtype)
        why_df = why_df.join(scrap_rate_df)

        numeric_cols_why = [
            "LEAD_TIME",
            "SAFETY_STOCK",
            "AVG_DAILY_CONSUMPTION",
            "IDEAL_SS",
            "SS_DEVIATION_QTY",  # 🆕
            "SS_DEVIATION_PCT",
            "AVG_PO_LEAD_TIME",
            "AVG_WO_LEAD_TIME",
            "AVG_SCRAP_RATE",
            "COMBINED_LT",
            "COMBINED_LT_ACCURACY_PCT",
            "PO_LT_ACCURACY_PCT",
            "WO_LT_ACCURACY_PCT",
        ]
        why_df = normalize_numeric_columns(why_df, numeric_cols_why)

        return why_df

    def build_all_orders_df(part_master_df, po_df, wo_df):
        # ✅ Only include closed orders from the start
        po_df = po_df.copy()
        wo_df = wo_df.copy()

        # Convert relevant dates
        po_df["RECEIPT_DATE"] = pd.to_datetime(po_df["RECEIPT_DATE"])
        po_df["NEED_BY_DATE"] = pd.to_datetime(po_df["NEED_BY_DATE"])
        po_df["START_DATE"] = pd.to_datetime(po_df["START_DATE"])

        wo_df["COMPLETION_DATE"] = pd.to_datetime(wo_df["COMPLETION_DATE"])
        wo_df["DUE_DATE"] = pd.to_datetime(wo_df["DUE_DATE"])
        wo_df["START_DATE"] = pd.to_datetime(wo_df["START_DATE"])

        # Actual lead time in days
        po_df["ACTUAL_LT_DAYS"] = np.where(
            po_df["STATUS"].str.lower() == "closed",
            (po_df["RECEIPT_DATE"] - po_df["START_DATE"]).dt.days,
            np.nan,
        )
        wo_df["ACTUAL_LT_DAYS"] = np.where(
            wo_df["STATUS"].str.lower() == "closed",
            (wo_df["COMPLETION_DATE"] - wo_df["START_DATE"]).dt.days,
            np.nan,
        )

        # Days late
        po_df["DAYS_LATE"] = np.where(
            po_df["STATUS"].str.lower() == "closed",
            (po_df["RECEIPT_DATE"] - po_df["NEED_BY_DATE"]).dt.days,
            np.nan,
        )
        wo_df["DAYS_LATE"] = np.where(
            wo_df["STATUS"].str.lower() == "closed",
            (wo_df["COMPLETION_DATE"] - wo_df["DUE_DATE"]).dt.days,
            np.nan,
        )

        # Late flag
        po_df["IS_LATE"] = po_df["DAYS_LATE"] > 0
        wo_df["IS_LATE"] = wo_df["DAYS_LATE"] > 0

        # % late relative to actual lead time
        po_df["PCT_LATE"] = (
            po_df["DAYS_LATE"] / po_df["ACTUAL_LT_DAYS"].replace(0, np.nan)
        ).clip(lower=0)
        wo_df["PCT_LATE"] = (
            wo_df["DAYS_LATE"] / wo_df["ACTUAL_LT_DAYS"].replace(0, np.nan)
        ).clip(lower=0)

        # Standardize order fields
        po_df["ORDER_ID"] = po_df["PO_LINE_ID"]
        po_df["ORDER_TYPE"] = "PO"
        po_df["NEED_BY_DATE"] = po_df["NEED_BY_DATE"]
        po_df["RECEIPT_DATE"] = po_df["RECEIPT_DATE"]

        wo_df["ORDER_ID"] = wo_df["WO_ID"]
        wo_df["ORDER_TYPE"] = "WO"
        wo_df["NEED_BY_DATE"] = wo_df["DUE_DATE"]
        wo_df["RECEIPT_DATE"] = wo_df["COMPLETION_DATE"]

        # Add PLANNING_METHOD from part master
        po_df["PLANNING_METHOD"] = po_df["PART_ID"].map(
            part_master_df.set_index("PART_ID")["PLANNING_METHOD"]
        )
        wo_df["PLANNING_METHOD"] = wo_df["PART_ID"].map(
            part_master_df.set_index("PART_ID")["PLANNING_METHOD"]
        )
        # ERP Lead Time from Part master
        po_df["ERP_LEAD_TIME"] = po_df["PART_ID"].map(
            part_master_df.set_index("PART_ID")["LEAD_TIME"]
        )
        wo_df["ERP_LEAD_TIME"] = wo_df["PART_ID"].map(
            part_master_df.set_index("PART_ID")["LEAD_TIME"]
        )

        # Merge PO + WO
        all_orders_df = pd.concat([po_df, wo_df], ignore_index=True)
        all_orders_df["ORDER_ID"] = all_orders_df["ORDER_ID"].astype(str)
        all_orders_df["PART_ID"] = all_orders_df["PART_ID"].astype(str)

        numeric_cols_orders = ["ACTUAL_LT_DAYS", "DAYS_LATE", "PCT_LATE"]
        all_orders_df = normalize_numeric_columns(all_orders_df, numeric_cols_orders)

        # Save to session state
        st.session_state["all_orders_df"] = all_orders_df.copy()

        return all_orders_df

    # --- Combine WHAT and WHY part-level data ---

    @st.cache_data
    def build_combined_part_df(what_df, why_df):
        combined_df = what_df.set_index("PART_ID").join(
            why_df.drop(
                columns=what_df.columns.intersection(why_df.columns), errors="ignore"
            ),
            how="outer",
        )

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
            "SHORTAGE_YN": "Flag indicating if the part is in material shortage per MRP",
            "EXCESS_YN": "Flag indicating if the part has excess inventory over needed demand",
            "INVENTORY_TURNS": "How often current inventory turns over, or gets consumed, based on trailing consumption",
            "IDEAL_MINIMUM": "Recommended minimum inventory based on consumption, lead time and statistical calculation",
            "IDEAL_MAXIMUM": "Recommended maximum inventory based on consumption, lead time and statistical calculation",
            "HAS_MRP_WITHIN_LT": "Indicates whether MRP has generated a suggested planned order within the part’s lead time",
            "IDEAL_SS": "Recommended ideal safety stock based on consumption, lead time and statistical calculation",
            "SS_COMPLIANT_PART": "Whether ERP safety stock is within defined buffer % of the ideal value",
            "SS_DEVIATION_QTY": "The quantity difference between ERP safety stock and the statistically recommended ideal safety stock.",
            "SS_DEVIATION_PCT": "The percent deviation between ERP safety stock and ideal safety stock. Used to flag accuracy.",
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
            "scrap_transaction_type": "Transaction type used to identify scrap in WIP transactions",
            "LATE_MRP_NEED_QTY": "Total quantity of demand covered by late POs, late WOs, or MRP suggestions within lead time. Represents unfulfilled or misaligned supply.",
            "SHORTAGE_QTY": "The numeric quantity by which a part is short, based on its planning method. For ROP/Min/Max parts, this is Ideal Min - On Hand. For MRP parts, this is Late MRP Need - On Hand.",
            "EXCESS_QTY": "The numeric quantity of excess inventory. For ROP/MinMax, it's On Hand - Ideal Max. For MRP, it's On Hand - Late MRP Need.",
            "EOQ": "The statistically calculated Economic Order Quantity for the part, based on trailing consumption, ordering cost, and holding cost.",
            "INVENTORY_TURNS": "Measures how quickly inventory is consumed, calculated as trailing consumption divided by on-hand inventory.",
            "STD_DEV_CONSUMPTION": "Standard deviation of daily consumption used in ideal SS calculation.",
            "PO_LT_ACCURACY_PCT": "Percent deviation between average actual PO lead time and ERP lead time.",
            "WO_LT_ACCURACY_PCT": "Percent deviation between average actual WO lead time and ERP lead time.",
            "COMBINED_LT": "Weighted average lead time across PO and WO orders, weighted by order volume.",
            "COMBINED_LT_ACCURATE": "Flag indicating if the combined LT is within the allowed tolerance of ERP LT.",
            "COMBINED_LT_ACCURACY_PCT": "Percent deviation between combined LT and ERP LT.",
            "HIGH_SCRAP_PART": "Flag indicating whether the part's scrap rate exceeds the defined high scrap threshold.",
            "SCRAP_DENOMINATOR": "Total quantity used in the scrap rate denominator (scrap + valid consumption).",
        }

    # Cache dictionary for AI agent access
    column_definitions = define_column_dictionary()
    st.session_state["column_definitions"] = column_definitions
    # NOW insert the function calls here:
    what_part_detail_df, shortage_percent, excess_percent, avg_turns = (
        calculate_what_metrics(
            part_master_df, inventory_df, consumption_df, mrp_df, po_df, wo_df
        )
    )

    why_part_detail_df = calculate_why_metrics(
        part_master_df, consumption_df, po_df, wo_df
    )

    all_orders_df = build_all_orders_df(part_master_df, po_df, wo_df)

    combined_part_detail_df = build_combined_part_df(
        what_part_detail_df, why_part_detail_df
    )
    numeric_cols_combined = [
        "LEAD_TIME",
        "SAFETY_STOCK",
        "MIN_QTY",
        "MAX_QTY",
        "ON_HAND_QUANTITY",
        "TRAILING_CONSUMPTION",
        "AVG_DAILY_CONSUMPTION",
        "IDEAL_SS",
        "AVG_PO_LEAD_TIME",
        "AVG_WO_LEAD_TIME",
        "AVG_SCRAP_RATE",
        "IDEAL_MINIMUM",  # Required for GPT shortage amounts
        "IDEAL_MAXIMUM",  # Used in excess logic
        "EOQ",  # Shown in UI for ROP/MinMax sizing
        "INVENTORY_TURNS",  # Displayed and summarized in UI
        "LATE_MRP_NEED_QTY",  # Your new field: must normalize
        "SHORTAGE_QTY",
        "EXCESS_QTY",
        "COMBINED_LT",
        "PO_LT_ACCURACY_PCT",
        "WO_LT_ACCURACY_PCT",
    ]
    combined_part_detail_df = normalize_numeric_columns(
        combined_part_detail_df, numeric_cols_combined
    )

    st.session_state["combined_part_detail_df"] = combined_part_detail_df

    # ------------------------------------
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
        st.dataframe(
            what_part_detail_df[
                [
                    "PART_NUMBER",
                    "PLANNING_METHOD",
                    "INVENTORY_TURNS",
                    "ON_HAND_QUANTITY",
                    "IDEAL_MINIMUM",
                    "LATE_MRP_NEED_QTY",
                    "SHORTAGE_YN",
                    "SHORTAGE_QTY",
                    "EXCESS_YN",
                    "EXCESS_QTY",
                    "TRAILING_CONSUMPTION",
                    "AVG_DAILY_CONSUMPTION",
                    "SAFETY_STOCK",
                    "MIN_QTY",
                    "MAX_QTY",
                    "LEAD_TIME",
                ]
            ]
        )

    po_late_pct = (
        all_orders_df.loc[all_orders_df["ORDER_TYPE"] == "PO", "IS_LATE"]
        .dropna()
        .mean()
        * 100
    )
    wo_late_pct = (
        all_orders_df.loc[all_orders_df["ORDER_TYPE"] == "WO", "IS_LATE"]
        .dropna()
        .mean()
        * 100
    )

    # --- UI for WHY Metrics ---
    st.markdown("🔍 WHY Metrics Results")
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    col1.metric("📦 % Late Purchase Orders", f"{po_late_pct:.1f}%")
    col2.metric("🏭 % Late Work Orders", f"{wo_late_pct:.1f}%")
    col3.metric(
        "📈 PO Lead Time Accuracy",
        f"{(why_part_detail_df['PO_LEAD_TIME_ACCURATE'].dropna().mean() * 100):.1f}%",
    )
    col4.metric(
        "🛠️ WO Lead Time Accuracy",
        f"{(why_part_detail_df['WO_LEAD_TIME_ACCURATE'].dropna().mean() * 100):.1f}%",
    )
    col5.metric(
        "📊 Combined LT Accuracy",
        f"{(why_part_detail_df['COMBINED_LT_ACCURATE'].dropna().mean() * 100):.1f}%",
    )
    col6.metric(
        "📈 % Parts w/ Ideal Safety Stock",
        f"{(why_part_detail_df['SS_COMPLIANT_PART'].dropna().mean() * 100):.1f}%",
    )
    col7.metric(
        "🧯 % of Parts with High Scrap",
        f"{(why_part_detail_df['HIGH_SCRAP_PART'].dropna().mean() * 100):.1f}%",
    )

    with st.expander("📄 Show detailed WHY part-level table"):
        st.markdown("### Detailed WHY Metrics Table — by Part")
        st.dataframe(
            why_part_detail_df.reset_index()[
                [
                    "PART_ID",
                    "PART_NUMBER",
                    "LEAD_TIME",
                    "SAFETY_STOCK",
                    "AVG_DAILY_CONSUMPTION",
                    "IDEAL_SS",
                    "SS_DEVIATION_PCT",
                    "SS_COMPLIANT_PART",
                    "PO_LEAD_TIME_ACCURATE",
                    "PO_LT_ACCURACY_PCT",
                    "WO_LEAD_TIME_ACCURATE",
                    "WO_LT_ACCURACY_PCT",
                    "AVG_WO_LEAD_TIME",
                    "AVG_PO_LEAD_TIME",
                    "COMBINED_LT",
                    "COMBINED_LT_ACCURATE",
                    "COMBINED_LT_ACCURACY_PCT",
                    "AVG_SCRAP_RATE",
                    "HIGH_SCRAP_PART",
                ]
            ]
        )

    with st.expander("📄 Show detailed WHY order-level table"):
        st.markdown("### Detailed WHY Metrics Table — by Order")

        st.dataframe(
            all_orders_df[
                [
                    "ORDER_TYPE",
                    "ORDER_ID",
                    "PART_ID",
                    "PART_NUMBER",
                    "PLANNING_METHOD",
                    "QUANTITY",
                    "NEED_BY_DATE",
                    "RECEIPT_DATE",
                    "START_DATE",
                    "STATUS",
                    "IS_LATE",
                    "ACTUAL_LT_DAYS",
                ]
            ]
        )

# -----------------------------
# ------ GPT Text Blocks ------
# -----------------------------


friendly_name_map = {
    "get_material_shortage_summary": "Material Shortages",
    "get_excess_inventory_summary": "Excess Inventory",
    "get_inventory_turns_summary": "Inventory Turns",
    "get_scrap_rate_summary": "Scrap Rate",
    "get_lead_time_accuracy_summary": "Lead Time Accuracy",
    "get_ss_accuracy_summary": "Safety Stock Accuracy",
    "get_late_orders_summary": "Late Orders",
    "smart_filter_rank_summary": "Filtered & Ranked Summary",  # ✅ new name added
}

# ------ Combined Function Text ------------

with st.expander("💬 Ask GPT: Multi-Metric Supply & Demand Questions"):
    user_prompt = st.text_input(
        "Ask a question about shortages, lead time, scrap, safety stock, or late orders:"
    )

    if st.button("Ask GPT (Multi-Metric)"):
        try:
            function_names, match_type = detect_functions_from_prompt(user_prompt)

            if not function_names:
                st.warning(
                    "GPT did not match your question to any known audit functions."
                )
            else:
                pretty_names = [
                    friendly_name_map.get(fn.get("name"), fn.get("name"))
                    for fn in function_names
                ]

                st.success(f"✅ GPT matched: {', '.join(pretty_names)}")
                st.info(f"🔗 Merge logic: {match_type.upper()} (based on your prompt)")

                results = []
                for fn in function_names:
                    fn_name = fn.get("name")
                    fn_args = fn.get("arguments", {})
                    result = route_gpt_function_call(fn_name, fn_args)

                    if isinstance(result, list):
                        df = pd.DataFrame(result)
                        if not df.empty:
                            # ✅ Drop SORT_VALUE for clean display
                            if "SORT_VALUE" in df.columns:
                                df = df.drop(columns=["SORT_VALUE"])
                            results.append(df)
                    else:
                        st.warning(
                            f"⚠️ {fn_name} did not return a list. Returned: {result}"
                        )

                if len(results) == 0:
                    st.warning("No results returned from any matched function.")
                elif len(results) == 1:
                    st.dataframe(results[0])  # ✅ already cleaned above

                else:
                    if match_type == "intersection":
                        merged_df = results[0]
                        for df in results[1:]:
                            merged_df = pd.merge(
                                merged_df,
                                df,
                                on="PART_NUMBER",
                                how="inner",
                                suffixes=("", "_dup"),
                            )
                        merged_df = merged_df.loc[
                            :, ~merged_df.columns.str.endswith("_dup")
                        ]

                        if merged_df.empty:
                            st.info("⚠️ No parts matched all selected criteria.")
                        else:
                            # ✅ Drop SORT_VALUE again just in case
                            if "SORT_VALUE" in merged_df.columns:
                                merged_df = merged_df.drop(columns=["SORT_VALUE"])

                            st.info(
                                f"📘 Merge Type: INTERSECTION — showing parts that meet all {len(results)} criteria."
                            )
                            st.success(
                                f"✅ {len(merged_df)} parts matched all criteria across {len(results)} functions."
                            )
                            st.dataframe(merged_df)

                    else:  # union
                        combined_df = pd.concat(results, ignore_index=True)
                        combined_df = combined_df.drop_duplicates(subset="PART_NUMBER")

                        # ✅ Drop SORT_VALUE from union result
                        if "SORT_VALUE" in combined_df.columns:
                            combined_df = combined_df.drop(columns=["SORT_VALUE"])

                        st.info(
                            f"📘 Merge Type: UNION — showing all parts that met at least one of the {len(results)} criteria."
                        )
                        st.success(
                            f"✅ {len(combined_df)} unique parts matched at least one of the selected metrics."
                        )
                        st.dataframe(combined_df)

        except Exception as e:
            st.error(f"Function match or execution failed: {e}")
