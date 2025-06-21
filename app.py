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
    po_df = (
        pd.read_excel(uploaded_file, sheet_name="Purchase Orders")
        if "Purchase Orders" in sheet_names
        else None
    )
    wo_df = (
        pd.read_excel(uploaded_file, sheet_name="Work Orders")
        if "Work Orders" in sheet_names
        else None
    )
    forecast_df = (
        pd.read_excel(uploaded_file, sheet_name="Forecast")
        if "Forecast" in sheet_names
        else None
    )
    consumption_df = (
        pd.read_excel(uploaded_file, sheet_name="Consumption")
        if "Consumption" in sheet_names
        else None
    )
    settings_df = (
        pd.read_excel(uploaded_file, sheet_name="Item Settings")
        if "Item Settings" in sheet_names
        else None
    )
    mrp_df = (
        pd.read_excel(uploaded_file, sheet_name="MRP Messages")
        if "MRP Messages" in sheet_names
        else None
    )

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
        "MRP Action Metrics": {},
    }

    # --------------------
    # Metrics Calculations
    # ---------------------

    # -------------------------------
    # Procurement Metrics
    # -------------------------------
     # Filter only closed purchase orders
    closed_pos = po_df[po_df["Status"].str.lower() == "closed"]

    # % Late Purchase Orders
    late_po_count = (closed_pos["Received Date"] > closed_pos["Required Date"]).sum()
    total_closed_pos = len(closed_pos)
    percent_late_po = (
        (late_po_count / total_closed_pos) * 100 if total_closed_pos > 0 else 0
    )

    # PO Lead Time Accuracy (parts with ≥3 closed POs)
    closed_pos["Actual Lead Time"] = (
        closed_pos["Received Date"] - closed_pos["Order Date"]
    ).dt.days
    po_lead_time_accuracy = 0
    valid_po_parts = 0

    for part, group in closed_pos.groupby("Item"):
        if len(group) >= 3:
            avg_actual_lt = group["Actual Lead Time"].mean()
            planned_lt = item_df.loc[item_df["Item"] == part, "Lead Time (Days)"].values
            if len(planned_lt) > 0 and planned_lt[0] > 0:
                if abs(avg_actual_lt - planned_lt[0]) / planned_lt[0] <= 0.10:
                    po_lead_time_accuracy += 1
                valid_po_parts += 1

    po_lead_time_accuracy_percent = (
        (po_lead_time_accuracy / valid_po_parts) * 100 if valid_po_parts > 0 else 0
    )
    # -------------------------------
    # Production Metrics
    # -------------------------------
  # Filter only closed work orders
    closed_wos = wo_df[wo_df["Status"].str.lower() == "closed"]

    # % Late Work Orders
    late_wo_count = (closed_wos["Completed Date"] > closed_wos["Due Date"]).sum()
    total_closed_wos = len(closed_wos)
    percent_late_wo = (
        (late_wo_count / total_closed_wos) * 100 if total_closed_wos > 0 else 0
    )

    # WO Lead Time Accuracy (parts with ≥3 closed WOs)
    closed_wos["Actual Cycle Time"] = (
        closed_wos["Completed Date"] - closed_wos["Start Date"]
    ).dt.days
    wo_lead_time_accuracy = 0
    valid_wo_parts = 0

    for part, group in closed_wos.groupby("Item"):
        if len(group) >= 3:
            avg_actual_cycle = group["Actual Cycle Time"].mean()
            avg_planned_cycle = (
                group["Due Date"].sub(group["Start Date"]).dt.days.mean()
            )
            if avg_planned_cycle > 0:
                if (
                    abs(avg_actual_cycle - avg_planned_cycle) / avg_planned_cycle
                    <= 0.10
                ):
                    wo_lead_time_accuracy += 1
                valid_wo_parts += 1

    wo_lead_time_accuracy_percent = (
        (wo_lead_time_accuracy / valid_wo_parts) * 100 if valid_wo_parts > 0 else 0
    )

    # ----------------------------------------
    # Forecasting Metrics
    # ----------------------------------------
    if forecast_df is not None and consumption_df is not None:
        f = forecast_df.groupby("Item")["Forecast Qty"].sum()
        c = consumption_df.groupby("Item")["Qty Used"].sum()
        fc = pd.concat([f, c], axis=1).dropna()
        fc["Accuracy"] = 1 - abs(fc["Forecast Qty"] - fc["Qty Used"]) / fc["Qty Used"]

        v = forecast_df.groupby(["Item"])["Forecast Qty"].std()

        results["Forecasting Metrics"]["Forecast Accuracy"] = (
            fc["Accuracy"].mean() * 100 if not fc.empty else 0
        )
        results["Forecasting Metrics"]["Forecast Volatility"] = (
            v.mean() if not v.empty else 0
        )

    # ----------------------------------------
    # Planning Parameter Metrics
    # ----------------------------------------
    if settings_df is not None and consumption_df is not None:
        param_df = settings_df.copy()
        c = consumption_df.groupby("Item")["Qty Used"].mean().rename("Avg Usage")
        merged = pd.merge(param_df, c, on="Item", how="inner")

        # Assume Planning Method column exists with "Reorder Point", "Min/Max", "MRP"
        ro_df = merged[
            merged["Planning Method"].isin(["Reorder Point", "Min/Max"])
        ].copy()

        # Statistical recommendations
        ro_df["Reco Reorder Point"] = ro_df["Avg Usage"] * 1.5  # simplified formula
        ro_df["Reco Min"] = ro_df["Avg Usage"]
        ro_df["Reco Max"] = ro_df["Avg Usage"] * 2

        def within_threshold(actual, recommended):
            return abs(actual - recommended) / recommended <= 0.1

        reorder_effective = ro_df.apply(
            lambda r: within_threshold(r["Reorder Point"], r["Reco Reorder Point"]),
            axis=1,
        )
        min_effective = ro_df.apply(
            lambda r: within_threshold(r["Min Qty"], r["Reco Min"]), axis=1
        )
        max_effective = ro_df.apply(
            lambda r: within_threshold(r["Max Qty"], r["Reco Max"]), axis=1
        )

        results["Planning Parameter Metrics"]["Safety Stock Coverage"] = (
            ro_df["Safety Stock"].notna().mean() * 100
        )
        results["Planning Parameter Metrics"]["Min/Max Appropriateness"] = (
            min(min_effective.mean(), max_effective.mean()) * 100
        )
        results["Planning Parameter Metrics"]["Reorder Point Effectiveness"] = (
            reorder_effective.mean() * 100
        )

    # ----------------------------------------
    # MRP Action Metrics
    # ----------------------------------------
    if mrp_df is not None:
        df = mrp_df.copy()
        df["Message Date"] = pd.to_datetime(df["Message Date"])
        df["Action Date"] = pd.to_datetime(df["Action Date"])
        df["Lead Time"] = (df["Action Date"] - df["Message Date"]).dt.days

        results["MRP Action Metrics"]["MRP Message Timeliness"] = (
            df["Lead Time"].mean() if not df.empty else 0
        )

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
                st.metric(
                    label="% Late Purchase Orders", value=f"{po_late_percent:.1f}%"
                )

        with col2:
            if po_lead_time_accuracy is not None:
                st.metric(
                    label="PO Lead Time Accuracy", value=f"{po_lead_time_accuracy:.1f}%"
                )

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
                st.metric(
                    label="WO Lead Time Accuracy", value=f"{wo_lead_time_accuracy:.1f}%"
                )

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
