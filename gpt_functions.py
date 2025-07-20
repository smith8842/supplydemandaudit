import streamlit as st
import pandas as pd
import json
from openai import OpenAI as OpenAIClient

openai_api_key = st.secrets["OPENAI_API_KEY"]

# ✅ Master Column Metadata Dictionary

column_metadata = {
    # Boolean fields
    "SHORTAGE_YN": "bool",
    "EXCESS_YN": "bool",
    "SS_COMPLIANT_PART": "bool",
    "HIGH_SCRAP_PART": "bool",
    "ACCURATE": "bool",
    "IS_LATE": "bool",
    "PO_LEAD_TIME_ACCURATE": "bool",
    "WO_LEAD_TIME_ACCURATE": "bool",
    "COMBINED_LT_ACCURATE": "bool",
    # Numeric fields
    "SHORTAGE_QTY": "numeric",
    "EXCESS_QTY": "numeric",
    "INVENTORY_TURNS": "numeric",
    "AVG_SCRAP_RATE": "numeric",
    "COMBINED_LT_ACCURACY_PCT": "numeric",
    "PO_LT_ACCURACY_PCT": "numeric",
    "WO_LT_ACCURACY_PCT": "numeric",
    "AVG_PO_LEAD_TIME": "numeric",
    "AVG_WO_LEAD_TIME": "numeric",
    "COMBINED_LT": "numeric",
    "ERP_LEAD_TIME": "numeric",
    "AVG_LEAD_TIME": "numeric",
    "SS_DEVIATION_PCT": "numeric",
    "SS_DEVIATION_QTY": "numeric",
    "DAYS_LATE": "numeric",
    "ON_HAND_QUANTITY": "numeric",
    "IDEAL_MINIMUM": "numeric",
    "IDEAL_MAXIMUM": "numeric",
    "SAFETY_STOCK": "numeric",
    "IDEAL_SS": "numeric",
    "TRAILING_CONSUMPTION": "numeric",
    # String / Categorical fields
    "PART_NUMBER": "string",
    "PART_ID": "string",
    "DESCRIPTION": "string",
    "PLANNING_METHOD": "string",
    "MAKE_BUY_CODE": "string",
    "ORDER_TYPE": "string",
    "STATUS": "string",
    "NEED_BY_DATE": "string",
    "RECEIPT_DATE": "string",
    "PLANNER": "string",
    "BUYER": "string",
    "SUPPLIER": "string",
    "COMMODITY": "string",
    "START_DATE": "string",
    "DUE_DATE": "string",
    "COMPLETION_DATE": "string",
    "PCT_LATE": "numeric",
}

# ✅ Column list for GPT guidance (for prompt system messages)
column_list = {
    "bool_columns": [
        "SHORTAGE_YN",
        "EXCESS_YN",
        "SS_COMPLIANT_PART",
        "HIGH_SCRAP_PART",
        "ACCURATE",
        "IS_LATE",
        "PO_LEAD_TIME_ACCURATE",
        "WO_LEAD_TIME_ACCURATE",
        "COMBINED_LT_ACCURATE",
    ],
    "numeric_columns": [
        "SHORTAGE_QTY",
        "EXCESS_QTY",
        "INVENTORY_TURNS",
        "AVG_SCRAP_RATE",
        "COMBINED_LT_ACCURACY_PCT",
        "PO_LT_ACCURACY_PCT",
        "WO_LT_ACCURACY_PCT",
        "AVG_PO_LEAD_TIME",
        "AVG_WO_LEAD_TIME",
        "COMBINED_LT",
        "ERP_LEAD_TIME",
        "AVG_LEAD_TIME",
        "SS_DEVIATION_PCT",
        "SS_DEVIATION_QTY",
        "DAYS_LATE",
        "LT_DAYS",
        "ON_HAND_QUANTITY",
        "IDEAL_MINIMUM",
        "IDEAL_MAXIMUM",
        "SAFETY_STOCK",
        "IDEAL_SS",
        "TRAILING_CONSUMPTION",
        "PCT_LATE",
    ],
    "string_columns": [
        "PART_NUMBER",
        "PART_ID",
        "DESCRIPTION",
        "PLANNING_METHOD",
        "MAKE_BUY_CODE",
        "ORDER_TYPE",
        "STATUS",
        "NEED_BY_DATE",
        "RECEIPT_DATE",
        "PLANNER",
        "BUYER",
        "SUPPLIER",
        "COMMODITY",
        "START_DATE",
        "DUE_DATE",
        "COMPLETION_DATE",
    ],
}

column_list_text = (
    "Available columns:\n"
    f"- Boolean: {', '.join(column_list['bool_columns'])}\n"
    f"- Numeric: {', '.join(column_list['numeric_columns'])}\n"
    f"- String: {', '.join(column_list['string_columns'])}\n"
)

# ----------------------------------------
# GPT Callable Metric Functions
# ----------------------------------------

# --------- Universal Filters --------------


def apply_universal_filters(df: pd.DataFrame, **kwargs) -> pd.DataFrame:

    # Display only non-null filter values
    non_null_filters = {k: v for k, v in kwargs.items() if v is not None}
    st.info(f"🌐 Universal filters applied:\n{json.dumps(non_null_filters)}")

    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    # PART_NUMBER
    if "part_number" in kwargs and "PART_NUMBER" in df.columns:
        df = df[df["PART_NUMBER"] == kwargs["part_number"]]

    # PLANNING_METHOD
    if "planning_method" in kwargs and "PLANNING_METHOD" in df.columns:
        pm = kwargs["planning_method"]
        if isinstance(pm, dict):
            pm = list(pm.values())
        elif isinstance(pm, str):
            pm = [pm]
        df = df[df["PLANNING_METHOD"].isin(pm)]

    # # ACCURACY FILTER (can be boolean or %)
    # if "accuracy_filter" in kwargs and "ACCURATE" in df.columns:
    #     val = kwargs["accuracy_filter"]
    #     if val == "accurate":
    #         df = df[df["ACCURATE"] == True]
    #     elif val == "inaccurate":
    #         df = df[df["ACCURATE"] == False]

    # COMPLIANT ONLY
    if kwargs.get("compliant_only") and "SS_COMPLIANT_PART" in df.columns:
        df = df[df["SS_COMPLIANT_PART"] == True]

    # HIGH ONLY
    if kwargs.get("high_only") and "HIGH_SCRAP_PART" in df.columns:
        df = df[df["HIGH_SCRAP_PART"] == True]

    # PLACEHOLDER FIELDS
    for field in ["planner", "buyer", "supplier", "make_or_buy", "commodity"]:
        col = field.upper()
        if field in kwargs and col in df.columns:
            df = df[df[col] == kwargs[field]]

    return df


# ------------ SHORTAGE CALL ---------------
def get_material_shortage_summary(
    top_n: int = None,
    part_number: str = None,
    planning_method: str | list[str] = None,
    accuracy_filter: str = None,
    compliant_only: bool = None,
    high_only: bool = None,
    planner: str = None,
    buyer: str = None,
    supplier: str = None,
    make_or_buy: str = None,
    commodity: str = None,
):
    """
    GPT-callable: Returns a filtered list of parts with material shortages.

    Parameters
    ----------
    top_n : int
        Max number of shortage results to return (sorted by total late MRP need).
    part_number : str
        If provided, only return shortage status for this part number.
    planning_method : str or list of str
        If specified, restrict to one or more planning methods (e.g., "MRP", "ROP").

    Returns
    -------
    List[Dict]
        A list of shortage records including key audit fields.
    """
    df = st.session_state.get("combined_part_detail_df", pd.DataFrame())
    if df.empty or "SHORTAGE_YN" not in df.columns:
        return []

    # 🔁 Apply universal filters using kwargs (same as your parameter names)
    args = {k: v for k, v in locals().items() if k not in ["df", "args"]}

    df = apply_universal_filters(df, **args)

    # Now filter to just shortage parts
    df_short = df[df["SHORTAGE_YN"] == True].copy()

    # Only show meaningful shortage values
    df_short = df_short[df_short["SHORTAGE_QTY"].notnull()]
    df_short = df_short[df_short["SHORTAGE_QTY"] > 0]
    df_short = df_short.sort_values(by="SHORTAGE_QTY", ascending=False)

    if top_n is not None:
        df_short = df_short.head(top_n)

    cols_to_return = [
        "PART_NUMBER",
        "PLANNING_METHOD",
        "ON_HAND_QUANTITY",
        "IDEAL_MINIMUM",
        "LATE_MRP_NEED_QTY",
        "SHORTAGE_QTY",
    ]

    missing_cols = set(cols_to_return) - set(df_short.columns)
    if missing_cols:
        st.warning(
            f"⚠️ Some expected columns are missing from the shortage table: {missing_cols}"
        )

    existing_cols = [col for col in cols_to_return if col in df_short.columns]
    return df_short[existing_cols].to_dict("records")


# --------- EXCESS CALL -------------


def get_excess_inventory_summary(
    top_n: int = None,
    part_number: str = None,
    planning_method: str | list[str] = None,
    accuracy_filter: str = None,
    compliant_only: bool = None,
    high_only: bool = None,
    planner: str = None,
    buyer: str = None,
    supplier: str = None,
    make_or_buy: str = None,
    commodity: str = None,
):
    """
    GPT-callable: Returns a filtered list of parts with excess inventory.

    Parameters
    ----------
    top_n : int
        Max number of excess results to return (sorted by excess amount).
    part_number : str
        If provided, only return excess status for this part number.
    planning_method : str or list of str
        If specified, restrict to one or more planning methods (e.g., "MRP", "ROP").

    Returns
    -------
    List[Dict]
        A list of excess inventory records including key audit fields.
    """
    df = st.session_state.get("combined_part_detail_df", pd.DataFrame())
    if df.empty or "EXCESS_YN" not in df.columns:
        return []

    # 🔁 Apply universal filters using kwargs (same as your parameter names)
    args = {k: v for k, v in locals().items() if k not in ["df", "args"]}

    df = apply_universal_filters(df, **args)

    # ✅ Now filter to just excess parts
    df_excess = df[df["EXCESS_YN"] == True].copy()

    # ✅ Continue with post-filtering logic
    df_excess = df_excess[df_excess["EXCESS_QTY"].notnull()]
    df_excess = df_excess[df_excess["EXCESS_QTY"] > 0]
    df_excess = df_excess.sort_values(by="EXCESS_QTY", ascending=False)

    if top_n is not None:
        df_excess = df_excess.head(top_n)

    cols_to_return = [
        "PART_NUMBER",
        "PLANNING_METHOD",
        "ON_HAND_QUANTITY",
        "IDEAL_MAXIMUM",
        "LATE_MRP_NEED_QTY",
        "EXCESS_QTY",
    ]

    existing_cols = [col for col in cols_to_return if col in df_excess.columns]
    return df_excess[existing_cols].to_dict("records")


# --------- INVENTORY TURNS CALL -------------


def get_inventory_turns_summary(
    top_n: int = None,
    part_number: str = None,
    planning_method: str | list[str] = None,
    accuracy_filter: str = None,
    compliant_only: bool = None,
    high_only: bool = None,
    planner: str = None,
    buyer: str = None,
    supplier: str = None,
    make_or_buy: str = None,
    commodity: str = None,
):
    """
    GPT-callable: Returns a filtered list of parts with inventory turns data.

    Parameters
    ----------
    top_n : int
        Max number of parts to return (sorted by lowest turns).
    part_number : str
        If provided, only return data for this part number.
    planning_method : str or list of str
        If specified, restrict to one or more planning methods (e.g., "MRP", "ROP").

    Returns
    -------
    List[Dict]
        A list of inventory turn records.
    """
    df = st.session_state.get("combined_part_detail_df", pd.DataFrame())
    if df.empty or "INVENTORY_TURNS" not in df.columns:
        return []

    # 🔁 Apply universal filters using kwargs (same as your parameter names)
    args = {k: v for k, v in locals().items() if k not in ["df", "args"]}
    df = apply_universal_filters(df, **args)

    df_turns = df.copy()

    df_turns = df_turns[df_turns["INVENTORY_TURNS"].notnull()]
    df_turns = df_turns.sort_values(by="INVENTORY_TURNS", ascending=True)

    if top_n:
        df_turns = df_turns.head(top_n)

    cols_to_return = [
        "PART_NUMBER",
        "PLANNING_METHOD",
        "ON_HAND_QUANTITY",
        "TRAILING_CONSUMPTION",
        "INVENTORY_TURNS",
    ]

    existing_cols = [col for col in cols_to_return if col in df_turns.columns]
    return df_turns[existing_cols].to_dict("records")


# --------- SCRAP RATE CALL -------------


def get_scrap_rate_summary(
    top_n: int = None,
    part_number: str = None,
    planning_method: str | list[str] = None,
    accuracy_filter: str = None,
    compliant_only: bool = None,
    high_only: bool = None,
    planner: str = None,
    buyer: str = None,
    supplier: str = None,
    make_or_buy: str = None,
    commodity: str = None,
):
    """
    GPT-callable: Returns a filtered list of parts with scrap rate data.

    Parameters
    ----------
    top_n : int
        Max number of parts to return (sorted by highest scrap rate).
    part_number : str
        If provided, only return scrap rate for this part.
    high_only : bool
        If True, only return parts with scrap rates above defined threshold.

    Returns
    -------
    List[Dict]
        A list of parts with their average scrap rates.
    """
    df = st.session_state.get("combined_part_detail_df", pd.DataFrame())
    if df.empty or "AVG_SCRAP_RATE" not in df.columns:
        return []

    # 🔁 Apply universal filters using kwargs (same as your parameter names)
    args = {k: v for k, v in locals().items() if k not in ["df", "args"]}
    df = apply_universal_filters(df, **args)

    df_scrap = df.copy()
    # ✅ Remove parts with no valid denominator (i.e., no consumption or scrap activity)
    if "SCRAP_DENOMINATOR" in df_scrap.columns:
        df_scrap = df_scrap[df_scrap["AVG_SCRAP_RATE"].notnull()]
        df_scrap = df_scrap[df_scrap["SCRAP_DENOMINATOR"] > 0]

    # Determine sort direction: high = bad, so default to ascending unless high_only is True
    sort_desc = locals().get("high_only", False)
    df_scrap = df_scrap.sort_values(by="AVG_SCRAP_RATE", ascending=not sort_desc)

    if top_n:
        df_scrap = df_scrap.head(top_n)

    cols_to_return = [
        "PART_NUMBER",
        "AVG_SCRAP_RATE",
        "PLANNING_METHOD",
    ]
    return df_scrap[cols_to_return].to_dict("records")


# -------- Lead Time Accuracy ----------------
def get_lead_time_accuracy_summary(
    top_n: int = None,
    part_number: str = None,
    planning_method: str | list[str] = None,
    order_type: str = None,
    accuracy_filter: str = None,
    compliant_only: bool = None,
    planner: str = None,
    buyer: str = None,
    supplier: str = None,
    make_or_buy: str = None,
    commodity: str = None,
):
    """
    GPT-callable: Returns a filtered list of parts with lead time accuracy data.

    Parameters
    ----------
    top_n : int
        Max number of parts to return.
    part_number : str
        If provided, only return data for this part.
    order_type : str
        PO, WO, or Combined (affects which accuracy/avg_lt column is used).
    accuracy_filter : str
        If provided, filters based on accuracy level.

    Returns
    -------
    List[Dict]
        A list of lead time accuracy records.
    """

    df = st.session_state.get("combined_part_detail_df", pd.DataFrame())
    if df.empty:
        return []

    # 🆕 Assign accuracy columns based on order_type
    if order_type == "PO":
        df = df.copy()
        df["ACCURATE"] = df.get("PO_LEAD_TIME_ACCURATE", None)
        df["AVG_LEAD_TIME"] = df.get("AVG_PO_LEAD_TIME", None)
        df["ACCURACY"] = df.get("PO_LT_ACCURACY_PCT", None)
    elif order_type == "WO":
        df = df.copy()
        df["ACCURATE"] = df.get("WO_LEAD_TIME_ACCURATE", None)
        df["AVG_LEAD_TIME"] = df.get("AVG_WO_LEAD_TIME", None)
        df["ACCURACY"] = df.get("WO_LT_ACCURACY_PCT", None)
    else:
        df = df.copy()
        df["ACCURATE"] = df.get("COMBINED_LT_ACCURATE", None)
        df["AVG_LEAD_TIME"] = df.get("COMBINED_LT", None)
        df["ACCURACY"] = df.get("COMBINED_LT_ACCURACY_PCT", None)

    df = df[df["ACCURACY"].notnull()]  # Required for sorting + completeness

    # ✅ Universal filters
    args = {k: v for k, v in locals().items() if k != "df"}
    df = apply_universal_filters(df, **args)

    # ✅ Sort only if no accuracy_filter is specified (i.e. it's a ranking prompt)
    if accuracy_filter in ["accurate", "inaccurate"]:
        # Already filtered by flag in universal filters
        sort_desc = accuracy_filter == "accurate"
        df = df.sort_values(by="AVG_LEAD_TIME", ascending=not sort_desc)
    else:
        df = df[df["ACCURACY"].notnull()]
        df = df.sort_values(by="ACCURACY", ascending=True)

    if top_n:
        df = df.head(top_n)

    return df[
        [
            "PART_NUMBER",
            "ORDER_TYPE",
            "ERP_LEAD_TIME",
            "AVG_LEAD_TIME",
            "ACCURATE",
            "ACCURACY",
        ]
    ].to_dict("records")


# ------- Safety Stock Accuracy ---------


def get_ss_accuracy_summary(
    top_n: int = None,
    part_number: str = None,
    planning_method: str | list[str] = None,
    accuracy_filter: str = None,
    compliant_only: bool = None,
    planner: str = None,
    buyer: str = None,
    supplier: str = None,
    make_or_buy: str = None,
    commodity: str = None,
):
    """
    GPT-callable: Returns a list of parts with Safety Stock compliance accuracy.

    Parameters
    ----------
    top_n : int
        Max number of parts to return, sorted by largest deviation.
    part_number : str
        If provided, return data for that specific part.
    compliant_only : bool
        If True, only return parts where SS_COMPLIANT_PART == True.

    Returns
    -------
    List[Dict]
        A list of parts with ERP vs ideal safety stock comparison.
    """
    df = st.session_state.get("combined_part_detail_df", pd.DataFrame())
    if df.empty or "SS_COMPLIANT_PART" not in df.columns:
        return []

    # 🔁 Apply universal filters using kwargs (same as your parameter names)
    args = {k: v for k, v in locals().items() if k not in ["df", "args"]}
    df = apply_universal_filters(df, **args)

    df_ss = df.copy()
    df_ss["ACCURACY"] = df_ss["SS_DEVIATION_PCT"]

    # Sort direction: best = lowest deviation if asking for accuracy
    sort_desc = locals().get("accuracy_filter") != "accurate"
    df_ss = df_ss[df_ss["SS_DEVIATION_PCT"].notnull()]
    df_ss = df_ss.assign(SS_ABS_DEV=abs(df_ss["SS_DEVIATION_PCT"]))
    df_ss = df_ss.sort_values(by="SS_ABS_DEV", ascending=not sort_desc)

    if top_n:
        df_ss = df_ss.head(top_n)

    return df_ss[
        [
            "PART_NUMBER",
            "PLANNING_METHOD",
            "SAFETY_STOCK",
            "IDEAL_SS",
            "SS_DEVIATION_QTY",
            "SS_DEVIATION_PCT",
            "SS_COMPLIANT_PART",
        ]
    ].to_dict("records")


# --------- Late Orders ------------


def get_late_orders_summary(
    filters: dict = None,
    top_n: int = None,
    part_number: str = None,
    order_type: str = None,
    planning_method: str | list[str] = None,
    late_only: bool = None,
    accuracy_filter: str = None,
    compliant_only: bool = None,
    planner: str = None,
    buyer: str = None,
    supplier: str = None,
    make_or_buy: str = None,
    commodity: str = None,
):
    """
    GPT-callable: Returns late PO and WO orders from the all_orders_df table.

    Parameters
    ----------
    top_n : int
        Max number of orders to return (sorted by most late days).
    order_type : str
        "PO" or "WO", optional filter.
    part_number : str
        If provided, filters to a specific part number.
    late_only : bool
        If True, returns only late orders.

    Returns
    -------
    List[Dict]
        A list of order-level records with lateness info.
    """
    df = st.session_state.get("all_orders_df", pd.DataFrame())
    if df.empty:
        return []

    # 🔁 Apply universal filters using kwargs (same as your parameter names)
    args = {k: v for k, v in locals().items() if k not in ["df", "args"]}
    df = apply_universal_filters(df, **args)

    df_filtered = df.copy()

    # 🧠 Apply dynamic filters from GPT 'filters' dict
    if filters:
        for col, val in filters.items():
            matched_cols = [c for c in df_filtered.columns if c.upper() == col.upper()]
            if matched_cols:
                df_filtered = df_filtered[df_filtered[matched_cols[0]] == val]
            else:
                st.warning(f"⚠️ Filter field '{col}' not found in dataframe — skipping.")

    if order_type:
        df_filtered = df_filtered[df_filtered["ORDER_TYPE"] == order_type]

    if late_only:
        df_filtered = df_filtered[df_filtered["IS_LATE"] == True]

    df_filtered = df_filtered[
        df_filtered["RECEIPT_DATE"].notnull() & df_filtered["NEED_BY_DATE"].notnull()
    ]

    df_filtered = df_filtered.sort_values(by="DAYS_LATE", ascending=False)

    if top_n:
        df_filtered = df_filtered.head(top_n)

    return df_filtered[
        [
            "ORDER_TYPE",
            "ORDER_ID",
            "PART_NUMBER",
            "PLANNING_METHOD",
            "STATUS",
            "START_DATE",
            "NEED_BY_DATE",
            "RECEIPT_DATE",
            "IS_LATE",
            "DAYS_LATE",
            "PCT_LATE",
            "ERP_LEAD_TIME",
            "ACTUAL_LT_DAYS",
        ]
    ].to_dict("records")


# ---------- Combined Function Call ---------

# Accepts in the user prompt, identifies function to be called, sorts and filter arguments to be used, and other details needed to fun the later functions

import re


def detect_functions_from_prompt(prompt: str):
    """
    Uses GPT to identify audit functions AND whether the user intends 'intersection' (AND) or 'union' (OR) logic.
    Returns:
        - A list of function names (e.g., ["get_lead_time_accuracy_summary", ...])
        - A match type string: "intersection" or "union"
    """
    client = OpenAIClient(api_key=openai_api_key)

    function_descriptions = "\n".join(
        f"- {spec['name']}: {spec['description']}" for spec in all_function_specs
    )

    system_prompt = """
    You are an expert assistant helping users analyze supply and demand performance in a manufacturing ERP environment. Your job is to map user questions to one or more function calls from the available audit metrics.
    
    Each function provides a specific type of supply or demand insight — such as shortages, excess inventory, scrap rates, safety stock accuracy, or lead time issues. Users may ask vague or compound questions. Handle these cases by:
    
    1. Matching every function that reasonably applies.
    2. Extracting relevant arguments like part_number, order_type, top_n, planner, planning_method, or accuracy_filter when mentioned.
    3. Assuming defaults when needed (e.g. if part_number is not specified, return top rows).
    4. Supporting logical operators (AND/OR). If the user asks about multiple concerns, match all applicable functions.
    5. If the user mentions any filtering criteria (e.g., "ROP", "planner = David", "only shortages", "just POs"), place those inside a `filters` dictionary:
        Example: 
        "filters": {
            "PLANNING_METHOD": "ROP",
            "PLANNER": "David",
            "SHORTAGE_YN": true
        }
    6. If the user prompt suggests ranking or ordering (e.g., "top 5", "worst shortages", "best accuracy"), include a `sort` dictionary:
        Example:
        "sort": {
            "field": "SS_DEVIATION_PCT",
            "ascending": false
        }
    7. Only assign numeric columns to the `sort.field` value — like SHORTAGE_QTY, AVG_SCRAP_RATE, INVENTORY_TURNS, etc.
    8. Do NOT assign boolean fields like SHORTAGE_YN or EXCESS_YN to the sort field. These can only be used in the `filters` dictionary.
    9. Interpret ranking language:
        - "best", "highest", "top", "most" → ascending = false
        - "worst", "lowest", "bottom", "least" → ascending = true
    10. There are certain fields where ranking language would have an inverse rank order. Currently those including Shortages, Excess, Safety Stock Deviations, Scrap Rates, and Days Late. For example - the worst safety stock deviation would be "ascending = false"
    11. If the user adds ranking language in conjunction with terms that you intepret as a column (like the phrase "worst shortages"), then the user is looking for a sort and ranking on the SHORTAGE_QTY field, not a filter on the SHORTAGE_YN field.
    12. The get_late_orders_summary function is looking at data about orders. So, it should only ever be chosen if the user asks for order-centric data. For example, "Show me all late orders." But something like "show me all parts with late orders," the prompt is asking for part-centric data.
    You MUST return a valid JSON dictionary in the following format:
    {
      "functions": [
        {
          "name": "smart_filter_rank_summary",
          "arguments": {
            "filters": {
              "SHORTAGE_YN": true,
              "PLANNING_METHOD": "ROP"
            },
            "sort": {
              "field": "SS_DEVIATION_PCT",
              "ascending": false
            },
            "top_n": 5
          }
        }
      ],
      "match_type": "intersection"
    }
    If no functions apply to the user prompt, return:
    {
      "functions": [],
      "match_type": "intersection"
    }
    """.strip()

    user_prompt = (
        f"Available functions:\n{function_descriptions}\n\n"
        f"{column_list_text}\n\n"
        f"User prompt:\n{prompt}\n\n"
        "Return a JSON dictionary with keys 'functions' and 'match_type'."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Remove markdown wrapping if present
    if raw.startswith("```"):
        raw = raw.strip("`").strip("json").strip()

    try:
        parsed = json.loads(raw)
        functions = parsed.get("functions", [])
        match_type = parsed.get("match_type", "intersection").lower()
        return functions, match_type
    except Exception as e:
        print(f"⚠️ Failed to parse GPT response: {e}")
        return [], "intersection"


# # -------- Prompt Parameter Extraction ---------------


# def extract_common_parameters(prompt: str) -> dict:
#     prompt = prompt.lower()
#     params = {}

#     if "top 5" in prompt:
#         params["top_n"] = 5
#     elif "top 10" in prompt:
#         params["top_n"] = 10
#     elif "top 3" in prompt:
#         params["top_n"] = 3
#     elif "top 7" in prompt:
#         params["top_n"] = 7

#     ranking_adjectives = ["most", "top", "best", "worst", "lowest", "highest"]
#     is_ranking_prompt = any(x in prompt for x in ranking_adjectives)

#     # Accuracy logic: ranking vs filtering
#     if "accurate" in prompt or "inaccurate" in prompt:
#         if "inaccurate" in prompt:
#             params["accuracy_filter"] = "inaccurate"
#         elif "accurate" in prompt:
#             params["accuracy_filter"] = None if is_ranking_prompt else "accurate"
#     elif "worst" in prompt or "lowest" in prompt:
#         params["accuracy_filter"] = "inaccurate"
#     elif "best" in prompt or "highest" in prompt:
#         params["accuracy_filter"] = None

#     # Safety Stock compliant logic
#     if "compliant" in prompt:
#         if is_ranking_prompt:
#             params["compliant_only"] = None
#         else:
#             params["compliant_only"] = True

#     # High Scrap logic
#     if "high scrap" in prompt:
#         if is_ranking_prompt:
#             params["high_only"] = None
#         else:
#             params["high_only"] = True

#     # Late Order logic
#     if "late" in prompt:
#         if is_ranking_prompt:
#             params["late_only"] = None
#         else:
#             params["late_only"] = True

#     # Shortage logic
#     if "shortage" in prompt:
#         params["filter_field"] = None if is_ranking_prompt else "SHORTAGE_YN"

#     # Excess logic
#     if "excess" in prompt:
#         params["filter_field"] = None if is_ranking_prompt else "EXCESS_YN"

#     # Order Type
#     if "po" in prompt:
#         params["order_type"] = "PO"
#     elif "wo" in prompt:
#         params["order_type"] = "WO"

#     # Planning method
#     if "min/max" in prompt or "min max" in prompt or "min-max" in prompt:
#         params["planning_method"] = "MIN_MAX"
#     elif "mrp" in prompt:
#         params["planning_method"] = "MRP"
#     elif "rop" in prompt:
#         params["planning_method"] = "ROP"

#     st.write("🧪 Extracted params from prompt:", params)

#     return params


# ---------- Routed Function Call -----------


def route_gpt_function_call(name: str, args: dict):
    """
    Routes GPT function call name to the corresponding Python function.
    Filters args based on function signature.
    """
    router = {
        # "get_material_shortage_summary": get_material_shortage_summary,
        # "get_excess_inventory_summary": get_excess_inventory_summary,
        # "get_inventory_turns_summary": get_inventory_turns_summary,
        # "get_scrap_rate_summary": get_scrap_rate_summary,
        # "get_lead_time_accuracy_summary": get_lead_time_accuracy_summary,
        # "get_ss_accuracy_summary": get_ss_accuracy_summary,
        "get_late_orders_summary": get_late_orders_summary,
        "smart_filter_rank_summary": smart_filter_rank_summary,
    }

    if name not in router:
        return f"⚠️ No matching function found for: {name}"

    st.write("🧠 GPT raw function name + args:", name, args)

    fn = router[name]

    # Filter args based on the function's accepted parameters
    import inspect

    sig = inspect.signature(fn)
    accepted_args = {k: v for k, v in args.items() if k in sig.parameters}

    st.write("📦 Final accepted args to function:", accepted_args)

    # 👇 Debug display in UI
    import json

    st.info(
        "🔍 Function `{}` called with parameters:\n\n{}".format(
            name, json.dumps(accepted_args, indent=2)
        )
    )

    return fn(**accepted_args)


# ------ SMART FILTER ROUTER ---------


def smart_filter_rank_summary(
    filters: dict = None,
    sort: dict = None,
    top_n: int = 5,
    return_fields=None,
    part_number: str = None,
    planning_method: str | list[str] = None,
    accuracy_filter: str = None,
    compliant_only: bool = None,
    high_only: bool = None,
    planner: str = None,
    buyer: str = None,
    supplier: str = None,
    make_or_buy: str = None,
    commodity: str = None,
    order_type: str = None,
):
    """
    Dynamically filters and ranks combined_part_detail_df based on provided fields.

    Parameters
    ----------
    filters : dict
        Dictionary of column-value pairs to filter the data.
    sort : dict
        Dictionary with 'field' and 'ascending' keys for sorting.
    top_n : int
        Number of rows to return after sorting.

    Returns
    -------
    List[Dict]
        Filtered and sorted part records.
    """

    df = st.session_state.get("combined_part_detail_df", pd.DataFrame())

    args = {
        k: v for k, v in locals().items() if k not in ["df", "args", "filters", "sort"]
    }

    st.info("🔍 Smart filter args:\n" + json.dumps(args, indent=2))

    df = apply_universal_filters(df, **args)

    # # 🔁 Inject lead time accuracy columns based on ORDER_TYPE
    # order_type = args.get("order_type", None)

    # if order_type == "PO":
    #     df["ACCURATE"] = df.get("PO_LEAD_TIME_ACCURATE")
    #     df["AVG_LEAD_TIME"] = df.get("AVG_PO_LEAD_TIME")
    #     df["ACCURACY"] = df.get("PO_LT_ACCURACY_PCT")
    # elif order_type == "WO":
    #     df["ACCURATE"] = df.get("WO_LEAD_TIME_ACCURATE")
    #     df["AVG_LEAD_TIME"] = df.get("AVG_WO_LEAD_TIME")
    #     df["ACCURACY"] = df.get("WO_LT_ACCURACY_PCT")
    # else:
    #     df["ACCURATE"] = df.get("COMBINED_LT_ACCURATE")
    #     df["AVG_LEAD_TIME"] = df.get("COMBINED_LT")
    #     df["ACCURACY"] = df.get("COMBINED_LT_ACCURACY_PCT")

    # 🧠 Apply dynamic filters
    if filters:
        for col, val in filters.items():
            matched_cols = [c for c in df.columns if c.upper() == col.upper()]
            if matched_cols:
                df = df[df[matched_cols[0]] == val]
            else:
                st.warning(f"⚠️ Filter field '{col}' not found in dataframe — skipping.")

    # 🧠 Apply dynamic sort fields
    if not sort or "field" not in sort:
        st.info("ℹ️ No sort field provided — returning unranked filtered results.")
        return df.head(top_n).to_dict("records") if top_n else df.to_dict("records")

    sort_field = sort["field"]
    ascending = sort.get("ascending", False)

    # 🧠 Case-insensitive column match for sort field
    matched_sort_cols = [c for c in df.columns if c.upper() == sort_field.upper()]
    if not matched_sort_cols:
        st.warning(f"⚠️ Sort field '{sort_field}' not found in dataframe — skipping.")
        return []

    sort_field = matched_sort_cols[0]

    if not pd.api.types.is_numeric_dtype(df[sort_field]):
        st.warning(f"⚠️ Sort field '{sort_field}' is not numeric — skipping.")
        return []

    # ✅ Universal null filter for sort_field
    df = df[df[sort_field].notnull()]

    df["SORT_VALUE"] = abs(df[sort_field])

    df = df.sort_values(by="SORT_VALUE", ascending=ascending).head(top_n)

    if return_fields:
        # Only return fields that exist in the dataframe
        existing_cols = [col for col in return_fields if col in df.columns]
        df = df[existing_cols]

    return df.to_dict("records")


# ----------------------------------------
# 🧠 GPT Function-Calling Prompt Interface
# ----------------------------------------

import json  # ✅ Needed for function_call args decoding

# ------ UNIVERSAL PROPERTIES LIST ---------

# Shared universal properties for all GPT function specs
universal_properties = {
    "top_n": {"type": "integer"},
    "part_number": {"type": "string"},
    "planning_method": {"type": "string"},
    "accuracy_filter": {"type": "string"},
    "compliant_only": {"type": "boolean"},
    "high_only": {"type": "boolean"},
    "planner": {"type": "string"},
    "buyer": {"type": "string"},
    "supplier": {"type": "string"},
    "make_or_buy": {"type": "string"},
    "commodity": {"type": "string"},
}

# ------SHORTAGE CALL --------
shortage_function_spec = {
    "name": "get_material_shortage_summary",
    "description": "Returns a filtered list of parts with material shortages. Higher values indicate worse performance.",
    "parameters": {
        "type": "object",
        "properties": {
            **universal_properties,
        },
        "required": [],
    },
}


# ------EXCESS CALL --------

excess_function_spec = {
    "name": "get_excess_inventory_summary",
    "description": "Returns a filtered list of parts with excess inventory.",
    "parameters": {
        "type": "object",
        "properties": {
            **universal_properties,
        },
        "required": [],
    },
}


# ------INVENTORY TURNS CALL --------
inventory_turns_function_spec = {
    "name": "get_inventory_turns_summary",
    "description": "Returns a filtered list of parts with inventory turns performance.",
    "parameters": {
        "type": "object",
        "properties": {
            **universal_properties,
        },
        "required": [],
    },
}

# ------SCRAP RATE CALL --------
scrap_rate_function_spec = {
    "name": "get_scrap_rate_summary",
    "description": "Returns a filtered list of parts with average scrap rate.",
    "parameters": {
        "type": "object",
        "properties": {
            **universal_properties,
        },
        "required": [],
    },
}

# -------- Lead Time Accuracy ----------------

lead_time_accuracy_function_spec = {
    "name": "get_lead_time_accuracy_summary",
    "description": "Returns a filtered list of parts with PO, WO, or combined lead time accuracy.",
    "parameters": {
        "type": "object",
        "properties": {
            **universal_properties,
            "order_type": {
                "type": "string",
                "enum": ["PO", "WO", "Combined"],
                "description": "Choose order type: 'PO', 'WO', or 'Combined'.",
            },
        },
        "required": [],
    },
}

# ------ Safety Stock Accuracy -------------

ss_accuracy_function_spec = {
    "name": "get_ss_accuracy_summary",
    "description": "Returns a list of parts comparing ERP vs ideal safety stock values and ranking them based on a deviation % from the ideal.",
    "parameters": {
        "type": "object",
        "properties": {
            **universal_properties,
        },
        "required": [],
    },
}

# ------ Late Orders --------

late_orders_function_spec = {
    "name": "get_late_orders_summary",
    "description": "Returns a filtered list of PO and WO orders with lateness info.",
    "parameters": {
        "type": "object",
        "properties": {
            **universal_properties,
            "order_type": {
                "type": "string",
                "enum": ["PO", "WO"],
                "description": "Filter by order type (PO or WO).",
            },
            "late_only": {
                "type": "boolean",
                "description": "If true, only return late orders.",
            },
            "start_date": {
                "type": "string",
                "description": "Optional start date for the order (used for filtering).",
            },
            "need_by_date": {
                "type": "string",
                "description": "Optional due date for the order (used for filtering).",
            },
            "receipt_date": {
                "type": "string",
                "description": "Optional completion date for the order (used for filtering).",
            },
            "status": {
                "type": "string",
                "description": "Optional order status filter.",
            },
            "pct_late": {
                "type": "number",
                "description": "Percent days late relative to total lead time (0–100).",
            },
        },
        "required": [],
    },
}

# -------- Smart Filters Function Spec ----------

smart_filter_rank_function_spec = {
    "name": "smart_filter_rank_summary",
    "description": (
        "Dynamically filters and ranks the main dataset using a boolean filter column "
        "and a numeric sort column. Useful for prompts like 'top shortages with worst safety stock accuracy'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            **universal_properties,
            "filters": {
                "type": "object",
                "description": (
                    "Dictionary of field-value pairs to apply as filters. Boolean fields are interpreted as is "
                    "(e.g., SHORTAGE_YN: true), while string fields will filter for exact matches "
                    "(e.g., PLANNING_METHOD: 'ROP')."
                ),
                "additionalProperties": {
                    "oneOf": [
                        {"type": "boolean"},
                        {"type": "string"},
                        {"type": "number"},
                    ]
                },
            },
            "sort": {
                "type": "object",
                "description": "Field and direction to sort the data by.",
                "properties": {
                    "field": {
                        "type": "string",
                        "description": "Name of the numeric column to sort by.",
                    },
                    "ascending": {
                        "type": "boolean",
                        "description": "If true, sort in ascending order.",
                    },
                },
                "required": ["field"],
            },
        },
    },
}


# --------- Combined Functional Spec ------------

all_function_specs = [
    # shortage_function_spec,
    # excess_function_spec,
    # inventory_turns_function_spec,
    # scrap_rate_function_spec,
    # lead_time_accuracy_function_spec,
    # ss_accuracy_function_spec,
    late_orders_function_spec,
    smart_filter_rank_function_spec,
]
