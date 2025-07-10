import streamlit as st
import pandas as pd
import json
from openai import OpenAI as OpenAIClient

openai_api_key = st.secrets["OPENAI_API_KEY"]

# ----------------------------------------
# GPT Callable Metric Functions
# ----------------------------------------


# ------------ SHORTAGES ---------------
def get_material_shortage_summary(
    top_n: int = None, part_number: str = None, planning_method: list[str] | str = None
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
    if df.empty or "SHORTAGE" not in df.columns:
        return []

    df_short = df[df["SHORTAGE"] == True].copy()

    # Optional: filter by part number
    if part_number:
        df_short = df_short[df_short["PART_NUMBER"] == part_number]

    # Optional: filter by planning method(s)
    if planning_method and "PLANNING_METHOD" in df_short.columns:
        if isinstance(planning_method, str):
            planning_method = [planning_method]
        df_short = df_short[df_short["PLANNING_METHOD"].isin(planning_method)]

    # Only show meaningful shortage values
    df_short = df_short[df_short["SHORTAGE_AMOUNT"] > 0]
    df_short = df_short.sort_values(by="SHORTAGE_AMOUNT", ascending=False)

    if top_n is not None:
        df_short = df_short.head(top_n)

    cols_to_return = [
        "PART_NUMBER",
        "PLANNING_METHOD",
        "ON_HAND_QUANTITY",
        "IDEAL_MINIMUM",
        "LATE_MRP_NEED_QTY",
        "SHORTAGE_AMOUNT",
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
    top_n: int = None, part_number: str = None, planning_method: list[str] | str = None
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
    if df.empty or "EXCESS" not in df.columns:
        return []

    df_excess = df[df["EXCESS"] == True].copy()

    if part_number:
        df_excess = df_excess[df_excess["PART_NUMBER"] == part_number]

    if planning_method and "PLANNING_METHOD" in df_excess.columns:
        if isinstance(planning_method, str):
            planning_method = [planning_method]
        df_excess = df_excess[df_excess["PLANNING_METHOD"].isin(planning_method)]

    df_excess = df_excess[df_excess["EXCESS_AMOUNT"] > 0]
    df_excess = df_excess.sort_values(by="EXCESS_AMOUNT", ascending=False)

    if top_n is not None:
        df_excess = df_excess.head(top_n)

    cols_to_return = [
        "PART_NUMBER",
        "PLANNING_METHOD",
        "ON_HAND_QUANTITY",
        "IDEAL_MAXIMUM",
        "LATE_MRP_NEED_QTY",
        "EXCESS_AMOUNT",
    ]

    existing_cols = [col for col in cols_to_return if col in df_excess.columns]
    return df_excess[existing_cols].to_dict("records")


# --------- INVENTORY TURNS CALL -------------


def get_inventory_turns_summary(
    top_n: int = None, part_number: str = None, planning_method: list[str] | str = None
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

    df_turns = df.copy()

    if part_number:
        df_turns = df_turns[df_turns["PART_NUMBER"] == part_number]

    if planning_method and "PLANNING_METHOD" in df_turns.columns:
        if isinstance(planning_method, str):
            planning_method = [planning_method]
        df_turns = df_turns[df_turns["PLANNING_METHOD"].isin(planning_method)]

    df_turns = df_turns[df_turns["TRAILING_CONSUMPTION"] > 0]
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
    high_only: bool = False,
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

    df_scrap = df.copy()

    if part_number:
        df_scrap = df_scrap[df_scrap["PART_NUMBER"] == part_number]

    if high_only:
        df_scrap = df_scrap[df_scrap["AVG_SCRAP_RATE"] > 0.10]

    df_scrap = df_scrap[df_scrap["AVG_SCRAP_RATE"] > 0]
    df_scrap = df_scrap.sort_values(by="AVG_SCRAP_RATE", ascending=False)

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
    order_type: str = None,
    accuracy_filter: str = None,
):
    """
    GPT-callable: Returns lead time accuracy summary for POs, WOs, or combined.

    Parameters
    ----------
    top_n : int
        Max rows to return, sorted by lowest accuracy.
    part_number : str
        If specified, filters to this part number.
    order_type : str
        "PO", "WO", or "Combined"
    accuracy_filter : str
        "accurate", "inaccurate", or None

    Returns
    -------
    List[Dict]
        A filtered list of lead time accuracy records.
    """
    df = st.session_state.get("combined_part_detail_df", pd.DataFrame())
    if df.empty:
        return []

    records = []

    if order_type in [None, "PO"]:
        po_df = df.copy()
        if part_number:
            po_df = po_df[po_df["PART_NUMBER"] == part_number]
        if accuracy_filter == "accurate":
            po_df = po_df[po_df["PO_LEAD_TIME_ACCURATE"] == True]
        elif accuracy_filter == "inaccurate":
            po_df = po_df[po_df["PO_LEAD_TIME_ACCURATE"] == False]
        po_df = po_df[po_df["AVG_PO_LEAD_TIME"].notnull()]
        po_df["ORDER_TYPE"] = "PO"
        po_df["ACCURATE"] = po_df["PO_LEAD_TIME_ACCURATE"]
        po_df["AVG_LEAD_TIME"] = po_df["AVG_PO_LEAD_TIME"]
        po_df["ERP_LEAD_TIME"] = po_df["LEAD_TIME"]
        records.append(po_df)

    if order_type in [None, "WO"]:
        wo_df = df.copy()
        if part_number:
            wo_df = wo_df[wo_df["PART_NUMBER"] == part_number]
        if accuracy_filter == "accurate":
            wo_df = wo_df[wo_df["WO_LEAD_TIME_ACCURATE"] == True]
        elif accuracy_filter == "inaccurate":
            wo_df = wo_df[wo_df["WO_LEAD_TIME_ACCURATE"] == False]
        wo_df = wo_df[wo_df["AVG_WO_LEAD_TIME"].notnull()]
        wo_df["ORDER_TYPE"] = "WO"
        wo_df["ACCURATE"] = wo_df["WO_LEAD_TIME_ACCURATE"]
        wo_df["AVG_LEAD_TIME"] = wo_df["AVG_WO_LEAD_TIME"]
        wo_df["ERP_LEAD_TIME"] = wo_df["LEAD_TIME"]
        records.append(wo_df)

    if order_type in ["Combined"]:
        combined_df = df.copy()
        if part_number:
            combined_df = combined_df[combined_df["PART_NUMBER"] == part_number]
        if accuracy_filter == "accurate":
            combined_df = combined_df[combined_df["COMBINED_LT_ACCURATE"] == True]
        elif accuracy_filter == "inaccurate":
            combined_df = combined_df[combined_df["COMBINED_LT_ACCURATE"] == False]
        combined_df = combined_df[combined_df["COMBINED_LT"].notnull()]
        combined_df["ORDER_TYPE"] = "Combined"
        combined_df["ACCURATE"] = combined_df["COMBINED_LT_ACCURATE"]
        combined_df["AVG_LEAD_TIME"] = combined_df["COMBINED_LT"]
        combined_df["ERP_LEAD_TIME"] = combined_df["LEAD_TIME"]
        records.append(combined_df)

    if not records:
        return []

    full_df = pd.concat(records)
    full_df = full_df.sort_values(by="ACCURATE", ascending=True)

    if top_n:
        full_df = full_df.head(top_n)

    return full_df[
        ["PART_NUMBER", "ORDER_TYPE", "ERP_LEAD_TIME", "AVG_LEAD_TIME", "ACCURATE"]
    ].to_dict("records")


# ------- Safety Stock Accuracy ---------


def get_ss_accuracy_summary(
    top_n: int = None,
    part_number: str = None,
    compliant_only: bool = False,
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
        If True, only return parts where SS is within tolerance.

    Returns
    -------
    List[Dict]
        A list of parts with ERP vs ideal safety stock comparison.
    """
    df = st.session_state.get("combined_part_detail_df", pd.DataFrame())
    if df.empty or "SS_COMPLIANT_PART" not in df.columns:
        return []

    df_ss = df.copy()

    if part_number:
        df_ss = df_ss[df_ss["PART_NUMBER"] == part_number]

    if compliant_only:
        df_ss = df_ss[df_ss["SS_COMPLIANT_PART"] == True]

    df_ss["SS_DEVIATION"] = abs(df_ss["SAFETY_STOCK"] - df_ss["IDEAL_SS"])
    df_ss = df_ss.sort_values(by="SS_DEVIATION", ascending=False)

    if top_n:
        df_ss = df_ss.head(top_n)

    return df_ss[
        [
            "PART_NUMBER",
            "PLANNING_METHOD",
            "SAFETY_STOCK",
            "IDEAL_SS",
            "SS_COMPLIANT_PART",
        ]
    ].to_dict("records")


# --------- Late Orders ------------


def get_late_orders_summary(
    top_n: int = None,
    order_type: str = None,
    part_number: str = None,
    late_only: bool = False,
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

    df_filtered = df.copy()

    if order_type:
        df_filtered = df_filtered[df_filtered["ORDER_TYPE"] == order_type]

    if part_number:
        df_filtered = df_filtered[df_filtered["PART_NUMBER"] == part_number]

    if late_only:
        df_filtered = df_filtered[df_filtered["IS_LATE"] == True]

    df_filtered["DAYS_LATE"] = (
        df_filtered["RECEIPT_DATE"] - df_filtered["NEED_BY_DATE"]
    ).dt.days
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
            "NEED_BY_DATE",
            "RECEIPT_DATE",
            "IS_LATE",
            "DAYS_LATE",
            "ERP_LEAD_TIME",
            "LT_DAYS",
        ]
    ].to_dict("records")


# ---------- Combined Function Call ---------

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

    system_prompt = (
        "You are an intelligent audit assistant. You will receive a list of supply chain audit functions and a user prompt.\n"
        "Your job is to:\n"
        "1. Identify which functions are relevant.\n"
        "2. Determine whether the user is asking for results that match ALL functions (AND/intersection) or ANY function (OR/union).\n"
        "You MUST return a valid JSON dictionary like this:\n"
        "{\n"
        '  "functions": ["get_x", "get_y"],\n'
        '  "match_type": "intersection"\n'
        "}\n"
        "Use 'intersection' if the user seems to expect all conditions to be true (e.g. 'and', 'both').\n"
        "Use 'union' if the user asks for any of several options (e.g. 'or', 'either')."
    )

    user_prompt = (
        f"Available functions:\n{function_descriptions}\n\n"
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


# -------- Prompt Parameter Extraction ---------------


def extract_common_parameters(prompt: str) -> dict:
    prompt = prompt.lower()
    params = {}

    if "top 5" in prompt:
        params["top_n"] = 5
    elif "top 10" in prompt:
        params["top_n"] = 10
    elif "top 3" in prompt:
        params["top_n"] = 3

    if "worst" in prompt or "lowest" in prompt:
        params["accuracy_filter"] = "inaccurate"
    elif "best" in prompt or "highest" in prompt:
        params["accuracy_filter"] = "accurate"

    if "po" in prompt:
        params["order_type"] = "PO"
    elif "wo" in prompt:
        params["order_type"] = "WO"

    return params


# ---------- Routed Function Call -----------


def route_gpt_function_call(name: str, args: dict):
    """
    Routes GPT function call name to the corresponding Python function.
    Filters args based on function signature.
    """
    router = {
        "get_material_shortage_summary": get_material_shortage_summary,
        "get_excess_inventory_summary": get_excess_inventory_summary,
        "get_inventory_turns_summary": get_inventory_turns_summary,
        "get_scrap_rate_summary": get_scrap_rate_summary,
        "get_lead_time_accuracy_summary": get_lead_time_accuracy_summary,
        "get_ss_accuracy_summary": get_ss_accuracy_summary,
        "get_late_orders_summary": get_late_orders_summary,
    }

    if name not in router:
        return f"⚠️ No matching function found for: {name}"

    fn = router[name]

    # Filter args based on the function's accepted parameters
    import inspect

    sig = inspect.signature(fn)
    accepted_args = {k: v for k, v in args.items() if k in sig.parameters}

    return fn(**accepted_args)


# ----------------------------------------
# 🧠 GPT Function-Calling Prompt Interface
# ----------------------------------------

import json  # ✅ Needed for function_call args decoding

# ------SHORTAGE CALL --------
shortage_function_spec = {
    "name": "get_material_shortage_summary",
    "description": "Returns a filtered list of parts with material shortages.",
    "parameters": {
        "type": "object",
        "properties": {
            "top_n": {
                "type": "integer",
                "description": "Number of shortage rows (excluding headers) to return to the user.",
            },
            "part_number": {
                "type": "string",
                "description": "Optional: filter by specific part number",
            },
            "planning_method": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ],
                "description": "Restrict to planning method(s), e.g. 'MRP', 'ROP', or ['ROP', 'MRP']",
            },
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
            "top_n": {
                "type": "integer",
                "description": "Number of excess rows (excluding headers) to return to the user.",
            },
            "part_number": {
                "type": "string",
                "description": "Optional: filter by specific part number",
            },
            "planning_method": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ],
                "description": "Restrict to planning method(s), e.g. 'MRP', 'ROP', or ['ROP', 'MRP']",
            },
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
            "top_n": {
                "type": "integer",
                "description": "Max number of results to return, sorted by lowest inventory turns.",
            },
            "part_number": {
                "type": "string",
                "description": "Optional: filter by specific part number.",
            },
            "planning_method": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "array", "items": {"type": "string"}},
                ],
                "description": "Restrict to planning method(s), e.g. 'MRP', 'ROP', or ['ROP', 'MRP']",
            },
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
            "top_n": {
                "type": "integer",
                "description": "Max number of results to return, sorted by highest scrap rate.",
            },
            "part_number": {
                "type": "string",
                "description": "Optional: filter by specific part number.",
            },
            "high_only": {
                "type": "boolean",
                "description": "If true, only return parts with high scrap rates (above threshold).",
            },
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
            "top_n": {
                "type": "integer",
                "description": "Max number of results to return.",
            },
            "part_number": {
                "type": "string",
                "description": "Optional: filter by part number.",
            },
            "order_type": {
                "type": "string",
                "enum": ["PO", "WO", "Combined"],
                "description": "Choose order type: 'PO', 'WO', or 'Combined'.",
            },
            "accuracy_filter": {
                "type": "string",
                "enum": ["accurate", "inaccurate"],
                "description": "Filter to only 'accurate' or 'inaccurate' parts (optional).",
            },
        },
        "required": [],
    },
}

# ------ Safety Stock Accuracy -------------

ss_accuracy_function_spec = {
    "name": "get_ss_accuracy_summary",
    "description": "Returns a list of parts comparing ERP vs ideal safety stock values.",
    "parameters": {
        "type": "object",
        "properties": {
            "top_n": {
                "type": "integer",
                "description": "Max number of results to return, sorted by greatest deviation.",
            },
            "part_number": {
                "type": "string",
                "description": "Optional: filter by specific part number.",
            },
            "compliant_only": {
                "type": "boolean",
                "description": "If true, only return parts where SS is within acceptable tolerance.",
            },
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
            "top_n": {
                "type": "integer",
                "description": "Max number of late orders to return.",
            },
            "order_type": {
                "type": "string",
                "enum": ["PO", "WO"],
                "description": "Filter by order type (PO or WO).",
            },
            "part_number": {
                "type": "string",
                "description": "Optional filter by part number.",
            },
            "late_only": {
                "type": "boolean",
                "description": "If true, only return late orders.",
            },
        },
        "required": [],
    },
}

# --------- Combined Functional Spec ------------

all_function_specs = [
    shortage_function_spec,
    excess_function_spec,
    inventory_turns_function_spec,
    scrap_rate_function_spec,
    lead_time_accuracy_function_spec,
    ss_accuracy_function_spec,
    late_orders_function_spec,
]
