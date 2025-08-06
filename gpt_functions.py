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
    "IDEAL_LT_ACCURATE": "bool",
    # Numeric fields
    "SHORTAGE_QTY": "numeric",
    "EXCESS_QTY": "numeric",
    "INVENTORY_TURNS": "numeric",
    "AVG_SCRAP_RATE": "numeric",
    "IDEAL_LT_ACCURACY_PCT": "numeric",
    "PO_LT_ACCURACY_PCT": "numeric",
    "WO_LT_ACCURACY_PCT": "numeric",
    "AVG_PO_LEAD_TIME": "numeric",
    "AVG_WO_LEAD_TIME": "numeric",
    "IDEAL_LEAD_TIME": "numeric",
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
        "IDEAL_LT_ACCURATE",
    ],
    "numeric_columns": [
        "SHORTAGE_QTY",
        "EXCESS_QTY",
        "INVENTORY_TURNS",
        "AVG_SCRAP_RATE",
        "IDEAL_LT_ACCURACY_PCT",
        "PO_LT_ACCURACY_PCT",
        "WO_LT_ACCURACY_PCT",
        "AVG_PO_LEAD_TIME",
        "AVG_WO_LEAD_TIME",
        "IDEAL_LEAD_TIME",
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
    """
    Applies all keyword filters to matching dataframe columns (case-insensitive).
    Supports all fields that exist in the dataframe, including booleans.
    """
    non_null_filters = {k: v for k, v in kwargs.items() if v is not None}
    ##st.info(f"🌐 Universal filters applied:\n{json.dumps(non_null_filters)}")

    for k, v in non_null_filters.items():
        # Case-insensitive column match
        matching_cols = [col for col in df.columns if col.upper() == k.upper()]
        if not matching_cols:
            continue  # Skip unknown fields
        col = matching_cols[0]

        # Normalize boolean filter values (handle string "true"/"false")
        if column_metadata.get(col) == "bool":
            if isinstance(v, str):
                v = v.strip().lower() == "true"
            elif isinstance(v, dict):
                # Handle GPT cases like {"value": true}
                v = list(v.values())[0]

        df = df[df[col] == v]

    return df


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
    2. If the user is trying to diagnose a problem and asks **why** something is happening, or what is **causing** a problem (e.g., “why are there so many excess parts,” “what’s causing our low inventory turns”), match `get_root_cause_explanation`.
    3. If the user asks for recommendations on how to change parameters, or suggestions of new values for parameters, then match 'get_parameter_recommendations.' If the user asks *what* needs to change, that is likely either 'get_root_cause_explanation' or 'smart_filter_summary' depending on other prompt language, NOT 'get_parameter_recommendation'.
    4. If the user ask for more details or data about certain PARTS, match the 'smart_filter_rank_summary' function.
    5. If the user ask for more details or data about certain ORDERS, match the 'smart_filter_rank_summary' function.
    6. Extracting relevant arguments like part_number, order_type, top_n, planner, planning_method, or accuracy_filter when mentioned.
    7. Assuming defaults when needed (e.g. if part_number is not specified, return top rows).
    8. Supporting logical operators (AND/OR). If the user asks about multiple concerns, match all applicable functions.
    89. If the user mentions any filtering criteria (e.g., "ROP", "planner = David", "only shortages", "just POs"), place those inside a `filters` dictionary:
        Example: 
        "filters": {
            "PLANNING_METHOD": "ROP",
            "PLANNER": "David",
            "SHORTAGE_YN": true
        }
    10. If the user prompt suggests ranking or ordering (e.g., "top 5", "worst shortages", "best accuracy"), include a `sort` dictionary:
        Example:
        "sort": {
            "field": "SS_DEVIATION_PCT",
            "ascending": false
        }
    11. Only assign numeric columns to the `sort.field` value — like SHORTAGE_QTY, AVG_SCRAP_RATE, INVENTORY_TURNS, etc.
    12. Do NOT assign boolean fields like SHORTAGE_YN or EXCESS_YN to the sort field. These can only be used in the `filters` dictionary.
    13. Interpret ranking language:
        - "best", "highest", "top", "most" → ascending = false
        - "worst", "lowest", "bottom", "least" → ascending = true
    14. There are certain fields where ranking language would have an inverse rank order. Currently those including Shortages, Excess, Safety Stock Deviations, Scrap Rates, and Days Late. For example - the worst safety stock deviation would be "ascending = false"
    15. If the user adds ranking language in conjunction with terms that you intepret as a column (like the phrase "worst shortages"), then the user is looking for a sort and ranking on the SHORTAGE_QTY field, not a filter on the SHORTAGE_YN field.
    16. The get_late_orders_summary function is looking at data about orders. So, it should only ever be chosen if the user asks for order-centric data. For example, "Show me all late orders." But something like "show me all parts with late orders," the prompt is asking for part-centric data.
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

    column_descriptions = json.dumps(
        st.session_state.get("column_definitions", {}), indent=2
    )

    user_prompt = (
        f"Available functions:\n{function_descriptions}\n\n"
        f"{column_list_text}\n\n"
        f"Column meanings:\n{column_descriptions}\n\n"
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
        ##st.write("🧾 Raw GPT function parse output:", raw)
        parsed = json.loads(raw)
        functions = parsed.get("functions", [])
        match_type = parsed.get("match_type", "intersection").lower()

        # Store filters in session state to inject into all matched functions
        first_func_args = functions[0]["arguments"] if functions else {}
        st.session_state["last_detected_filters"] = first_func_args.get("filters", {})

        return functions, match_type
    except Exception as e:
        print(f"⚠️ Failed to parse GPT response: {e}")
        return [], "intersection"


# ---------- Routed Function Call -----------


def route_gpt_function_call(name: str, args: dict):
    """
    Routes GPT function call name to the corresponding Python function.
    Filters args based on function signature.
    """
    router = {
        "get_late_orders_summary": get_late_orders_summary,
        "smart_filter_rank_summary": smart_filter_rank_summary,
        "get_root_cause_explanation": get_root_cause_explanation,
        "get_parameter_recommendations": get_parameter_recommendations,
    }

    if name not in router:
        return f"⚠️ No matching function found for: {name}"

    # st.write("🧠 GPT raw function name + args:", name, args)

    fn = router[name]

    # Filter args based on the function's accepted parameters
    import inspect

    sig = inspect.signature(fn)
    accepted_args = {}

    # Global filters from last GPT parse (shared across all matched functions)
    shared_filters = st.session_state.get("last_detected_filters", {})

    # st.write("🌐 Shared filters available to all functions:", shared_filters)

    for k, v in args.items():
        if k in sig.parameters:
            accepted_args[k] = v

    # Always pass filters if supported by the function
    if "filters" in sig.parameters:
        accepted_args["filters"] = accepted_args.get("filters", shared_filters)

    # Inject user_prompt if the function expects it and it's missing
    if "user_prompt" in sig.parameters and "user_prompt" not in accepted_args:
        accepted_args["user_prompt"] = st.session_state.get("last_user_prompt", "")

    ##st.write("📦 Final accepted args to function:", accepted_args)

    # 👇 Debug display in UI
    import json

    # st.info(
    #     "🔍 Function `{}` called with parameters:\n\n{}".format(
    #         name, json.dumps(accepted_args, indent=2)
    #     )
    # )

    return fn(**accepted_args)


# ------ SMART FILTER ROUTER ---------


def smart_filter_rank_summary(
    filters: dict = None,
    sort: dict = None,
    top_n: int = None,
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

    # st.info("🔍 Smart filter args:\n" + json.dumps(args, indent=2))

    df = apply_universal_filters(df, **args)

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
        # st.info("ℹ️ No sort field provided — returning unranked filtered results.")
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


# ------------ Root Cause Explanation Function ------------


def get_root_cause_explanation(
    user_prompt: str, part_number: str = None, filters: dict = None
) -> str:
    """
    Generates a GPT-based root cause explanation for a specific part, filtered group of parts, or overall dataset.
    """

    part_df = st.session_state.get("combined_part_detail_df", pd.DataFrame())
    orders_df = st.session_state.get("all_orders_df", pd.DataFrame())

    if part_df.empty:
        return "No part-level audit data available to analyze root causes."

    # Apply universal filters to both part-level and order-level data
    args = {"part_number": part_number}
    args.update(filters or {})  # Expand filters into args

    ##st.write("📥 Merged filters in get_root_cause_explanation:", args)

    part_df = apply_universal_filters(part_df, **args)
    orders_df = apply_universal_filters(orders_df, **args)

    if part_number:
        orders_df = orders_df[orders_df["PART_NUMBER"] == part_number]

    if part_df.empty:
        return "No matching part-level data found based on the provided filters."

    part_df = part_df.head(10)
    orders_df = orders_df.head(10)

    # Select key fields for GPT analysis
    part_fields = [
        "PART_NUMBER",
        "PLANNING_METHOD",
        "SHORTAGE_YN",
        "SHORTAGE_QTY",
        "EXCESS_YN",
        "EXCESS_QTY",
        "SS_COMPLIANT_PART",
        "SS_DEVIATION_QTY",
        "SS_DEVIATION_PCT",
        "SAFETY_STOCK",
        "IDEAL_SS",
        "AVG_SCRAP_RATE",
        "HIGH_SCRAP_PART",
        "SCRAP_DENOMINATOR",
        "TRAILING_CONSUMPTION",
        "STD_DEV_CONSUMPTION",
        "PO_LEAD_TIME_ACCURATE",
        "PO_LT_ACCURACY_PCT",
        "AVG_PO_LEAD_TIME",
        "PO_COUNT",
        "WO_LEAD_TIME_ACCURATE",
        "WO_LT_ACCURACY_PCT",
        "AVG_WO_LEAD_TIME",
        "WO_COUNT",
        "IDEAL_LT_ACCURATE",
        "IDEAL_LT_ACCURACY_PCT",
        "IDEAL_LEAD_TIME",
        "ERP_LEAD_TIME",
        "INVENTORY_TURNS",
    ]
    order_fields = [
        "ORDER_TYPE",
        "ORDER_ID",
        "PART_NUMBER",
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

    part_prompt = part_df[[c for c in part_fields if c in part_df.columns]]
    orders_prompt = orders_df[[c for c in order_fields if c in orders_df.columns]]

    # Convert to markdown tables
    part_markdown = part_prompt.to_markdown(index=False)
    orders_markdown = (
        orders_prompt.to_markdown(index=False)
        if not orders_prompt.empty
        else "*No matching orders found.*"
    )

    system_msg = """
    1. You are a supply chain analyst. A planner is asking why a part (or set of parts) is performing poorly/well, or why planning parameters or good/bad  — for example, when the user asks for root causes, explanations, biggest issues, most likely problems, reasons for excess inventory or shortages, or general performance concerns.
    2. Analyze the provided part-level audit metrics and relevant orders. Identify which metrics or signals most likely explain the issue(s). Focus on root causes and call out which problem is most important to fix first. Do not list every field — just the most relevant ones.
    3. There is a general order of operations when approaching planning fixes. Assuming all variables have issues, it tends to be appropriate to first fix how a part is planned (planning method). Then, amongst the other variables, it is important to note that lead time is a factor in almost everything, including calculating things like Safty Stock. So it is important to get that right.
    4. If the user is just asking for greater clarification or explanation of an issue, there is no need to give a step by step "how to fix" solution.
    5. Keep your response concise, direct, and focused. Avoid unnecessary elaboration or repetition. If the user asks for a specific explanation, don't add 
    6. Keep your response dedicated to the user's request. For example, if the user asks for why a part has so much excess inventory, keep the response entirely focused on the root causes of excess inventory. If the user asks for what effect bad lead times would have on a part focus entirely on the affects of bad lead times.
    """.strip()

    user_msg = f"""
    User question: {user_prompt}

    Part-level audit summary:
    {part_markdown}

    Order-level sample:
    {orders_markdown}
    """

    client = OpenAIClient(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    return response.choices[0].message.content.strip()


# --------- Parameter Recommendation -------------
def get_parameter_recommendations(
    user_prompt: str, part_number: str = None, filters: dict = None
) -> str:
    """
    GPT-callable: Suggests planning parameter changes (SS, LT, Min/Max, Planning Method, etc.) using audit data.
    Applies accuracy filters in Python, and delegates reasoning and explanation to GPT.
    """
    df = st.session_state.get("combined_part_detail_df", pd.DataFrame())
    if df.empty:
        return "No part-level audit data available to generate parameter suggestions."

    args = {"part_number": part_number}
    args.update(filters or {})  # Merge filters into args dict

    ##st.write("📥 Merged filters in get_parameter_recommendations:", args)

    df = apply_universal_filters(df, **args)

    if df.empty:
        return "No matching part data found for parameter recommendation."

    # Map GPT-triggered keywords to parameter columns
    focus_fields = {
        "LEAD_TIME": ["lead time"],
        "SAFETY_STOCK": ["safety stock", "ss"],
        "MIN_QTY": ["min"],
        "MAX_QTY": ["max"],
        "REORDER_POINT": ["reorder point", "rop"],
        "PLANNING_METHOD": ["planning method"],
    }

    user_lower = user_prompt.lower()
    requested_fields = [
        f
        for f, triggers in focus_fields.items()
        if any(t in user_lower for t in triggers)
    ]

    if not requested_fields:
        requested_fields = list(focus_fields.keys())

    # Define mapping and accuracy filters
    column_pairs = {
        "LEAD_TIME": ("LEAD_TIME", "IDEAL_LEAD_TIME"),
        "SAFETY_STOCK": ("SAFETY_STOCK", "IDEAL_SS"),
        "MIN_QTY": ("MIN_QTY", "IDEAL_MINIMUM"),
        "MAX_QTY": ("MAX_QTY", "IDEAL_MAXIMUM"),
        "REORDER_POINT": ("REORDER_POINT", "IDEAL_MINIMUM"),
        "PLANNING_METHOD": ("PLANNING_METHOD", "IDEAL_PLANNING_METHOD"),
    }
    accuracy_flags = {
        "LEAD_TIME": "IDEAL_LT_ACCURATE",
        "SAFETY_STOCK": "SS_COMPLIANT_PART",
    }

    # Build long-form filtered table
    rows = []
    for _, row in df.iterrows():
        for param in requested_fields:
            if param not in column_pairs:
                continue

            current_field, ideal_field = column_pairs[param]
            if current_field not in df.columns or ideal_field not in df.columns:
                continue

            # Enforce accuracy-based filters
            if param in accuracy_flags:
                flag_col = accuracy_flags[param]
                if (
                    flag_col not in row
                    or row[flag_col] is True
                    or pd.isna(row[flag_col])
                ):
                    continue

            current_value = row.get(current_field, None)
            ideal_value = row.get(ideal_field, None)

            # Must have valid values
            if pd.isna(current_value) or pd.isna(ideal_value):
                continue

            # Reject boolean or None
            if isinstance(ideal_value, bool) or ideal_value is None:
                continue

            rows.append(
                {
                    "PART_NUMBER": row["PART_NUMBER"],
                    "PARAMETER": param,
                    "CURRENT_VALUE": current_value,
                    "RECOMMENDED_VALUE": ideal_value,
                }
            )

    if not rows:
        return "No suggested parameter updates based on the accuracy flags and current data."

    gpt_df = pd.DataFrame(rows)

    table_md = gpt_df.to_markdown(index=False)

    system_msg = """
    You are a supply chain planning assistant helping a planner identify which parameters need to be updated.

    The table provided shows only parts and parameters that are flagged for update, based on pre-calculated audit flags.
    Simply confirm the CURRENT and RECOMMENDED values and provide helpful context in your explanation.

    Rules:
    - Do not second guess whether a change is needed. The logic has already filtered the list.
    - Display a markdown table with: PART_NUMBER, PARAMETER, CURRENT_VALUE, RECOMMENDED_VALUE
    - Then summarize why these changes are being recommended and what patterns you observe.
    """.strip()

    user_msg = f"""
    User request: {user_prompt}

    Recommended parameter updates:
    {table_md}
    """

    client = OpenAIClient(api_key=openai_api_key)
    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    return response.choices[0].message.content.strip()


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

# --------- Root Cause Analyis Spec -----------


root_cause_explanation_spec = {
    "name": "get_root_cause_explanation",
    "description": "Analyzes audit metrics and order history to explain the most likely root causes of issues for a specific part, group of parts, or the overall business.",
    "parameters": {
        "type": "object",
        "properties": {
            "user_prompt": {
                "type": "string",
                "description": "The user's original question, used to guide GPT's reasoning (e.g., 'Why is PN123 always short?').",
            },
            "part_number": {
                "type": "string",
                "description": "Optional specific part number to focus the analysis on a single part.",
            },
            "filters": {
                "type": "object",
                "description": "Optional dictionary of filters to limit the analysis to a specific group of parts (e.g., planning method = MRP, planner = John).",
                "additionalProperties": {"type": ["string", "number", "boolean"]},
            },
        },
        "required": ["user_prompt"],
    },
}

# --------- Parameter Recommendation Spec -----------

parameter_recommendation_spec = {
    "name": "get_parameter_recommendations",
    "description": (
        "Suggests planning parameter changes such as safety stock, lead time, reorder point, and planning method. "
        "Uses existing audit metrics and IDEAL_ values to determine which parameters should be adjusted. "
        "Useful for prompts like 'What should my SS for PN001 be?' or 'Which parts need planning method changes?'"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "user_prompt": {
                "type": "string",
                "description": "The user's original question to guide GPT's parameter logic.",
            },
            "part_number": {
                "type": "string",
                "description": "Optional specific part number to evaluate.",
            },
            "filters": {
                "type": "object",
                "description": "Optional dictionary of filters to narrow the scope (e.g., planning method = ROP, shortage = true).",
                "additionalProperties": {"type": ["string", "number", "boolean"]},
            },
        },
        "required": ["user_prompt"],
    },
}


# --------- Combined Functional Spec ------------

all_function_specs = [
    late_orders_function_spec,
    smart_filter_rank_function_spec,
    root_cause_explanation_spec,
    parameter_recommendation_spec,
]
