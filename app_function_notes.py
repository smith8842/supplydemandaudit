## Universal Filters

"""
Applies standard filters to a given DataFrame using keyword arguments.

Inputs:
- df (pd.DataFrame): The source data to filter.
- **kwargs: Any filter fields, such as part_number, planner, planning_method, etc.

What it does:
- Filters the DataFrame using common fields like PART_NUMBER, PLANNER, BUYER, SUPPLIER, etc.
- Uses case-insensitive matching.
- Handles both simple and compound filters (e.g., lists or dicts for planning methods).

Outputs:
- Filtered version of the input DataFrame.
"""

### Late Orders
"""
Returns a list of late purchase/work orders with optional filters.

Inputs:
- filters (dict): GPT-parsed dictionary of field filters.
- top_n (int): How many rows to return.
- part_number (str), order_type (str), etc.: Specific filters.
- Also supports planner, buyer, supplier, make_or_buy, etc.

What it does:
- Pulls from 'all_orders_df' in session_state.
- Applies universal and dynamic filters.
- Optionally filters to late orders only.
- Returns data sorted by DAYS_LATE.

Outputs:
- List of dictionaries representing order records (for display).
"""

### Detect Functions

"""
Uses GPT to parse the user's prompt and return which functions and filters to apply.

Inputs:
- prompt (str): The user's natural language query.

What it does:
- Passes the prompt to GPT with instructions.
- GPT returns which functions to run, the filters, sorting logic, and match logic (AND/OR).
- Stores the filter dictionary in session state for downstream use.

Outputs:
- Tuple: (functions list, match_type string)
"""

### Route Functions

"""
Routes the GPT-identified function name to the actual Python function.

Inputs:
- name (str): Function name from GPT response (e.g., "smart_filter_rank_summary").
- args (dict): Arguments for the function, including filters and top_n.

What it does:
- Validates the function name.
- Filters args to only those accepted by the function.
- Pulls any shared filters from session_state.
- Calls the appropriate function with cleaned args.

Outputs:
- Whatever the routed function returns (typically a list of dicts or a string).
"""


### Smart Filter Summary

"""
Filters and ranks part records using supplied filters and sort logic.

Inputs:
- filters (dict): Column-value filters from GPT.
- sort (dict): Dictionary with 'field' and 'ascending' keys.
- top_n (int): Max number of records to return.
- return_fields (list): Specific columns to include in output.
- Other universal filter fields like part_number, planner, etc.

What it does:
- Filters part data from 'combined_part_detail_df'.
- Sorts the result using the chosen column.
- Handles nulls and non-numeric sort issues gracefully.

Outputs:
- List of ranked and filtered part records.
"""

### Root Cause

"""
Generates a root cause explanation using GPT for part performance issues.

Inputs:
- user_prompt (str): The full natural language query.
- part_number (str): Optional part to focus on.
- filters (dict): Dictionary of field filters.

What it does:
- Filters both part- and order-level data from session_state.
- Prepares markdown tables of the filtered rows.
- Asks GPT to explain the most likely reasons for the performance issues.

Outputs:
- A GPT-generated string containing the explanation.
"""

### Parameter Recommendation

"""
Suggests parameter updates (e.g., lead time, safety stock) using precomputed audit flags.

Inputs:
- user_prompt (str): The full user query, used to detect which parameters to focus on.
- part_number (str): Optional specific part.
- filters (dict): Dictionary of filters for narrowing the dataset.

What it does:
- Applies filters to part data.
- Parses the prompt to identify which parameters to review (or all if none specified).
- Checks precomputed accuracy flags to determine which values need updating.
- Passes final recommendation list to GPT for explanation.

Outputs:
- A markdown-formatted GPT summary including a recommendation table.
"""
