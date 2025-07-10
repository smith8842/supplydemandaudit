# -----------------------------
# ------ GPT Text Blocks ------
# -----------------------------

# # ---- SHORTAGE ----

# with st.expander("💬 Ask GPT: Shortage Q&A"):
#     user_prompt = st.text_input("Ask a question about shortages:")

#     if st.button("Ask GPT"):
#         client = OpenAIClient(
#             api_key=openai_api_key, organization="org-3Va0Uv9V3lCF4EWsBURKlCAG"
#         )

#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You are a helpful supply chain assistant that answers questions using available functions.",
#                     },
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 functions=[shortage_function_spec],
#                 function_call="auto",
#                 temperature=0.4,
#             )

#             choice = response.choices[0]
#             if choice.finish_reason == "function_call":
#                 name = choice.message.function_call.name
#                 args = json.loads(choice.message.function_call.arguments)

#                 if name == "get_material_shortage_summary":
#                     result = get_material_shortage_summary(**args)
#                     st.success("GPT called the function and received:")
#                     st.dataframe(pd.DataFrame(result))
#             else:
#                 st.info(choice.message.content)

#         except Exception as e:
#             st.error(f"Function call failed: {e}")

# # ----- EXCESS -----

# with st.expander("💬 Ask GPT: Excess Q&A"):
#     user_prompt = st.text_input("Ask a question about excess inventory:")

#     if st.button("Ask GPT (Excess)"):
#         client = OpenAIClient(
#             api_key=openai_api_key, organization="org-3Va0Uv9V3lCF4EWsBURKlCAG"
#         )

#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You are a helpful supply chain assistant that answers questions using available functions.",
#                     },
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 functions=[excess_function_spec],
#                 function_call="auto",
#                 temperature=0.4,
#             )

#             choice = response.choices[0]
#             if choice.finish_reason == "function_call":
#                 name = choice.message.function_call.name
#                 args = json.loads(choice.message.function_call.arguments)

#                 if name == "get_excess_inventory_summary":
#                     result = get_excess_inventory_summary(**args)
#                     st.success("GPT called the function and received:")
#                     st.dataframe(pd.DataFrame(result))
#             else:
#                 st.info(choice.message.content)

#         except Exception as e:
#             st.error(f"Function call failed: {e}")

# # ------ Inventory Turns -------

# with st.expander("💬 Ask GPT: Inventory Turns Q&A"):
#     user_prompt = st.text_input("Ask a question about inventory turns:")

#     if st.button("Ask GPT (Turns)"):
#         client = OpenAIClient(
#             api_key=openai_api_key, organization="org-3Va0Uv9V3lCF4EWsBURKlCAG"
#         )

#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You are a helpful supply chain assistant that answers questions using available functions.",
#                     },
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 functions=[inventory_turns_function_spec],
#                 function_call="auto",
#                 temperature=0.4,
#             )

#             choice = response.choices[0]
#             if choice.finish_reason == "function_call":
#                 name = choice.message.function_call.name
#                 args = json.loads(choice.message.function_call.arguments)

#                 if name == "get_inventory_turns_summary":
#                     result = get_inventory_turns_summary(**args)
#                     st.success("GPT called the function and received:")
#                     st.dataframe(pd.DataFrame(result))
#             else:
#                 st.info(choice.message.content)

#         except Exception as e:
#             st.error(f"Function call failed: {e}")

# # ------ Scrap Rates -------

# with st.expander("💬 Ask GPT: Scrap Rate Q&A"):
#     user_prompt = st.text_input("Ask a question about scrap rates:")

#     if st.button("Ask GPT (Scrap)"):
#         client = OpenAIClient(
#             api_key=openai_api_key, organization="org-3Va0Uv9V3lCF4EWsBURKlCAG"
#         )

#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You are a helpful supply chain assistant that answers questions using available functions.",
#                     },
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 functions=[scrap_rate_function_spec],
#                 function_call="auto",
#                 temperature=0.4,
#             )

#             choice = response.choices[0]
#             if choice.finish_reason == "function_call":
#                 name = choice.message.function_call.name
#                 args = json.loads(choice.message.function_call.arguments)

#                 if name == "get_scrap_rate_summary":
#                     result = get_scrap_rate_summary(**args)
#                     st.success("GPT called the function and received:")
#                     st.dataframe(pd.DataFrame(result))
#             else:
#                 st.info(choice.message.content)

#         except Exception as e:
#             st.error(f"Function call failed: {e}")

# # ---------- Lead Time Accuracy --------------

# with st.expander("💬 Ask GPT: Lead Time Accuracy Q&A"):
#     user_prompt = st.text_input(
#         "Ask a question about PO, WO, or combined lead time accuracy:"
#     )

#     if st.button("Ask GPT (Lead Time Accuracy)"):
#         client = OpenAIClient(
#             api_key=openai_api_key, organization="org-3Va0Uv9V3lCF4EWsBURKlCAG"
#         )

#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You are a helpful supply chain assistant that answers questions using available functions.",
#                     },
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 functions=[lead_time_accuracy_function_spec],
#                 function_call="auto",
#                 temperature=0.4,
#             )

#             choice = response.choices[0]
#             if choice.finish_reason == "function_call":
#                 name = choice.message.function_call.name
#                 args = json.loads(choice.message.function_call.arguments)

#                 if name == "get_lead_time_accuracy_summary":
#                     result = get_lead_time_accuracy_summary(**args)
#                     st.success("GPT called the function and received:")
#                     st.dataframe(pd.DataFrame(result))
#             else:
#                 st.info(choice.message.content)

#         except Exception as e:
#             st.error(f"Function call failed: {e}")

# # ------- Late Orders -----------

# with st.expander("💬 Ask GPT: Late Orders Q&A"):
#     user_prompt = st.text_input("Ask a question about late purchase or work orders:")

#     if st.button("Ask GPT (Late Orders)"):
#         client = OpenAIClient(
#             api_key=openai_api_key, organization="org-3Va0Uv9V3lCF4EWsBURKlCAG"
#         )

#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You are a helpful supply chain assistant that answers questions using available functions.",
#                     },
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 functions=[late_orders_function_spec],
#                 function_call="auto",
#                 temperature=0.4,
#             )

#             choice = response.choices[0]
#             if choice.finish_reason == "function_call":
#                 name = choice.message.function_call.name
#                 args = json.loads(choice.message.function_call.arguments)

#                 if name == "get_late_orders_summary":
#                     result = get_late_orders_summary(**args)
#                     st.success("GPT called the function and received:")
#                     st.dataframe(pd.DataFrame(result))
#             else:
#                 st.info(choice.message.content)

#         except Exception as e:
#             st.error(f"Function call failed: {e}")

# with st.expander("💬 Ask GPT: Supply & Demand Audit Q&A"):
#     user_prompt = st.text_input(
#         "Ask a question about shortages, excess, lead time, scrap, safety stock, or late orders:"
#     )

#     if st.button("Ask GPT (Unified)"):
#         client = OpenAIClient(
#             api_key=openai_api_key, organization="org-3Va0Uv9V3lCF4EWsBURKlCAG"
#         )

#         try:
#             response = client.chat.completions.create(
#                 model="gpt-4o",
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": "You are a helpful supply chain audit assistant. Answer questions by choosing the correct function from the list provided. Do not guess. If the user asks something you can't fulfill, politely say it's outside scope.",
#                     },
#                     {"role": "user", "content": user_prompt},
#                 ],
#                 functions=all_function_specs,
#                 function_call="auto",
#                 temperature=0.4,
#             )

#             choice = response.choices[0]

#             if choice.finish_reason == "function_call":
#                 name = choice.message.function_call.name
#                 args = json.loads(choice.message.function_call.arguments)
#                 result = route_gpt_function_call(name, args)

#                 if isinstance(result, str) and result.startswith("⚠️"):
#                     st.warning(result)
#                 else:
#                     st.success(f"✅ GPT called `{name}` and returned:")
#                     st.dataframe(pd.DataFrame(result))

#             else:
#                 st.warning(
#                     "❌ GPT could not match your question to a supported metric function.\n\n"
#                     "Try asking about shortages, excess inventory, lead time, scrap, safety stock, or late orders."
#                 )

#         except Exception as e:
#             st.error(f"Function call failed: {e}")
