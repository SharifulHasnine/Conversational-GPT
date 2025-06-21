import streamlit as st
import mysql.connector
import requests
import os
import re

# --- COLUMN NAMES (single line for reference) ---
COLUMN_NAMES = "Sale_Date, Customer_Name, Customer_Address, Customer_City, Customer_State, Product_Name, Product_Category, Product_Specification, Unit_of_Measure, Quantity_Sold, Unit_Price, Total_Sales_Amount, Discount_Amount, Tax_Amount, Net_Sales_Amount, Salesperson_Name, Salesperson_Contact, Store_Name, Store_Region, Payment_Method, Transaction_Status, Currency_Code, Sale_Month_Year, Sale_Day, Sale_Quarter, Sales Month, Sales Year, Sales Quarter"

# --- CONFIG ---
# Using st.secrets.get() ensures fallback if secrets.toml isn't fully configured
DB_HOST = st.secrets.get("DB_HOST", "localhost")
DB_USER = st.secrets.get("DB_USER", "root")
DB_PASSWORD = st.secrets.get("DB_PASSWORD", "")
DB_NAME = st.secrets.get("DB_NAME", "SalesChatBot")
OLLAMA_URL = st.secrets.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL", "llama3")

# --- SCHEMA CONTEXT (IMPROVED - CONCISE) ---
SALESFACT_SCHEMA = (
    "The `sales_fact_data` table (aliased as SalesFact) contains sales transaction data. "
    "Here's a concise overview of its key columns and usage instructions:\n\n"

    # --- COLUMN DEFINITIONS ---
    "**COLUMNS & TYPES:**\n"
    "• `Sale_Date`: DATETIME (Full transaction timestamp, use sparingly for time-series unless day-level detail is needed).\n"
    "• `Customer_Name`, `Customer_City`, `Customer_State`: VARCHAR(255) (Customer details).\n"
    "• `Product_Name`, `Product_Category`, `Product_Specification`: VARCHAR(255) (Product details).\n"
    "• `Quantity_Sold`: INT (Units sold).\n"
    "• `Unit_Price`: DECIMAL(10, 2).\n"
    "• `Net_Sales_Amount`: DECIMAL(10, 2) (Primary metric for revenue. Always SUM this for total sales).\n"
    "• `Discount_Amount`, `Tax_Amount`: DECIMAL(10, 2) (Financial adjustments).\n"
    "• `Salesperson_Name`, `Salesperson_Contact`: VARCHAR(255) (Salesperson details).\n"
    "• `Store_Name`, `Store_Region`: VARCHAR(255) (Store location).\n"
    "• `Payment_Method`, `Transaction_Status`: VARCHAR(255) (Payment & transaction status).\n"
    "• `Currency_Code`: VARCHAR(255).\n"

    # --- TIME DIMENSION (CRITICAL USAGE) ---
    "**TIME DIMENSIONS (PRIORITY FOR FILTERING/GROUPING):**\n"
    "• `Sales Month`: VARCHAR(255) (e.g., 'Jan', 'Feb'). Use `ORDER BY FIELD(\\`Sales Month\\`, 'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')` for chronological order.\n"
    "• `Sales Year`: INT (e.g., 2024). \n"
    "• `Sales Quarter`: VARCHAR(255) (e.g., 'Q1', 'Q2').\n"
    "**IMPORTANT:**\n"
    "  - Always use `Sales Year`, `Sales Quarter`, `Sales Month` for year/quarter/month filtering and grouping.\n"
    "  - DO NOT use `Sale_Date` with `EXTRACT()` or `YEAR()`, `MONTH()`, `QUARTER()` functions for these purposes.\n"
    "  - DO NOT use `Sale_Month_Year` or the original `Sale_Quarter` columns.\n"
    "  - Always use backticks (`) for column names with spaces, e.g., ``Sales Year``.\n"

    # --- GENERAL GUIDELINES ---
    "**GENERAL SQL GUIDELINES:**\n"
    "• Only use the `SalesFact` table. Do NOT use or join any other tables.\n"
    "• Use exact column names including capitalization.\n"
    "• For ranking (e.g., `RANK()`, `DENSE_RANK()`), use a subquery or CTE. Do NOT use window functions directly in `SELECT` with `GROUP BY`.\n"
    "• Format output numbers as currency (৳) and add line breaks in answers for clarity.\n"

    # --- EXAMPLE QUERIES (CONCISE) ---
    "**COMMON QUERY PATTERNS:**\n"
    "1.  **Total Sales (Net) for a Specific Year & Quarter (e.g., 2024 Q1):**\n"
    "    ```sql\n"
    "    SELECT SUM(Net_Sales_Amount) FROM SalesFact WHERE `Sales Year` = 2024 AND `Sales Quarter` = 'Q1';\n"
    "    ```\n"
    "2.  **Top N Products by Revenue (e.g., Top 5 in 2024):**\n"
    "    ```sql\n"
    "    SELECT Product_Name, SUM(Net_Sales_Amount) AS TotalRevenue\n"
    "    FROM SalesFact\n"
    "    WHERE `Sales Year` = 2024\n"
    "    GROUP BY Product_Name\n"
    "    ORDER BY TotalRevenue DESC LIMIT 5;\n"
    "    ```\n"
    "3.  **Monthly Revenue Trend (e.g., for 2024):**\n"
    "    ```sql\n"
    "    SELECT `Sales Month`, SUM(Net_Sales_Amount) AS MonthlySales\n"
    "    FROM SalesFact\n"
    "    WHERE `Sales Year` = 2024\n"
    "    GROUP BY `Sales Month`\n"
    "    ORDER BY FIELD(`Sales Month`, 'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec');\n"
    "    ```\n"
    "4.  **Quarterly Year-over-Year Comparison (e.g., 2023 vs 2024):**\n"
    "    ```sql\n"
    "    SELECT `Sales Quarter`,\n"
    "           SUM(CASE WHEN `Sales Year` = 2023 THEN Net_Sales_Amount ELSE 0 END) AS Revenue_2023,\n"
    "           SUM(CASE WHEN `Sales Year` = 2024 THEN Net_Sales_Amount ELSE 0 END) AS Revenue_2024\n"
    "    FROM SalesFact\n"
    "    WHERE `Sales Year` IN (2023, 2024)\n"
    "    GROUP BY `Sales Quarter` ORDER BY `Sales Quarter`;\n"
    "    ```\n"
    "5.  **Sales by Store Region & Salesperson:**\n"
    "    ```sql\n"
    "    SELECT Store_Region, Salesperson_Name, SUM(Net_Sales_Amount) AS TotalSales\n"
    "    FROM SalesFact\n"
    "    GROUP BY Store_Region, Salesperson_Name\n"
    "    ORDER BY Store_Region, TotalSales DESC;\n"
    "    ```\n"
    "6.  **Count Transactions by Status:**\n"
    "    ```sql\n"
    "    SELECT Transaction_Status, COUNT(*) AS TransactionCount\n"
    "    FROM SalesFact\n"
    "    GROUP BY Transaction_Status;\n"
    "    ```\n"
)

def parse_ollama_response(response_text):
    """
    Parses the combined Ollama response to extract SQL and explanation.
    Expected format:
    SQL:
    ```sql
    ...
    ```
    Explanation:
    ...
    """
    sql_match = re.search(r"SQL:\s*```sql\n(.*?)```", response_text, re.DOTALL)
    explanation_match = re.search(r"Explanation:\s*(.*)", response_text, re.DOTALL)

    sql_query = sql_match.group(1).strip() if sql_match else ""
    explanation = explanation_match.group(1).strip() if explanation_match else ""

    # Clean up any remaining backticks or markdown from SQL
    sql_query = re.sub(r"```[a-zA-Z]*\n?", "", sql_query)
    sql_query = re.sub(r"```", "", sql_query)
    sql_query = sql_query.strip("` \n")

    return sql_query, explanation

# --- Database Connection with Caching ---
@st.cache_resource(ttl=3600) # Cache connection for 1 hour
def get_db_connection_cached():
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        return conn
    except mysql.connector.Error as err:
        st.error(f"Database connection error: {err}")
        return None

def fetch_sales_facts(query):
    conn = get_db_connection_cached()
    if conn is None:
        return None # Return None if connection failed

    # Ping the connection to ensure it's still alive, reconnect if necessary
    try:
        conn.ping(reconnect=True, attempts=3)
    except mysql.connector.Error as err:
        st.error(f"Database connection lost, attempting to re-establish... Error: {err}")
        # Clear the cache to force a new connection on next attempt
        st.cache_resource.clear()
        conn = get_db_connection_cached() # Try to get a new connection
        if conn is None:
            return None # If reconnection also failed, return None

    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        return results
    except mysql.connector.Error as err:
        st.error(f"SQL query error: {err}\n\nQuery:\n```sql\n{query}\n```")
        return None

# --- LLM ---
def ask_ollama(prompt):
    """
    Sends a prompt to Ollama and returns the full response string.
    Note on Streaming: For true token-by-token streaming, you would set 'stream': True
    in the JSON payload and then iterate over response.iter_lines(),
    feeding each chunk to Streamlit's st.write_stream(). This example
    remains blocking for simplicity but the prompt size reduction is applied.
    """
    try:
        response = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False})
        response.raise_for_status()
        return response.json().get("response", "No response.")
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Make sure Ollama server is running at " + OLLAMA_URL
    except requests.exceptions.RequestException as e:
        return f"Error from Ollama: {e}"
    except Exception as e:
        return f"An unexpected error occurred with Ollama: {e}"


# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Conversational GPT", page_icon="💡", layout="wide") # Changed icon

col1, col2 = st.columns([8, 2])
with col1:
    st.markdown("<h4 style='margin-bottom:0.65rem;'>Conversational GPT: Business insight, at the speed of thought.🤖</h4>", unsafe_allow_html=True)
with col2:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.user_question_input = "" # Clear input box immediately
        st.rerun()

# Display column names at the top, wrapped for readability
st.markdown("**Available Information:**")

with st.expander("Available Columns:"): # Changed expander title
    col_names = [col.strip() for col in COLUMN_NAMES.split(",")]
    for i in range(0, len(col_names), 5):
        cols = st.columns(5)
        for j, col in enumerate(col_names[i:i+5]):
            if j < len(cols):
                with cols[j]:
                    st.markdown(f"<div style='background:#f1f3f4;border-radius:6px;padding:6px 8px;margin:2px 0;text-align:center;border:1px solid #e0e0e0;font-size:0.7em;'>{col}</div>", unsafe_allow_html=True)

# Make the sidebar wider
st.markdown(
    """
    <style>
    [data-testid="stSidebar] {
        min-width: 450px;
        max-width: 500px;
        width: 400px;
    }
    [data-testid="stSidebar"] * {
        font-size: 0.85rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- SESSION STATE INIT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "user_question_input" not in st.session_state:
    st.session_state["user_question_input"] = ""

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

# Clear input safely before widget creation
if st.session_state.clear_input:
    st.session_state["user_question_input"] = ""
    st.session_state.clear_input = False
    st.rerun()

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 💡 Suggested Insights")
    grouped_questions = {
    "📦 Product Performance": [
        "What are the top 5 products by revenue in 2024?",
        "Which product sells best in each state?",
        "Which product has the highest average discount?",
        "What is the average price per unit by product?",
        "Which products had the highest unit prices?",
        "Which products had a drop in sales from Q1 to Q2 in 2024?",
        "Break down product sales by store.",
        "What's the effective tax rate per product?",
        "List high-value single-item purchases."
    ],
    "📈 Time Trends": [
        "Show monthly sales trend for 2024.",
        "Compare quarterly revenue for 2023 and 2024.",
        "Show year-over-year revenue growth by product.",
        "Show sales by region and quarter.",
        "How many units sold per category each year?",
        "Show average monthly sales across years.",
        "Which quarter historically performs the best?",
        "Show sales vs discounts over quarters.",
        "What was the best sales day overall?"
    ],
    "💰 Financial Metrics": [
        "How much revenue did each payment method generate?",
        "Compare revenue by store for the 'Electronics' category.",
        "Which product category had the highest total discount?",
        "What % of total revenue comes from each product category?",
        "Total quantity sold by unit of measure?",
        "How much tax was collected per year?",
        "Which product categories paid the most tax?",
        "Which category contributes the most revenue?",
        "Which products gave the highest estimated profit?"
    ],
    "🧍‍♂️ Customer Behavior": [
        "List customers who made repeat purchases.",
        "Who are the top 10 customers by lifetime value?",
        "What's the average order value by customer city?",
        "How many unique customers did we have each year?",
        "Who are the most frequent buyers?",
        "Which customers were inactive in 2025?",
        "Who bought from multiple product categories?"
    ],
    "🧑‍💼 Sales & Stores": [
        "Who are the top 10 performing salespeople by revenue?",
        "Show sales by region and salesperson.",
        "Rank stores by revenue per year.",
        "Rank stores by revenue for each quarter.",
        "Which store had the highest net sales?"
    ],
    "🌍 Geographic": [
        "Which 3 cities had the highest quantity sold in Q1 2025?",
        "Show sales by customer city.",
        "Show revenue by region and year.",
        "Which states had above-average sales?",
        "What is average customer spend by region?"
    ],
    "💳 Payments & Transactions": [
        "Break down sales by payment method.",
        "How often is each payment method used?",
        "How many transactions were successful vs. failed?",
        "Break down payment methods by region.",
        "What % of transactions were completed successfully?",
        "Summarize sales by transaction status (completed, failed, etc.)"
    ],
    "🧪 Operational Checks": [
        "Find transactions with zero or negative quantity.",
        "List orders where net sales amount was zero.",
        "List monthly sales for ML forecasting preparation."
    ]
}

    for group, questions in grouped_questions.items():
        with st.expander(group):
            for q in questions:
                if st.button(q, key=f"sidebar_{q}"):
                    st.session_state["user_question_input"] = q
                    st.rerun()

# --- INPUT AREA ---
st.markdown("<h5 style='margin-bottom:0.4rem;'>💬 GPT that helps you see</h5>", unsafe_allow_html=True)
input_col, send_col = st.columns([6, 1])

with input_col:
    user_question = st.text_input(
        "Ask your question:",
        value=st.session_state["user_question_input"],
        key="user_question_input",
        label_visibility="collapsed",
        placeholder="e.g., Show sales by region for 2024"
    )
with send_col:
    ask_clicked = st.button("Send", use_container_width=True)

# --- ON SEND ---
if (ask_clicked or (user_question and st.session_state["user_question_input"] != "")) and user_question.strip():
    user_question = user_question.strip()
    st.session_state.clear_input = True  # Set flag to clear input on next rerun
    with st.spinner("🤔 Thinking..."):
        # Combined SQL Generation and Explanation Prompt
        combined_prompt = (
            f"You are an expert SQL assistant and business analyst. {SALESFACT_SCHEMA}\n\n"
            f"Given the user's question: '{user_question}', first generate a MySQL query. "
            f"Then, after the SQL, provide a clear business explanation of the answer, "
            f"formatting numbers as currency (৳) where appropriate and adding line breaks for clarity. "
            f"Your output MUST be in the following format:\n\n"
            f"SQL:\n```sql\n[YOUR SQL QUERY HERE]\n```\n"
            f"Explanation:\n[YOUR BUSINESS EXPLANATION HERE]"
        )
        ollama_response = ask_ollama(combined_prompt)
        sql_query, answer_explanation = parse_ollama_response(ollama_response)

        data = None
        answer = "I couldn't process your request." # Default answer

        if sql_query: # Only proceed if SQL was successfully generated
            try:
                data = fetch_sales_facts(sql_query)
                if data is not None: # Check if data fetching was successful
                    if not answer_explanation: # Fallback if explanation wasn't parsed
                        answer = f"Here is the data for your query.\n\nQuery:\n```sql\n{sql_query}\n```"
                    else:
                        answer = answer_explanation
                else:
                    answer = "I could not retrieve data for the generated query. Please check the SQL query for errors or database connection."

            except Exception as e: # Catch any other unexpected errors during data fetching
                answer = f"An unexpected error occurred while processing results: {e}\n\nSQL tried:```sql\n{sql_query}\n```"
        else:
            answer = f"I could not generate a valid SQL query from your question. Please try rephrasing. Debug Info: Ollama Response:\n{ollama_response}"

    st.session_state.chat_history.append({
        "question": user_question,
        "sql": sql_query,
        "data": data,
        "answer": answer
    })
    st.rerun() # Rerun to update chat history and clear input box

# --- CHAT HISTORY DISPLAY ---
for chat in reversed(st.session_state.chat_history):
    st.markdown(f"<div style='padding:12px;border-radius:10px;margin-bottom:10px;background:#e0f7fa;text-align:right;'><b>You:</b> {chat['question']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='padding:12px;border-radius:10px;margin-bottom:10px;background:#f1f8e9;text-align:left;'><b>Bot:</b><br>{chat['answer']}</div>", unsafe_allow_html=True)
    if chat["data"] is not None:
        st.dataframe(chat["data"])
    st.caption(f"🧠 SQL used: `{chat['sql']}`")