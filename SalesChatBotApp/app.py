import streamlit as st # type: ignore
import mysql.connector # type: ignore
import requests # type: ignore
import os
import re
import warnings
from forecasting import SalesForecaster, get_forecasting_insights, TimeSeriesAnalyzer
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*OpenSSL.*')
warnings.filterwarnings('ignore', message='.*urllib3.*')

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
    Handles multiple response formats:
    1. Expected format with SQL: and Explanation: markers
    2. SQL wrapped in ```sql``` blocks
    3. SQL followed by explanation without specific markers
    """
    # First try the expected format
    sql_match = re.search(r"SQL:\s*```sql\n(.*?)```", response_text, re.DOTALL)
    explanation_match = re.search(r"Explanation:\s*(.*)", response_text, re.DOTALL)

    if sql_match:
        sql_query = sql_match.group(1).strip()
    else:
        # Try to find SQL in code blocks
        sql_match = re.search(r"```sql\n(.*?)```", response_text, re.DOTALL)
        if sql_match:
            sql_query = sql_match.group(1).strip()
        else:
            # Try to find SQL after "Here is the SQL query" or similar phrases
            sql_match = re.search(r"(?:Here is the SQL query|SQL query|Query):\s*(.*?)(?:\n\n|\nExplanation|\nHere's what|$)", response_text, re.DOTALL | re.IGNORECASE)
            if sql_match:
                sql_query = sql_match.group(1).strip()
            else:
                # Last resort: try to extract anything that looks like SQL
                lines = response_text.split('\n')
                sql_lines = []
                in_sql = False
                for line in lines:
                    if any(keyword in line.upper() for keyword in ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'LIMIT']):
                        in_sql = True
                    if in_sql:
                        sql_lines.append(line)
                        if line.strip().endswith(';'):
                            break
                sql_query = '\n'.join(sql_lines).strip() if sql_lines else ""

    if explanation_match:
        explanation = explanation_match.group(1).strip()
    else:
        # Try to find explanation after SQL
        if sql_query:
            # Find text after the SQL query
            sql_end = response_text.find(sql_query) + len(sql_query)
            remaining_text = response_text[sql_end:].strip()
            # Remove any remaining code blocks or SQL markers
            explanation = re.sub(r'```.*?```', '', remaining_text, flags=re.DOTALL)
            explanation = re.sub(r'SQL:.*?```.*?```', '', explanation, flags=re.DOTALL)
            explanation = explanation.strip()
        else:
            explanation = ""

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
    except Exception as e:
        st.error(f"MySQL is not installed or not running. Please install MySQL and ensure it's running.\nError: {e}")
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

# --- CHART CREATION FUNCTIONS ---
def create_chart(data, chart_type, x_column=None, y_column=None, color_column=None, title="Chart"):
    """Create various chart types based on data and user selection"""
    
    if data is None or len(data) == 0:
        return None
    
    df = pd.DataFrame(data)
    
    try:
        if chart_type == "Bar Chart":
            if x_column and y_column:
                fig = px.bar(df, x=x_column, y=y_column, color=color_column, title=title)
            else:
                # Auto-detect numeric columns for y-axis
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = px.bar(df, x=df.columns[0], y=numeric_cols[0], title=title)
                else:
                    fig = px.bar(df, x=df.columns[0], title=title)
        
        elif chart_type == "Line Chart":
            if x_column and y_column:
                fig = px.line(df, x=x_column, y=y_column, color=color_column, title=title)
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = px.line(df, x=df.columns[0], y=numeric_cols[0], title=title)
                else:
                    fig = px.line(df, x=df.columns[0], title=title)
        
        elif chart_type == "Scatter Plot":
            if x_column and y_column:
                fig = px.scatter(df, x=x_column, y=y_column, color=color_column, title=title)
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], title=title)
                else:
                    return None
        
        elif chart_type == "Pie Chart":
            if x_column and y_column:
                fig = px.pie(df, values=y_column, names=x_column, title=title)
            else:
                # For pie chart, we need categorical and numeric columns
                categorical_cols = df.select_dtypes(include=['object']).columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                    fig = px.pie(df, values=numeric_cols[0], names=categorical_cols[0], title=title)
                else:
                    return None
        
        elif chart_type == "Histogram":
            if x_column:
                fig = px.histogram(df, x=x_column, color=color_column, title=title)
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = px.histogram(df, x=numeric_cols[0], title=title)
                else:
                    return None
        
        elif chart_type == "Box Plot":
            if x_column and y_column:
                fig = px.box(df, x=x_column, y=y_column, color=color_column, title=title)
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = px.box(df, y=numeric_cols[0], title=title)
                else:
                    return None
        
        elif chart_type == "Heatmap":
            # Create correlation heatmap for numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) >= 2:
                corr_matrix = numeric_df.corr()
                fig = px.imshow(corr_matrix, 
                              labels=dict(color="Correlation"),
                              title=f"{title} - Correlation Heatmap")
            else:
                return None
        
        elif chart_type == "Area Chart":
            if x_column and y_column:
                fig = px.area(df, x=x_column, y=y_column, color=color_column, title=title)
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = px.area(df, x=df.columns[0], y=numeric_cols[0], title=title)
                else:
                    return None
        
        elif chart_type == "Violin Plot":
            if x_column and y_column:
                fig = px.violin(df, x=x_column, y=y_column, color=color_column, title=title)
            else:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = px.violin(df, y=numeric_cols[0], title=title)
                else:
                    return None
        
        elif chart_type == "Correlation Matrix":
            # Create correlation heatmap for numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if len(numeric_df.columns) >= 2:
                corr_matrix = numeric_df.corr()
                fig = px.imshow(corr_matrix, 
                              labels=dict(color="Correlation"),
                              title=f"{title} - Correlation Matrix",
                              color_continuous_scale='RdBu',
                              aspect="auto")
            else:
                return None
        
        elif chart_type == "Time Series":
            # For time series, we need date/time columns
            date_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'month', 'year', 'time'])]
            if date_cols and y_column:
                fig = px.line(df, x=date_cols[0], y=y_column, color=color_column, title=title)
            else:
                # Try to use first column as x-axis
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    fig = px.line(df, x=df.columns[0], y=numeric_cols[0], title=title)
                else:
                    return None
        
        elif chart_type == "Multi-Axis Chart":
            # Create a chart with multiple y-axes
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(x=df[df.columns[0]], y=df[numeric_cols[0]], name=numeric_cols[0]),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(x=df[df.columns[0]], y=df[numeric_cols[1]], name=numeric_cols[1]),
                    secondary_y=True
                )
                
                fig.update_layout(title=title)
                fig.update_xaxes(title_text=df.columns[0])
                fig.update_yaxes(title_text=numeric_cols[0], secondary_y=False)
                fig.update_yaxes(title_text=numeric_cols[1], secondary_y=True)
            else:
                return None
        
        else:
            return None
        
        # Update layout for better appearance
        fig.update_layout(
            title_x=0.5,
            title_font_size=16,
            height=500,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating chart: {e}")
        return None

def get_chart_suggestions(data):
    """Suggest appropriate chart types based on data characteristics"""
    if data is None or len(data) == 0:
        return []
    
    df = pd.DataFrame(data)
    suggestions = []
    
    # Count columns by type
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(numeric_cols) >= 2:
        suggestions.extend(["Scatter Plot", "Heatmap", "Correlation Matrix", "Multi-Axis Chart"])
    
    if len(numeric_cols) >= 1:
        suggestions.extend(["Bar Chart", "Line Chart", "Histogram", "Box Plot", "Area Chart", "Violin Plot"])
    
    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        suggestions.append("Pie Chart")
    
    # Check for time-related columns
    date_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'month', 'year', 'time'])]
    if date_cols and len(numeric_cols) >= 1:
        suggestions.append("Time Series")
    
    # Check for geographic columns
    geo_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['region', 'city', 'state', 'country', 'location'])]
    if geo_cols and len(numeric_cols) >= 1:
        suggestions.extend(["Bar Chart", "Pie Chart"])
    
    # Check for product/category columns
    product_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['product', 'category', 'item'])]
    if product_cols and len(numeric_cols) >= 1:
        suggestions.extend(["Bar Chart", "Box Plot", "Violin Plot"])
    
    return list(set(suggestions))  # Remove duplicates

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Conversational GPT", page_icon="💡", layout="wide") # Changed icon

# # Database connection status
# db_status = get_db_connection_cached()
# if db_status is None:
#     st.warning("⚠️ **Database Connection Issue:** MySQL is not installed or not running. The app will generate SQL queries but cannot execute them. Please install MySQL to use full functionality.")
# else:
#     st.success("✅ **Database Connected:** MySQL is running and connected successfully.")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat Assistant","📈 Charts & Analytics", "📊 Time Series Analysis", "🔮 ML Forecasting"])

with tab1:
    col1, col2 = st.columns([8, 2])
    with col1:
        st.markdown("<h4 style='margin-bottom:0.65rem;'>Conversational GPT: Business insight, at the speed of thought.🤖</h4>", unsafe_allow_html=True)
    with col2:
        if st.button("Clear Chat", use_container_width=True, key="clear_chat_tab1"):
            st.session_state.chat_history = []
            # Clear the input using the key
            if "user_question_input" in st.session_state:
                st.session_state["user_question_input"] = ""
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
                        st.markdown(f"<div style='background:#f1f3f4;border-radius:6px;padding:6px 8px;margin:2px 0;text-align:center;border:1px solid #e0e0e0;font-size:1.0em;'>{col}</div>", unsafe_allow_html=True)

    # Make the sidebar wider
    st.markdown(
        """
        <style>
        [data-testid="stSidebar] {
            min-width: 550px;
            max-width: 600px;
            width: 500px;
        }
        [data-testid="stSidebar"] * {
            font-size: 1.1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- SESSION STATE INIT ---
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "clear_input" not in st.session_state:
        st.session_state.clear_input = False

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
                        # Use a different approach to set the input value
                        st.session_state["sidebar_question"] = q
                        st.rerun()

    # --- INPUT AREA ---
    st.markdown("<h5 style='margin-bottom:0.8rem;'>💬 Talk to your data</h5>", unsafe_allow_html=True)
    input_col, send_col = st.columns([6, 1])

    with input_col:
        # Check if there's a sidebar question to set
        if "sidebar_question" in st.session_state:
            # Set the input value using the key
            st.session_state["user_question_input"] = st.session_state["sidebar_question"]
            # Clear the sidebar question
            del st.session_state["sidebar_question"]
        
        user_question = st.text_input(
            "Ask your question:",
            key="user_question_input",
            label_visibility="collapsed",
            placeholder="e.g., Show sales by region for 2024"
        )
    with send_col:
        ask_clicked = st.button("Send", use_container_width=True)

    # --- ON SEND ---
    if ask_clicked and user_question.strip():
        user_question = user_question.strip()
        
        with st.spinner("🤔 Thinking..."):
            # Combined SQL Generation and Explanation Prompt
            combined_prompt = (
                f"You are an expert SQL assistant and business analyst. {SALESFACT_SCHEMA}\n\n"
                f"Given the user's question: '{user_question}', generate a MySQL query and provide a business explanation.\n\n"
                f"**IMPORTANT: Your response MUST follow this EXACT format:**\n\n"
                f"SQL:\n```sql\n[YOUR SQL QUERY HERE]\n```\n"
                f"Explanation:\n[YOUR BUSINESS EXPLANATION HERE]\n\n"
                f"**Requirements:**\n"
                f"- Start with 'SQL:' followed by your query in a code block\n"
                f"- End with 'Explanation:' followed by your business analysis\n"
                f"- Format numbers as currency (৳) where appropriate\n"
                f"- Add line breaks for clarity\n"
                f"- Do NOT include any other text before 'SQL:' or after 'Explanation:'"
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
                        # Database connection failed but we have SQL
                        answer = f"✅ SQL Query Generated Successfully!\n\n```sql\n{sql_query}\n```\n\n❌ **Database Connection Issue:**\nThe SQL query was generated correctly, but I couldn't connect to the database to execute it. Please ensure:\n1. MySQL is installed and running\n2. Database credentials are correct\n3. The 'SalesChatBot' database exists\n\n**Generated SQL Explanation:**\n{answer_explanation if answer_explanation else 'Query generated but explanation not available.'}"

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
            # Create tabs for data and charts
            data_tab, chart_tab = st.tabs(["📊 Data", "📈 Charts"])
            
            with data_tab:
                st.dataframe(chat["data"])
            
            with chart_tab:
                if len(chat["data"]) > 0:
                    # Chart configuration
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Get available chart types
                        available_charts = get_chart_suggestions(chat["data"])
                        all_charts = ["Bar Chart", "Line Chart", "Scatter Plot", "Pie Chart", 
                                     "Histogram", "Box Plot", "Heatmap", "Area Chart", "Violin Plot"]
                        
                        chart_type = st.selectbox(
                            "Chart Type",
                            available_charts if available_charts else all_charts,
                            help="Select the type of chart to display",
                            key=f"chat_chart_type_{chat['question'][:20]}"
                        )
                    
                    with col2:
                        # Column selection
                        df = pd.DataFrame(chat["data"])
                        columns = list(df.columns)
                        
                        x_column = st.selectbox(
                            "X-Axis Column",
                            ["Auto"] + columns,
                            help="Select column for X-axis (or Auto for automatic selection)",
                            key=f"chat_x_column_{chat['question'][:20]}"
                        )
                        
                        y_column = st.selectbox(
                            "Y-Axis Column", 
                            ["Auto"] + columns,
                            help="Select column for Y-axis (or Auto for automatic selection)",
                            key=f"chat_y_column_{chat['question'][:20]}"
                        )
                    
                    with col3:
                        color_column = st.selectbox(
                            "Color Column (Optional)",
                            ["None"] + columns,
                            help="Select column for color coding",
                            key=f"chat_color_column_{chat['question'][:20]}"
                        )
                    
                    # Chart title
                    chart_title = st.text_input(
                        "Chart Title",
                        value=f"Visualization: {chat['question'][:50]}...",
                        help="Enter a title for the chart",
                        key=f"chat_chart_title_{chat['question'][:20]}"
                    )
                    
                    # Create and display chart
                    if st.button("🔄 Generate Chart", type="secondary", key=f"chat_generate_chart_{chat['question']}"):
                        with st.spinner("Creating chart..."):
                            # Convert "Auto" and "None" selections
                            x_col = None if x_column == "Auto" else x_column
                            y_col = None if y_column == "Auto" else y_column
                            color_col = None if color_column == "None" else color_column
                            
                            fig = create_chart(
                                chat["data"], 
                                chart_type, 
                                x_col, 
                                y_col, 
                                color_col, 
                                chart_title
                            )
                            
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Download chart as HTML
                                html_string = fig.to_html(include_plotlyjs='cdn')
                                st.download_button(
                                    label="📥 Download Chart (HTML)",
                                    data=html_string,
                                    file_name=f"chart_{chart_type.lower().replace(' ', '_')}.html",
                                    mime="text/html"
                                )
                            else:
                                st.warning("⚠️ Could not create chart with the selected parameters. Try different column selections.")
                    
                    # Show quick chart suggestions
                    if available_charts:
                        st.info(f"💡 **Suggested charts for this data:** {', '.join(available_charts[:3])}")
                    else:
                        st.info("💡 **Tip:** Try different chart types and column selections to visualize your data.")
                else:
                    st.info("No data available for charting.")
        
        st.caption(f"🧠 SQL used: `{chat['sql']}`")

# --- FORECASTING TAB ---
with tab4:
    st.markdown("<h4 style='margin-bottom:0.65rem;'>🔮 ML Sales Forecasting</h4>", unsafe_allow_html=True)
    st.markdown("Generate machine learning forecasts for sales by product, region, month, and quarter.")
    
    # Initialize forecaster
    if "forecaster" not in st.session_state:
        st.session_state.forecaster = SalesForecaster()
    
    # Get historical data for forecasting
    with st.spinner("Loading historical data for forecasting..."):
        historical_query = "SELECT * FROM SalesFact ORDER BY `Sales Year`, `Sales Month`"
        historical_data = fetch_sales_facts(historical_query)
    
    if historical_data is None:
        st.error("❌ Unable to load historical data for forecasting. Please ensure the database is connected.")
    else:
        st.success(f"✅ Loaded {len(historical_data)} historical records for forecasting.")
        
        # Forecasting configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_type = st.selectbox(
                "Forecast Type",
                ["Overall Sales", "By Product", "By Region", "By Product & Region"],
                help="Choose what to forecast",
                key="forecast_type_select"
            )
        
        with col2:
            model_type = st.selectbox(
                "ML Model",
                ["Gradient Boosting", "Random Forest", "Linear Regression", "Ridge Regression", "Lasso Regression"],
                help="Choose the machine learning algorithm",
                key="model_type_select"
            )
        
        with col3:
            forecast_periods = st.slider(
                "Forecast Periods (Months)",
                min_value=3,
                max_value=24,
                value=12,
                help="Number of months to forecast"
            )
        
        # Prepare data based on forecast type
        if st.button("🚀 Generate Forecast", type="primary", key="generate_forecast"):
            with st.spinner("Preparing data and training ML model..."):
                # Prepare forecasting data
                prepared_data = st.session_state.forecaster.prepare_forecasting_data(historical_data)
                
                if prepared_data is not None and not prepared_data.empty:
                    # Determine grouping based on forecast type
                    group_by = None
                    if forecast_type == "By Product":
                        group_by = "Product_Name"
                    elif forecast_type == "By Region":
                        group_by = "Store_Region"
                    elif forecast_type == "By Product & Region":
                        # For this case, we'll need to handle it differently
                        st.info("Product & Region forecasting will be implemented in the next version.")
                        group_by = None
                    
                    # Train model
                    model, scaler, metrics = st.session_state.forecaster.train_forecasting_model(
                        prepared_data, 
                        group_by=group_by,
                        model_type='gradient_boosting' if model_type == "Gradient Boosting" else 
                                  'random_forest' if model_type == "Random Forest" else
                                  'linear' if model_type == "Linear Regression" else
                                  'ridge' if model_type == "Ridge Regression" else
                                  'lasso' if model_type == "Lasso Regression" else 'gradient_boosting'
                    )
                    
                    if model is not None:
                        # Generate forecast
                        forecast_data = st.session_state.forecaster.generate_forecast(
                            model, scaler, prepared_data, periods=forecast_periods, group_by=group_by
                        )
                        
                        if forecast_data is not None:
                            # Display results
                            st.success("✅ Forecast generated successfully!")
                            
                            # Show model metrics
                            with st.expander("📊 Model Performance Metrics"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Mean Absolute Error", f"৳{metrics['MAE']:,.2f}")
                                with col2:
                                    st.metric("Root Mean Square Error", f"৳{metrics['RMSE']:,.2f}")
                                with col3:
                                    st.metric("R² Score", f"{metrics['R2']:.3f}")
                                
                                # Show cross-validation results
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Cross-Validation R²", f"{metrics['CV_R2_Mean']:.3f}")
                                with col2:
                                    st.metric("CV R² Std Dev", f"±{metrics['CV_R2_Std']:.3f}")
                                
                                # Show feature importance if available
                                if metrics['Feature_Importance']:
                                    st.subheader("🔍 Top Feature Importance")
                                    feature_importance = pd.DataFrame(
                                        list(metrics['Feature_Importance'].items()),
                                        columns=['Feature', 'Importance']
                                    ).sort_values('Importance', ascending=False).head(10)
                                    st.bar_chart(feature_importance.set_index('Feature'))
                            
                            # Show forecast visualization
                            st.subheader("📈 Forecast Visualization")
                            fig = st.session_state.forecaster.create_forecast_visualization(
                                historical_data, forecast_data, 
                                title=f"{forecast_type} Forecast ({forecast_periods} months)"
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            
                            # Show forecast data
                            st.subheader("📋 Forecast Data")
                            st.dataframe(forecast_data)
                            
                            # Show insights
                            st.subheader("💡 Forecasting Insights")
                            insights = get_forecasting_insights(historical_data, forecast_data, group_by)
                            st.markdown(insights)
                            
                            # Download forecast data
                            csv = forecast_data.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Forecast Data (CSV)",
                                data=csv,
                                file_name=f"sales_forecast_{forecast_type.lower().replace(' ', '_')}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.error("❌ Failed to generate forecast. Please try with different parameters.")
                    else:
                        st.error("❌ Failed to train model. Please check your data and try again.")
                else:
                    st.error("❌ Insufficient data for forecasting. Please ensure you have enough historical data.")
        
        # Show sample data for reference
        with st.expander("📊 Sample Historical Data"):
            if historical_data:
                sample_df = pd.DataFrame(historical_data[:10])
                st.dataframe(sample_df)

# --- TIME SERIES ANALYSIS TAB ---
with tab3:
    st.markdown("<h4 style='margin-bottom:0.65rem;'>📊 Time Series Analysis</h4>", unsafe_allow_html=True)
    st.markdown("Analyze time series data for sales trends, seasonality, and patterns.")
    
    # Initialize time series analyzer
    if "time_series_analyzer" not in st.session_state:
        st.session_state.time_series_analyzer = TimeSeriesAnalyzer()
    
    # Get historical data for analysis
    with st.spinner("Loading historical data for analysis..."):
        historical_query = "SELECT * FROM SalesFact ORDER BY `Sales Year`, `Sales Month`"
        historical_data = fetch_sales_facts(historical_query)
    
    if historical_data is None:
        st.error("❌ Unable to load historical data for analysis. Please ensure the database is connected.")
    else:
        st.success(f"✅ Loaded {len(historical_data)} historical records for analysis.")
        
        # Analysis configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            analysis_frequency = st.selectbox(
                "Time Frequency",
                ["M", "W", "D"],
                format_func=lambda x: {"M": "Monthly", "W": "Weekly", "D": "Daily"}[x],
                help="Aggregation frequency for time series",
                key="analysis_frequency_select"
            )
        
        with col2:
            group_by = st.selectbox(
                "Group By (Optional)",
                ["None", "Product_Name", "Store_Region", "Product_Category"],
                help="Group analysis by specific dimension",
                key="group_by_select"
            )
        
        with col3:
            group_value = None
            if group_by != "None":
                # Get unique values for the selected group
                group_values = list(set([row[group_by] for row in historical_data if row[group_by]]))
                group_value = st.selectbox(
                    f"Select {group_by}",
                    group_values,
                    help=f"Select specific {group_by} for analysis",
                    key="group_value_select"
                )
        
        # Prepare time series data
        if st.button("🚀 Perform Time Series Analysis", type="primary", key="perform_time_series_analysis"):
            with st.spinner("Preparing time series data and performing analysis..."):
                # Prepare time series data
                ts_data = st.session_state.time_series_analyzer.prepare_time_series_data(
                    historical_data, 
                    freq=analysis_frequency,
                    group_by=group_by if group_by != "None" else None,
                    group_value=group_value
                )
                
                if ts_data is not None and len(ts_data) > 0:
                    st.success(f"✅ Prepared time series data with {len(ts_data)} observations.")
                    
                    # Perform comprehensive analysis
                    with st.spinner("Performing comprehensive time series analysis..."):
                        
                        # 1. Statistical metrics
                        stats = st.session_state.time_series_analyzer.calculate_statistical_metrics(ts_data)
                        
                        # 2. Trend analysis
                        trend = st.session_state.time_series_analyzer.analyze_trend(ts_data)
                        
                        # 3. Seasonality analysis
                        seasonality = st.session_state.time_series_analyzer.analyze_seasonality(ts_data)
                        
                        # 4. Stationarity tests
                        stationarity = st.session_state.time_series_analyzer.test_stationarity(ts_data)
                        
                        # 5. Autocorrelation analysis
                        autocorr = st.session_state.time_series_analyzer.analyze_autocorrelation(ts_data)
                        
                        # 6. Cycle detection
                        cycles = st.session_state.time_series_analyzer.detect_cycles(ts_data)
                        
                        # Display results
                        st.success("✅ Time series analysis completed successfully!")
                        
                        # Create comprehensive visualization
                        st.subheader("📈 Time Series Visualization")
                        fig = st.session_state.time_series_analyzer.create_time_series_plots(
                            ts_data, trend, seasonality, autocorr, cycles
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display comprehensive report
                        st.subheader("📊 Analysis Report")
                        report = st.session_state.time_series_analyzer.generate_time_series_report(
                            ts_data, 
                            group_by=group_by if group_by != "None" else None,
                            group_value=group_value
                        )
                        st.markdown(report)
                        
                        # Display detailed metrics in expandable sections
                        if stats:
                            with st.expander("📈 Statistical Metrics"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Mean", f"৳{stats['mean']:,.2f}")
                                    st.metric("Median", f"৳{stats['median']:,.2f}")
                                    st.metric("Std Dev", f"৳{stats['std']:,.2f}")
                                with col2:
                                    st.metric("Skewness", f"{stats['skewness']:.3f}")
                                    st.metric("Kurtosis", f"{stats['kurtosis']:.3f}")
                                    st.metric("CV (%)", f"{stats['coefficient_of_variation']:.1f}")
                                with col3:
                                    st.metric("Avg Growth Rate", f"{stats['avg_growth_rate']:+.2f}%")
                                    st.metric("Growth Volatility", f"{stats['growth_volatility']:.2f}%")
                                    st.metric("Volatility", f"{stats['volatility']:.2f}%")
                        
                        if trend:
                            with st.expander("📈 Trend Analysis"):
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Trend Direction", trend['trend_direction'])
                                    st.metric("Trend Strength (R²)", f"{trend['trend_strength']:.3f}")
                                with col2:
                                    st.metric("Significance", trend['trend_significance'])
                                    st.metric("P-value", f"{trend['p_value']:.3f}")
                                with col3:
                                    st.metric("Total Change", f"{trend['total_change_percent']:+.1f}%")
                                    st.metric("Monthly Change", f"৳{trend['avg_monthly_change']:,.2f}")
                        
                        if seasonality:
                            with st.expander("🌊 Seasonality Analysis"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Seasonal Strength", f"{seasonality['seasonal_strength']:.3f}")
                                    st.metric("Seasonal Variation", f"{seasonality['seasonal_variation_percent']:.1f}%")
                                with col2:
                                    st.metric("Peak Month", seasonality['peak_month'].strftime('%B %Y'))
                                    st.metric("Trough Month", seasonality['trough_month'].strftime('%B %Y'))
                        
                        if stationarity:
                            with st.expander("📊 Stationarity Tests"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("ADF Test", "Stationary" if stationarity['adf_stationary'] else "Non-stationary")
                                    st.metric("ADF P-value", f"{stationarity['adf_pvalue']:.3f}")
                                with col2:
                                    st.metric("KPSS Test", "Stationary" if stationarity['kpss_stationary'] else "Non-stationary")
                                    st.metric("KPSS P-value", f"{stationarity['kpss_pvalue']:.3f}")
                                st.metric("Overall Assessment", stationarity['overall_stationarity'])
                        
                        if autocorr:
                            with st.expander("🔄 Autocorrelation Analysis"):
                                st.write(f"**Significant ACF lags:** {autocorr['significant_acf_lags'][:10]}")
                                st.write(f"**Significant PACF lags:** {autocorr['significant_pacf_lags'][:10]}")
                                st.write(f"**Significance threshold:** ±{autocorr['significance_threshold']:.3f}")
                        
                        if cycles:
                            with st.expander("🔄 Cycle Detection"):
                                if cycles['dominant_periods']:
                                    st.write(f"**Dominant periods:** {[f'{p:.1f} periods' for p in cycles['dominant_periods']]}")
                                else:
                                    st.write("No significant cycles detected.")
                        
                        # Download time series data
                        ts_df = pd.DataFrame({
                            'Date': ts_data.index,
                            'Sales': ts_data.values
                        })
                        csv = ts_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Time Series Data (CSV)",
                            data=csv,
                            file_name=f"time_series_data_{analysis_frequency}_{group_by if group_by != 'None' else 'overall'}.csv",
                            mime="text/csv"
                        )
                        
                else:
                    st.error("❌ Failed to prepare time series data. Please check your data and parameters.")
        
        # Show sample data for reference
        with st.expander("📊 Sample Historical Data"):
            if historical_data:
                sample_df = pd.DataFrame(historical_data[:10])
                st.dataframe(sample_df)

# --- CHARTS & ANALYTICS TAB ---
with tab2:
    st.markdown("<h4 style='margin-bottom:0.65rem;'>📈 Charts & Analytics</h4>", unsafe_allow_html=True)
    st.markdown("Create advanced visualizations and analytics for your sales data.")
    
    # Get data for visualization
    with st.spinner("Loading data for visualization..."):
        viz_query = "SELECT * FROM SalesFact ORDER BY `Sales Year`, `Sales Month`"
        viz_data = fetch_sales_facts(viz_query)
    
    if viz_data is None:
        st.error("❌ Unable to load data for visualization. Please ensure the database is connected.")
    else:
        st.success(f"✅ Loaded {len(viz_data)} records for visualization.")
        
        # Advanced chart options
        st.subheader("🎨 Advanced Chart Creator")
        
        # Chart type selection
        col1, col2 = st.columns(2)
        
        with col1:
            chart_category = st.selectbox(
                "Chart Category",
                ["Basic Charts", "Statistical Charts", "Advanced Analytics", "Custom Queries"],
                help="Select the category of charts you want to create",
                key="chart_category_select"
            )
        
        with col2:
            if chart_category == "Basic Charts":
                chart_types = ["Bar Chart", "Line Chart", "Pie Chart", "Area Chart"]
            elif chart_category == "Statistical Charts":
                chart_types = ["Histogram", "Box Plot", "Violin Plot", "Scatter Plot"]
            elif chart_category == "Advanced Analytics":
                chart_types = ["Heatmap", "Correlation Matrix", "Time Series", "Multi-Axis Chart"]
            else:  # Custom Queries
                chart_types = ["Custom SQL Chart"]
            
            chart_type = st.selectbox(
                "Chart Type",
                chart_types,
                help="Select the specific chart type",
                key="charts_chart_type_select"
            )
        
        # Custom query section
        if chart_category == "Custom Queries":
            st.subheader("🔍 Custom SQL Query")
            custom_query = st.text_area(
                "Enter your SQL query:",
                value="SELECT Product_Category, SUM(Net_Sales_Amount) as TotalSales FROM SalesFact GROUP BY Product_Category ORDER BY TotalSales DESC LIMIT 10",
                height=100,
                help="Write a custom SQL query to get data for visualization"
            )
            
            if st.button("🚀 Execute Query & Create Chart", type="primary", key="execute_custom_query"):
                with st.spinner("Executing query and creating chart..."):
                    custom_data = fetch_sales_facts(custom_query)
                    if custom_data:
                        st.success(f"✅ Query executed successfully! Retrieved {len(custom_data)} records.")
                        
                        # Auto-create chart based on data
                        df = pd.DataFrame(custom_data)
                        if len(df.columns) >= 2:
                            # Try to create a bar chart by default
                            fig = create_chart(custom_data, "Bar Chart", title="Custom Query Results")
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Show data
                        st.subheader("📊 Query Results")
                        st.dataframe(custom_data)
                    else:
                        st.error("❌ Query failed or returned no data.")
        
        else:
            # Predefined chart templates
            st.subheader("📋 Chart Configuration")
            
            # Get available columns
            df = pd.DataFrame(viz_data)
            columns = list(df.columns)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                x_column = st.selectbox(
                    "X-Axis Column",
                    columns,
                    help="Select column for X-axis",
                    key="charts_x_column_select"
                )
            
            with col2:
                y_column = st.selectbox(
                    "Y-Axis Column",
                    columns,
                    help="Select column for Y-axis",
                    key="charts_y_column_select"
                )
            
            with col3:
                color_column = st.selectbox(
                    "Color Column (Optional)",
                    ["None"] + columns,
                    help="Select column for color coding",
                    key="charts_color_column_select"
                )
            
            # Chart title and options
            chart_title = st.text_input(
                "Chart Title",
                value=f"{chart_type} - Sales Analysis",
                help="Enter a title for the chart"
            )
            
            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                col1, col2 = st.columns(2)
                
                with col1:
                    chart_height = st.slider(
                        "Chart Height",
                        min_value=300,
                        max_value=800,
                        value=500,
                        help="Set the height of the chart"
                    )
                    
                    show_legend = st.checkbox(
                        "Show Legend",
                        value=True,
                        help="Display chart legend"
                    )
                
                with col2:
                    chart_theme = st.selectbox(
                        "Chart Theme",
                        ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn"],
                        help="Select chart theme",
                        key="chart_theme_select"
                    )
                    
                    enable_animations = st.checkbox(
                        "Enable Animations",
                        value=True,
                        help="Enable chart animations"
                    )
            
            # Create chart
            if st.button("🔄 Generate Chart", type="primary", key="charts_generate_chart"):
                with st.spinner("Creating chart..."):
                    color_col = None if color_column == "None" else color_column
                    
                    fig = create_chart(
                        viz_data,
                        chart_type,
                        x_column,
                        y_column,
                        color_col,
                        chart_title
                    )
                    
                    if fig:
                        # Apply advanced options
                        fig.update_layout(
                            height=chart_height,
                            showlegend=show_legend,
                            template=chart_theme
                        )
                        
                        if not enable_animations:
                            fig.update_layout(
                                transition_duration=0
                            )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Download options
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # Download as HTML
                            html_string = fig.to_html(include_plotlyjs='cdn')
                            st.download_button(
                                label="📥 HTML Chart",
                                data=html_string,
                                file_name=f"{chart_type.lower().replace(' ', '_')}.html",
                                mime="text/html"
                            )
                        
                        with col2:
                            # Download as PNG
                            img_bytes = fig.to_image(format="png")
                            st.download_button(
                                label="📥 PNG Image",
                                data=img_bytes,
                                file_name=f"{chart_type.lower().replace(' ', '_')}.png",
                                mime="image/png"
                            )
                        
                        with col3:
                            # Download data as CSV
                            csv_data = pd.DataFrame(viz_data).to_csv(index=False)
                            st.download_button(
                                label="📥 CSV Data",
                                data=csv_data,
                                file_name=f"{chart_type.lower().replace(' ', '_')}_data.csv",
                                mime="text/csv"
                            )
                    else:
                        st.error("❌ Could not create chart with the selected parameters. Try different column selections.")
            
            # Quick chart templates
            st.subheader("🚀 Quick Chart Templates")
            
            template_col1, template_col2, template_col3, template_col4 = st.columns(4)
            
            with template_col1:
                if st.button("📊 Top Products by Sales", use_container_width=True, key="template_top_products"):
                    template_query = """
                    SELECT Product_Name, SUM(Net_Sales_Amount) as TotalSales 
                    FROM SalesFact 
                    GROUP BY Product_Name 
                    ORDER BY TotalSales DESC 
                    LIMIT 10
                    """
                    template_data = fetch_sales_facts(template_query)
                    if template_data:
                        fig = create_chart(template_data, "Bar Chart", "Product_Name", "TotalSales", title="Top 10 Products by Sales")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
            
            with template_col2:
                if st.button("📈 Monthly Sales Trend", use_container_width=True, key="template_monthly_trend"):
                    template_query = """
                    SELECT `Sales Month`, SUM(Net_Sales_Amount) as MonthlySales 
                    FROM SalesFact 
                    GROUP BY `Sales Month` 
                    ORDER BY FIELD(`Sales Month`, 'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec')
                    """
                    template_data = fetch_sales_facts(template_query)
                    if template_data:
                        fig = create_chart(template_data, "Line Chart", "Sales Month", "MonthlySales", title="Monthly Sales Trend")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
            
            with template_col3:
                if st.button("🥧 Sales by Region", use_container_width=True, key="template_sales_by_region"):
                    template_query = """
                    SELECT Store_Region, SUM(Net_Sales_Amount) as RegionalSales 
                    FROM SalesFact 
                    GROUP BY Store_Region 
                    ORDER BY RegionalSales DESC
                    """
                    template_data = fetch_sales_facts(template_query)
                    if template_data:
                        fig = create_chart(template_data, "Pie Chart", "Store_Region", "RegionalSales", title="Sales Distribution by Region")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
            
            with template_col4:
                if st.button("📊 Sales by Category", use_container_width=True, key="template_sales_by_category"):
                    template_query = """
                    SELECT Product_Category, SUM(Net_Sales_Amount) as CategorySales 
                    FROM SalesFact 
                    GROUP BY Product_Category 
                    ORDER BY CategorySales DESC
                    """
                    template_data = fetch_sales_facts(template_query)
                    if template_data:
                        fig = create_chart(template_data, "Bar Chart", "Product_Category", "CategorySales", title="Sales by Product Category")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)

            # Additional quick chart templates
            st.markdown("---")
            st.subheader("🎯 More Chart Templates")
            
            template_col5, template_col6, template_col7, template_col8 = st.columns(4)
            
            with template_col5:
                if st.button("📈 Quarterly Comparison", use_container_width=True, key="template_quarterly_comparison"):
                    template_query = """
                    SELECT `Sales Quarter`, `Sales Year`, SUM(Net_Sales_Amount) as QuarterlySales 
                    FROM SalesFact 
                    GROUP BY `Sales Quarter`, `Sales Year` 
                    ORDER BY `Sales Year`, `Sales Quarter`
                    """
                    template_data = fetch_sales_facts(template_query)
                    if template_data:
                        fig = create_chart(template_data, "Line Chart", "Sales Quarter", "QuarterlySales", "Sales Year", title="Quarterly Sales Comparison")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
            
            with template_col6:
                if st.button("📊 Payment Methods", use_container_width=True, key="template_payment_methods"):
                    template_query = """
                    SELECT Payment_Method, COUNT(*) as TransactionCount, SUM(Net_Sales_Amount) as TotalAmount 
                    FROM SalesFact 
                    GROUP BY Payment_Method 
                    ORDER BY TotalAmount DESC
                    """
                    template_data = fetch_sales_facts(template_query)
                    if template_data:
                        fig = create_chart(template_data, "Bar Chart", "Payment_Method", "TotalAmount", title="Sales by Payment Method")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
            
            with template_col7:
                if st.button("📈 Salesperson Performance", use_container_width=True, key="template_salesperson_performance"):
                    template_query = """
                    SELECT Salesperson_Name, SUM(Net_Sales_Amount) as TotalSales, COUNT(*) as TransactionCount 
                    FROM SalesFact 
                    GROUP BY Salesperson_Name 
                    ORDER BY TotalSales DESC 
                    LIMIT 10
                    """
                    template_data = fetch_sales_facts(template_query)
                    if template_data:
                        fig = create_chart(template_data, "Bar Chart", "Salesperson_Name", "TotalSales", title="Top 10 Salespeople by Revenue")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
            
            with template_col8:
                if st.button("📊 Customer Cities", use_container_width=True, key="template_customer_cities"):
                    template_query = """
                    SELECT Customer_City, SUM(Net_Sales_Amount) as CitySales 
                    FROM SalesFact 
                    GROUP BY Customer_City 
                    ORDER BY CitySales DESC 
                    LIMIT 10
                    """
                    template_data = fetch_sales_facts(template_query)
                    if template_data:
                        fig = create_chart(template_data, "Bar Chart", "Customer_City", "CitySales", title="Top 10 Cities by Sales")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)