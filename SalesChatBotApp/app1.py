import streamlit as st
import mysql.connector
import requests
import os
import re

# --- COLUMN NAMES (single line for reference) ---
COLUMN_NAMES = "Sale_Date, Customer_Name, Customer_Address, Customer_City, Customer_State, Product_Name, Product_Category, Product_Specification, Unit_of_Measure, Quantity_Sold, Unit_Price, Total_Sales_Amount, Discount_Amount, Tax_Amount, Net_Sales_Amount, Salesperson_Name, Salesperson_Contact, Store_Name, Store_Region, Payment_Method, Transaction_Status, Currency_Code, Sale_Month_Year, Sale_Day, Sale_Quarter, Sales Month, Sales Year, Sales Quarter"

# --- CONFIG ---
DB_HOST = st.secrets.get("DB_HOST", "localhost")
DB_USER = st.secrets.get("DB_USER", "root")
DB_PASSWORD = st.secrets.get("DB_PASSWORD", "")
DB_NAME = st.secrets.get("DB_NAME", "SalesChatBot")
OLLAMA_URL = st.secrets.get("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = st.secrets.get("OLLAMA_MODEL", "llama3")

# --- SCHEMA CONTEXT ---
# SALESFACT_SCHEMA = (
#     "The SalesFact table contains detailed sales transaction data with the following columns:\n\n"

#     # --- DATE & TIME ---
#     "📅 DATE COLUMNS:\n"
#     "• Sale_Date: Use for daily-level filtering.\n"
#     "• Sale_Day, Sales Month, Sales Year, Sales Quarter: Always use `Sales Year` (int) and `Sales Quarter` (e.g., 'Q1') for date filtering.\n"
#     "  ✅ WHERE `Sales Year` = 2024 AND `Sales Quarter` = 'Q2'.\n"
#     "• DO NOT use Sale_Month_Year, Sale_Quarter, or EXTRACT().\n\n"

#     # --- CUSTOMER INFO ---
#     "👤 CUSTOMER COLUMNS:\n"
#     "• Customer_Name, Customer_City, Customer_State: Use for segmentation and grouping.\n\n"

#     # --- PRODUCT INFO ---
#     "📦 PRODUCT COLUMNS:\n"
#     "• Product_Name, Product_Category, Product_Specification: Group and filter by these fields.\n\n"

#     # --- SALES METRICS ---
#     "💰 SALES & FINANCIAL COLUMNS:\n"
#     "• Quantity_Sold, Unit_Price, Total_Sales_Amount, Discount_Amount, Tax_Amount, Net_Sales_Amount.\n"
#     "  ➤ Use SUM(Net_Sales_Amount) for true revenue.\n\n"

#     # --- OTHER INFO ---
#     "🏬 Store_Name, Store_Region; 👨 Salesperson_Name; 💳 Payment_Method; 📦 Transaction_Status.\n\n"

#     # === EXAMPLE QUERIES ===\n"

#     "🔹 TOP-N ANALYSIS:\n"
#     "• Top 5 products by revenue in 2024:\n"
#     "  SELECT Product_Name, SUM(Net_Sales_Amount) AS Revenue\n"
#     "  FROM SalesFact\n"
#     "  WHERE `Sales Year` = 2024\n"
#     "  GROUP BY Product_Name\n"
#     "  ORDER BY Revenue DESC\n"
#     "  LIMIT 5;\n\n"

#     "• Top 3 cities by quantity sold in Q1 2025:\n"
#     "  SELECT Customer_City, SUM(Quantity_Sold) AS Total_Units\n"
#     "  FROM SalesFact\n"
#     "  WHERE `Sales Year` = 2025 AND `Sales Quarter` = 'Q1'\n"
#     "  GROUP BY Customer_City\n"
#     "  ORDER BY Total_Units DESC\n"
#     "  LIMIT 3;\n\n"

#     "🔹 TIME-SERIES & TRENDS:\n"
#     "• Monthly sales trend for 2024:\n"
#     "  SELECT `Sales Month`, SUM(Net_Sales_Amount) AS Monthly_Sales\n"
#     "  FROM SalesFact\n"
#     "  WHERE `Sales Year` = 2024\n"
#     "  GROUP BY `Sales Month`\n"
#     "  ORDER BY FIELD(`Sales Month`, 'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec');\n\n"

#     "• Quarterly comparison between 2023 and 2024:\n"
#     "  SELECT `Sales Quarter`,\n"
#     "         SUM(CASE WHEN `Sales Year` = 2023 THEN Net_Sales_Amount ELSE 0 END) AS Revenue_2023,\n"
#     "         SUM(CASE WHEN `Sales Year` = 2024 THEN Net_Sales_Amount ELSE 0 END) AS Revenue_2024\n"
#     "  FROM SalesFact\n"
#     "  WHERE `Sales Year` IN (2023, 2024)\n"
#     "  GROUP BY `Sales Quarter`\n"
#     "  ORDER BY `Sales Quarter`;\n\n"

#     "🔹 COMPARISON & BENCHMARKING:\n"
#     "• Compare sales by payment method:\n"
#     "  SELECT Payment_Method, SUM(Net_Sales_Amount) AS Revenue\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Payment_Method;\n\n"

#     "• Store-wise revenue comparison for a category:\n"
#     "  SELECT Store_Name, SUM(Net_Sales_Amount) AS Revenue\n"
#     "  FROM SalesFact\n"
#     "  WHERE Product_Category = 'Electronics'\n"
#     "  GROUP BY Store_Name\n"
#     "  ORDER BY Revenue DESC;\n\n"

#     "🔹 SALES REP & PERFORMANCE:\n"
#     "• Top-performing salespeople by revenue:\n"
#     "  SELECT Salesperson_Name, SUM(Net_Sales_Amount) AS Revenue\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Salesperson_Name\n"
#     "  ORDER BY Revenue DESC\n"
#     "  LIMIT 10;\n\n"

#     "• Sales by region and salesperson:\n"
#     "  SELECT Store_Region, Salesperson_Name, SUM(Net_Sales_Amount) AS Total_Sales\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Store_Region, Salesperson_Name\n"
#     "  ORDER BY Store_Region, Total_Sales DESC;\n\n"

#     "🔹 PRODUCT INSIGHTS:\n"
#     "• Best-selling product by state:\n"
#     "  SELECT Customer_State, Product_Name, SUM(Net_Sales_Amount) AS Revenue\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Customer_State, Product_Name\n"
#     "  ORDER BY Customer_State, Revenue DESC;\n\n"

#     "• Discount impact by category:\n"
#     "  SELECT Product_Category, SUM(Discount_Amount) AS Total_Discounts\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Product_Category\n"
#     "  ORDER BY Total_Discounts DESC;\n\n"

#     "🔹 ADVANCED PATTERNS:\n"
#     "• Customers with repeat purchases (same customer appearing multiple times):\n"
#     "  SELECT Customer_Name, COUNT(*) AS Transactions\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Customer_Name\n"
#     "  HAVING Transactions > 1;\n\n"

#     "• Sales drop detection (compare two quarters):\n"
#     "  SELECT Product_Name,\n"
#     "         SUM(CASE WHEN `Sales Year` = 2024 AND `Sales Quarter` = 'Q1' THEN Net_Sales_Amount ELSE 0 END) AS Q1_Sales,\n"
#     "         SUM(CASE WHEN `Sales Year` = 2024 AND `Sales Quarter` = 'Q2' THEN Net_Sales_Amount ELSE 0 END) AS Q2_Sales\n"
#     "  FROM SalesFact\n"
#     "  WHERE `Sales Year` = 2024\n"
#     "  GROUP BY Product_Name\n"
#     "  HAVING Q2_Sales < Q1_Sales;\n\n"

#     "• Refund or failed transactions (if included in Transaction_Status):\n"
#     "  SELECT Transaction_Status, COUNT(*) AS Count, SUM(Net_Sales_Amount) AS Revenue\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Transaction_Status;\n\n"

#     "🔹 OUTLIER DETECTION:\n"
#     "• Products with unusually high unit price:\n"
#     "  SELECT Product_Name, MAX(Unit_Price) AS Highest_Price\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Product_Name\n"
#     "  ORDER BY Highest_Price DESC\n"
#     "  LIMIT 5;\n\n"

#     "• Customers who bought the most units in a single transaction:\n"
#     "  SELECT Customer_Name, Quantity_Sold, Sale_Date\n"
#     "  FROM SalesFact\n"
#     "  ORDER BY Quantity_Sold DESC\n"
#     "  LIMIT 5;\n\n"

#     "🔹 MULTIPLE FILTERS:\n"
#     "• Net sales for electronics in Dhaka in Q3 2024:\n"
#     "  SELECT SUM(Net_Sales_Amount) AS Total_Revenue\n"
#     "  FROM SalesFact\n"
#     "  WHERE Customer_City = 'Dhaka'\n"
#     "    AND Product_Category = 'Electronics'\n"
#     "    AND `Sales Year` = 2024 AND `Sales Quarter` = 'Q3';\n\n"

#     "⚠️ REMINDERS:\n"
#     "• Always filter by `Sales Year` and `Sales Quarter`.\n"
#     "• Always use Net_Sales_Amount for financial reporting.\n"
#     "• Do not use Sale_Quarter, Sale_Month_Year, or EXTRACT().\n"
#     "• Format output numbers as currency (৳) and add line breaks in answers for clarity.\n"

#     "\nIMPORTANT:\n"
#     "- Only use the `SalesFact` table. Do NOT use or join any other tables such as `products`, `states`, etc.\n"
#     "- Always use the exact column names as listed above, including spaces and capitalization.\n"
#     "- Always use backticks around column names with spaces, e.g., `Sales Year`.\n"
#     "- If you use window functions (like DENSE_RANK or RANK), use a subquery or CTE.\n"
#     "- Do NOT use window functions directly in the SELECT list with GROUP BY; use a subquery.\n"
#     "- If MySQL version is less than 8.0, avoid window functions entirely.\n"
#     "- Example of what NOT to do: SELECT ..., DENSE_RANK() OVER (...) ... FROM ... GROUP BY ...\n"
#     "- Do NOT use snake_case or pluralized table names.\n"
#     "- Example of what NOT to do: `SELECT ... FROM sales_fact` or `JOIN products`.\n"
# )
# SALESFACT_SCHEMA += (

#     "🔹 CUSTOMER SEGMENTATION & BEHAVIOR:\n"
#     "• Top customers by lifetime sales:\n"
#     "  SELECT Customer_Name, SUM(Net_Sales_Amount) AS Lifetime_Value\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Customer_Name\n"
#     "  ORDER BY Lifetime_Value DESC\n"
#     "  LIMIT 10;\n\n"

#     "• Average order value by customer city:\n"
#     "  SELECT Customer_City, AVG(Net_Sales_Amount) AS Avg_Order_Value\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Customer_City\n"
#     "  ORDER BY Avg_Order_Value DESC;\n\n"

#     "• Unique customer count by year:\n"
#     "  SELECT `Sales Year`, COUNT(DISTINCT Customer_Name) AS Unique_Customers\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY `Sales Year`\n"
#     "  ORDER BY `Sales Year`;\n\n"

#     "🔹 PRODUCT-LEVEL PROFITABILITY & MIX:\n"
#     "• Product category revenue contribution:\n"
#     "  SELECT Product_Category,\n"
#     "         SUM(Net_Sales_Amount) AS Category_Revenue,\n"
#     "         ROUND(SUM(Net_Sales_Amount) * 100 / (SELECT SUM(Net_Sales_Amount) FROM SalesFact), 2) AS Contribution_Percent\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Product_Category\n"
#     "  ORDER BY Category_Revenue DESC;\n\n"

#     "• Average discount by product:\n"
#     "  SELECT Product_Name, AVG(Discount_Amount) AS Avg_Discount\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Product_Name\n"
#     "  ORDER BY Avg_Discount DESC;\n\n"

#     "🔹 PRICE & UNIT ANALYSIS:\n"
#     "• Average price per unit (normalized):\n"
#     "  SELECT Product_Name, SUM(Total_Sales_Amount) / SUM(Quantity_Sold) AS Avg_Unit_Price\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Product_Name\n"
#     "  ORDER BY Avg_Unit_Price DESC;\n\n"

#     "• Total quantity sold per unit of measure:\n"
#     "  SELECT Unit_of_Measure, SUM(Quantity_Sold) AS Total_Qty\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Unit_of_Measure\n"
#     "  ORDER BY Total_Qty DESC;\n\n"

#     "🔹 GEOGRAPHICAL ANALYSIS:\n"
#     "• Revenue by region and year:\n"
#     "  SELECT Store_Region, `Sales Year`, SUM(Net_Sales_Amount) AS Region_Revenue\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Store_Region, `Sales Year`\n"
#     "  ORDER BY Store_Region, `Sales Year`;\n\n"

#     "• States with above-average sales:\n"
#     "  SELECT Customer_State, SUM(Net_Sales_Amount) AS Total_Revenue\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Customer_State\n"
#     "  HAVING Total_Revenue > (SELECT AVG(Net_Sales_Amount) FROM SalesFact);\n\n"

#     "🔹 SALES CHANNEL EVALUATION:\n"
#     "• Payment method usage share:\n"
#     "  SELECT Payment_Method,\n"
#     "         COUNT(*) AS Transactions,\n"
#     "         ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM SalesFact), 2) AS Usage_Percent\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Payment_Method\n"
#     "  ORDER BY Transactions DESC;\n\n"

#     "• Successful vs. failed transaction count:\n"
#     "  SELECT Transaction_Status, COUNT(*) AS Count\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Transaction_Status;\n\n"

#     "🔹 TAX & COMPLIANCE INSIGHT:\n"
#     "• Total tax collected per year:\n"
#     "  SELECT `Sales Year`, SUM(Tax_Amount) AS Total_Tax\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY `Sales Year`\n"
#     "  ORDER BY `Sales Year`;\n\n"

#     "• Product category-wise tax burden:\n"
#     "  SELECT Product_Category, SUM(Tax_Amount) AS Tax_Collected\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Product_Category\n"
#     "  ORDER BY Tax_Collected DESC;\n\n"

#     "🔹 CROSS-TABS & MULTIDIMENSIONAL INSIGHT:\n"
#     "• Sales by Region and Quarter:\n"
#     "  SELECT Store_Region, `Sales Quarter`, SUM(Net_Sales_Amount) AS Total_Sales\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Store_Region, `Sales Quarter`\n"
#     "  ORDER BY Store_Region, `Sales Quarter`;\n\n"

#     "• Units sold by Category and Year:\n"
#     "  SELECT Product_Category, `Sales Year`, SUM(Quantity_Sold) AS Total_Units\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Product_Category, `Sales Year`\n"
#     "  ORDER BY Product_Category, `Sales Year`;\n\n"

#     "🔹 EDGE CASES & AUDIT:\n"
#     "• Negative or zero quantity transactions:\n"
#     "  SELECT * FROM SalesFact WHERE Quantity_Sold <= 0;\n\n"

#     "• Orders with zero revenue:\n"
#     "  SELECT * FROM SalesFact WHERE Net_Sales_Amount = 0;\n\n"

#     "🔹 RANKING:\n"
#     "• Rank stores by yearly revenue (using MySQL variable):\n"
#     "  SELECT Store_Name, `Sales Year`, SUM(Net_Sales_Amount) AS Revenue,\n"
#     "         RANK() OVER (PARTITION BY `Sales Year` ORDER BY SUM(Net_Sales_Amount) DESC) AS Rank\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Store_Name, `Sales Year`;\n\n"

#     "🔹 PERCENTAGE CHANGES:\n"
#     "• Year-over-year sales growth by product:\n"
#     "  SELECT Product_Name,\n"
#     "         SUM(CASE WHEN `Sales Year` = 2024 THEN Net_Sales_Amount ELSE 0 END) AS Sales_2024,\n"
#     "         SUM(CASE WHEN `Sales Year` = 2023 THEN Net_Sales_Amount ELSE 0 END) AS Sales_2023,\n"
#     "         ROUND((SUM(CASE WHEN `Sales Year` = 2024 THEN Net_Sales_Amount ELSE 0 END) -\n"
#     "               SUM(CASE WHEN `Sales Year` = 2023 THEN Net_Sales_Amount ELSE 0 END)) * 100.0 /\n"
#     "               NULLIF(SUM(CASE WHEN `Sales Year` = 2023 THEN Net_Sales_Amount ELSE 0 END), 0), 2) AS YoY_Growth_Percent\n"
#     "  FROM SalesFact\n"
#     "  WHERE `Sales Year` IN (2023, 2024)\n"
#     "  GROUP BY Product_Name\n"
#     "  ORDER BY YoY_Growth_Percent DESC;\n\n"

#     "✅ REMEMBER:\n"
#     "Use these query templates to generate precise answers.\n"
#     "Queries should be safe, accurate, and always reference actual column names exactly.\n"
# )
# SALESFACT_SCHEMA += (

#     "🔹 SALES TREND ANALYSIS:\n"
#     "• Monthly sales trend for a given year:\n"
#     "  SELECT `Sales Month`, SUM(Net_Sales_Amount) AS Monthly_Sales\n"
#     "  FROM SalesFact\n"
#     "  WHERE `Sales Year` = 2024\n"
#     "  GROUP BY `Sales Month`\n"
#     "  ORDER BY FIELD(`Sales Month`, 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December');\n\n"

#     "• Sales growth per quarter over years:\n"
#     "  SELECT `Sales Quarter`, `Sales Year`, SUM(Net_Sales_Amount) AS Revenue\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY `Sales Year`, `Sales Quarter`\n"
#     "  ORDER BY `Sales Year`, `Sales Quarter`;\n\n"

#     "🔹 SALES PERSONNEL PERFORMANCE:\n"
#     "• Top 5 performing salespeople by net revenue:\n"
#     "  SELECT Salesperson_Name, SUM(Net_Sales_Amount) AS Total_Revenue\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Salesperson_Name\n"
#     "  ORDER BY Total_Revenue DESC\n"
#     "  LIMIT 5;\n\n"

#     "• Salesperson performance by quarter:\n"
#     "  SELECT Salesperson_Name, `Sales Quarter`, SUM(Net_Sales_Amount) AS Revenue\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Salesperson_Name, `Sales Quarter`\n"
#     "  ORDER BY Salesperson_Name, `Sales Quarter`;\n\n"

#     "🔹 STORE PERFORMANCE:\n"
#     "• Best-performing store per region:\n"
#     "  SELECT Store_Region, Store_Name, SUM(Net_Sales_Amount) AS Regional_Sales\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Store_Region, Store_Name\n"
#     "  ORDER BY Store_Region, Regional_Sales DESC;\n\n"

#     "• Store-wise average transaction size:\n"
#     "  SELECT Store_Name, AVG(Net_Sales_Amount) AS Avg_Transaction\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Store_Name\n"
#     "  ORDER BY Avg_Transaction DESC;\n\n"

#     "🔹 LOYALTY & REPEAT PURCHASE:\n"
#     "• Repeat customer identification:\n"
#     "  SELECT Customer_Name, COUNT(*) AS Order_Count\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Customer_Name\n"
#     "  HAVING Order_Count > 1\n"
#     "  ORDER BY Order_Count DESC;\n\n"

#     "• Customer retention by year:\n"
#     "  SELECT a.Customer_Name\n"
#     "  FROM (SELECT DISTINCT Customer_Name FROM SalesFact WHERE `Sales Year` = 2023) a\n"
#     "  INNER JOIN (SELECT DISTINCT Customer_Name FROM SalesFact WHERE `Sales Year` = 2024) b\n"
#     "  ON a.Customer_Name = b.Customer_Name;\n\n"

#     "🔹 TRANSACTIONAL DISTRIBUTION:\n"
#     "• Distribution of sales amounts (buckets):\n"
#     "  SELECT\n"
#     "    CASE\n"
#     "      WHEN Net_Sales_Amount < 100 THEN '<$100'\n"
#     "      WHEN Net_Sales_Amount BETWEEN 100 AND 499 THEN '$100-$499'\n"
#     "      WHEN Net_Sales_Amount BETWEEN 500 AND 999 THEN '$500-$999'\n"
#     "      ELSE '$1000+'\n"
#     "    END AS Sales_Bracket,\n"
#     "    COUNT(*) AS Transaction_Count\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Sales_Bracket;\n\n"

#     "🔹 CATEGORY DEPENDENCY:\n"
#     "• Category performance by region:\n"
#     "  SELECT Store_Region, Product_Category, SUM(Net_Sales_Amount) AS Sales\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Store_Region, Product_Category\n"
#     "  ORDER BY Store_Region, Sales DESC;\n\n"

#     "🔹 HIGH/LOW OUTLIER ANALYSIS:\n"
#     "• Transactions with unusually high discount:\n"
#     "  SELECT * FROM SalesFact\n"
#     "  WHERE Discount_Amount > 100;\n\n"

#     "• High-value single-item orders:\n"
#     "  SELECT * FROM SalesFact\n"
#     "  WHERE Quantity_Sold = 1 AND Net_Sales_Amount > 1000;\n\n"

#     "🔹 TAXATION vs REVENUE:\n"
#     "• Effective tax rate per product:\n"
#     "  SELECT Product_Name, SUM(Tax_Amount)/SUM(Net_Sales_Amount) AS Effective_Tax_Rate\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Product_Name\n"
#     "  ORDER BY Effective_Tax_Rate DESC;\n\n"

#     "🔹 SALES DYNAMICS OVER TIME:\n"
#     "• Sales vs discounts over time:\n"
#     "  SELECT `Sales Year`, `Sales Quarter`,\n"
#     "         SUM(Net_Sales_Amount) AS Revenue,\n"
#     "         SUM(Discount_Amount) AS Total_Discounts\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY `Sales Year`, `Sales Quarter`\n"
#     "  ORDER BY `Sales Year`, `Sales Quarter`;\n\n"

#     "🔹 MOST PROFITABLE PAIRINGS (product + store or product + region):\n"
#     "• Product revenue per store:\n"
#     "  SELECT Store_Name, Product_Name, SUM(Net_Sales_Amount) AS Total_Sales\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Store_Name, Product_Name\n"
#     "  ORDER BY Total_Sales DESC;\n\n"

#     "🔹 CURRENCY IMPLICATIONS:\n"
#     "• Sales breakdown by currency code:\n"
#     "  SELECT Currency_Code, SUM(Net_Sales_Amount) AS Total_Sales\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Currency_Code;\n\n"

#     "🔹 PEAK ANALYSIS:\n"
#     "• Best sales day:\n"
#     "  SELECT Sale_Date, SUM(Net_Sales_Amount) AS Day_Revenue\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Sale_Date\n"
#     "  ORDER BY Day_Revenue DESC\n"
#     "  LIMIT 1;\n\n"

#     "• Peak sales hours (if hour info added in future):\n"
#     "  Add a 'Sale_Hour' column and group by it.\n\n"

#     "🛑 ALWAYS use proper casing, exact column names (with backticks if needed), and avoid EXTRACT or YEAR functions.\n"
#     "Use 'Sales Year' and 'Sales Quarter' (with space) when filtering by year/quarter.\n"
# )
# SALESFACT_SCHEMA += (

#     "🔹 SEASONALITY & TIME PATTERNS:\n"
#     "• Average monthly sales across years (detect seasonality):\n"
#     "  SELECT `Sales Month`, AVG(Net_Sales_Amount) AS Avg_Sales\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY `Sales Month`\n"
#     "  ORDER BY FIELD(`Sales Month`, 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December');\n\n"

#     "• Which quarter historically performs best:\n"
#     "  SELECT `Sales Quarter`, AVG(Net_Sales_Amount) AS Avg_Quarterly_Sales\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY `Sales Quarter`\n"
#     "  ORDER BY Avg_Quarterly_Sales DESC;\n\n"

#     "🔹 CUSTOMER-BASED SEGMENTATION:\n"
#     "• Top 10 customers by lifetime value:\n"
#     "  SELECT Customer_Name, SUM(Net_Sales_Amount) AS Lifetime_Value\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Customer_Name\n"
#     "  ORDER BY Lifetime_Value DESC\n"
#     "  LIMIT 10;\n\n"

#     "• Most frequent buyers:\n"
#     "  SELECT Customer_Name, COUNT(*) AS Purchases\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Customer_Name\n"
#     "  ORDER BY Purchases DESC\n"
#     "  LIMIT 10;\n\n"

#     "• Inactive customers (e.g., no purchase this year):\n"
#     "  SELECT DISTINCT Customer_Name\n"
#     "  FROM SalesFact\n"
#     "  WHERE `Sales Year` < 2025\n"
#     "  AND Customer_Name NOT IN (\n"
#     "    SELECT DISTINCT Customer_Name FROM SalesFact WHERE `Sales Year` = 2025\n"
#     "  );\n\n"

#     "🔹 GROSS PROFIT ESTIMATES (if cost known/added):\n"
#     "• Estimated profit = Net Sales - Discount - Cost (if Cost_Column is available):\n"
#     "  SELECT Product_Name, SUM(Net_Sales_Amount - Discount_Amount - Estimated_Cost) AS Estimated_Profit\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Product_Name\n"
#     "  ORDER BY Estimated_Profit DESC;\n\n"

#     "🔹 FORECASTING PREPARATION:\n"
#     "• Historical monthly sales (to use in ML forecasting models):\n"
#     "  SELECT `Sales Year`, `Sales Month`, SUM(Net_Sales_Amount) AS Total_Sales\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY `Sales Year`, `Sales Month`\n"
#     "  ORDER BY `Sales Year`, FIELD(`Sales Month`, 'January','February','March','April','May','June','July','August','September','October','November','December');\n\n"

#     "🔹 PRODUCT MIX PERFORMANCE:\n"
#     "• Which product category drives highest revenue:\n"
#     "  SELECT Product_Category, SUM(Net_Sales_Amount) AS Revenue\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Product_Category\n"
#     "  ORDER BY Revenue DESC;\n\n"

#     "• Product sales contribution by store:\n"
#     "  SELECT Store_Name, Product_Category, SUM(Net_Sales_Amount) AS Total_Sales\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Store_Name, Product_Category\n"
#     "  ORDER BY Store_Name, Total_Sales DESC;\n\n"

#     "🔹 PAYMENT METHOD INSIGHTS:\n"
#     "• Most used payment methods overall:\n"
#     "  SELECT Payment_Method, COUNT(*) AS Usage_Count\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Payment_Method\n"
#     "  ORDER BY Usage_Count DESC;\n\n"

#     "• Payment method breakdown per region:\n"
#     "  SELECT Store_Region, Payment_Method, COUNT(*) AS Count\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Store_Region, Payment_Method\n"
#     "  ORDER BY Store_Region, Count DESC;\n\n"

#     "🔹 CROSS-SELL INDICATORS:\n"
#     "• Customers who bought products from multiple categories:\n"
#     "  SELECT Customer_Name\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Customer_Name\n"
#     "  HAVING COUNT(DISTINCT Product_Category) > 1;\n\n"

#     "🔹 SALES RELIABILITY & SUCCESS RATE:\n"
#     "• % of transactions completed successfully:\n"
#     "  SELECT\n"
#     "    ROUND(\n"
#     "      100 * SUM(CASE WHEN Transaction_Status = 'Completed' THEN 1 ELSE 0 END) / COUNT(*), 2\n"
#     "    ) AS Completion_Rate\n"
#     "  FROM SalesFact;\n\n"

#     "🔹 REGIONAL SALES PENETRATION:\n"
#     "• Revenue per customer in each region:\n"
#     "  SELECT Store_Region, AVG(Net_Sales_Amount) AS Avg_Customer_Spend\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY Store_Region;\n\n"

#     "🔹 STORE RANKING PER QUARTER:\n"
#     "• Rank stores quarterly by revenue:\n"
#     "  SELECT `Sales Quarter`, Store_Name, SUM(Net_Sales_Amount) AS Revenue\n"
#     "  FROM SalesFact\n"
#     "  GROUP BY `Sales Quarter`, Store_Name\n"
#     "  ORDER BY `Sales Quarter`, Revenue DESC;\n\n"

#     "💡 TIP: Avoid using functions like YEAR(), QUARTER(), or MONTH(). Use provided fields like `Sales Year`, `Sales Quarter`, and `Sales Month`.\n"
# )


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
    "• `Sales Month`: VARCHAR(255) (e.g., 'Jan', 'Feb'). Use `ORDER BY FIELD(\\`Sales Month\\`, 'Jan',...,'Dec')` for chronological order.\n"
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

def clean_sql(sql):
    # Remove all code block markers and backticks
    sql = re.sub(r"```[a-zA-Z]*\n?", "", sql)
    sql = re.sub(r"```", "", sql)
    sql = sql.strip("` \n")
    # Remove any lines that are explanations (e.g., "Here is the SQL query:")
    lines = sql.splitlines()
    # Only keep lines that look like SQL (start with SELECT, WITH, or are indented SQL)
    sql_lines = []
    started = False
    for line in lines:
        if re.match(r"^(SELECT|WITH)\b", line.strip(), re.IGNORECASE):
            started = True
        if started:
            sql_lines.append(line)
    sql = "\n".join(sql_lines)
    # Remove any trailing HTML or explanations
    sql = re.split(r"</div>|\n\s*--|#|//", sql)[0]
    return sql.strip()


def get_db_connection():
    return mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME
    )

def fetch_sales_facts(query):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(query)
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return results

# --- LLM ---
def ask_ollama(prompt):
    try:
        response = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False})
        response.raise_for_status()
        return response.json().get("response", "No response.")
    except Exception as e:
        return f"Error from Ollama: {e}"

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Conversational GPT", page_icon="��", layout="wide")

col1, col2 = st.columns([8, 2])
with col1:
    st.markdown("<h4 style='margin-bottom:0.65rem;'>Conversational GPT: Business insight, at the speed of thought.🤖</h4>", unsafe_allow_html=True)
with col2:
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Display column names at the top, wrapped for readability
st.markdown("**Available Information:**")

with st.expander("Available Information:"):
    col_names = [col.strip() for col in COLUMN_NAMES.split(",")]
    for i in range(0, len(col_names), 6):
        cols = st.columns(7)
        for j, col in enumerate(col_names[i:i+7]):
            with cols[j]:
                st.markdown(f"<div style='background:#f1f3f4;border-radius:6px;padding:6px 8px;margin:2px 0;text-align:center;border:1px solid #e0e0e0;font-size:0.7em;'>{col}</div>", unsafe_allow_html=True)

# Make the sidebar wider
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
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

if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

if "set_chat_input" not in st.session_state:
    st.session_state.set_chat_input = None

if "clear_input" not in st.session_state:
    st.session_state.clear_input = False

# Clear input safely before widget creation
if st.session_state.clear_input:
    st.session_state.chat_input = ""
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
                    st.session_state.set_chat_input = q
                    st.rerun()

# --- SET CHAT INPUT BEFORE RENDERING WIDGET ---
if st.session_state.set_chat_input:
    st.session_state.chat_input = st.session_state.set_chat_input
    st.session_state.set_chat_input = None

# --- INPUT AREA ---
st.markdown("<h5 style='margin-bottom:0.4rem;'>💬 GPT that helps you see</h5>", unsafe_allow_html=True)
input_col, send_col = st.columns([6, 1])
with input_col:
    user_question = st.text_input(
        "Ask your question:",
        value=st.session_state.chat_input,
        key="chat_input",
        label_visibility="collapsed",
        placeholder="e.g., Show sales by region for 2024"
    )
with send_col:
    ask_clicked = st.button("Send", use_container_width=True)

# --- ON SEND ---
if (ask_clicked or user_question) and user_question.strip():
    user_question = user_question.strip()
    with st.spinner("🤔 Thinking..."):
        sql_prompt = f"You are an expert SQL assistant. {SALESFACT_SCHEMA} Given this question, write a MySQL query to answer it. Only return SQL. Question: {user_question}"
        sql_query = clean_sql(ask_ollama(sql_prompt))

        try:
            data = fetch_sales_facts(sql_query)
            response_prompt = f"Given the question: '{user_question}' and the result: {data}, explain the result in plain business language."
            answer = ask_ollama(response_prompt)
        except Exception as e:
            data = None
            answer = f"Error: {e}\n\nSQL tried:```sql\n{sql_query}\n```"

        st.session_state.chat_history.append({
            "question": user_question,
            "sql": sql_query,
            "data": data,
            "answer": answer
        })
        st.session_state.clear_input = True
        st.rerun()

# --- CHAT HISTORY DISPLAY ---
for chat in reversed(st.session_state.chat_history):
    st.markdown(f"<div style='padding:12px;border-radius:10px;margin-bottom:10px;background:#e0f7fa;text-align:right;'><b>You:</b> {chat['question']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='padding:12px;border-radius:10px;margin-bottom:10px;background:#f1f8e9;text-align:left;'><b>Bot:</b><br>{chat['answer']}</div>", unsafe_allow_html=True)
    if chat["data"]:
        st.dataframe(chat["data"])
    st.caption(f"🧠 SQL used: `{chat['sql']}`")