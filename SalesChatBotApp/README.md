# SalesChatBotApp

A Streamlit-based chat application that answers questions about your sales data using natural language processing and a local Ollama LLM.

## Features
- Ask any question about your sales data in natural language
- Automatically generates SQL queries for your MySQL `SalesFact` table
- Uses Ollama (local LLM) to interpret questions and generate answers
- User-friendly chat interface

## Setup

1. **Clone or copy this folder to your Desktop**
2. **Install dependencies** (in a virtual environment):
   ```bash
   cd ~/Desktop/SalesChatBotApp
   source venv/bin/activate
   pip install -r requirements.txt  # If you add more dependencies
   ```
3. **Configure your database and Ollama settings**:
   - Edit `.streamlit/secrets.toml` with your MySQL and Ollama details.
   - Ensure your MySQL server is running and has a `SalesFact` table in the specified database.
   - Ensure Ollama is running locally (default: `http://localhost:11434`).

## Running the App

```bash
streamlit run app.py
```

## Notes
- The app will use the LLM to generate SQL queries from your questions. Review the generated SQL for safety if needed.
- You can change the LLM model or endpoint in `.streamlit/secrets.toml`.

## Requirements
- Python 3.8+
- MySQL server with a `SalesFact` table
- Ollama installed and running locally

---

**Enjoy chatting with your sales data!** 