# StockSaathi: AI stock assistant for small retailers

AI Application End Term Project, use case 11 (Inventory / stock query assistant, Chatbot).

Ask about stock, reorders, expiry and sales in Hindi, English or Hinglish, typed or spoken.
Google Gemini understands the question and calls Python tools; every number comes from
`inventory_tools.py`, never from the AI.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit app: chat (Gemini + voice) and a no-AI dashboard |
| `inventory_tools.py` | Product matching, stock maths, reorder, expiry, orders, offline fallback |
| `StockSaathi_Data_and_Tool_Tests.ipynb` | Generates the sample data and tests every tool |
| `data/` | `skus.csv`, `sales_daily.csv`, `distributors.csv` |
| `requirements.txt`, `.streamlit/config.toml` | Dependencies and theme |

## Deploy (free, no software install needed)
1. Get a Gemini API key: https://aistudio.google.com/apikey (sign in, Create API key, copy it).
2. On github.com create a new **public** repository, e.g. `stocksaathi`.
3. Click **Add file > Upload files** and drag in everything from this folder, including the
   `data` and `.streamlit` folders. (If the `.streamlit` folder is hidden on your computer,
   create `.streamlit/config.toml` in GitHub with **Add file > Create new file** and paste its contents.)
   Commit.
4. Go to https://share.streamlit.io, sign in with GitHub, click **Create app > Deploy a public app from GitHub**.
   Choose the repo, branch `main`, main file `app.py`. Under **Advanced settings > Secrets** paste:
   ```
   GEMINI_API_KEY = "your-key-here"
   ```
   Pick a custom URL such as `stocksaathi-aakash`, then **Deploy**.
5. Never put the API key in any file in the repository.

## Run locally (optional)
```
pip install -r requirements.txt
mkdir -p .streamlit && echo 'GEMINI_API_KEY = "your-key"' > .streamlit/secrets.toml
streamlit run app.py
```
Without a key the app still runs: chat uses the built-in lookup and the dashboard works.
