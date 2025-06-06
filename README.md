# CMT Company Analyzer

This Streamlit app analyzes a company's profile based on website or description, and finds relevant matches from a database using vector similarity and filtering.

## Setup

1. Add your OpenAI API key to `.streamlit/secrets.toml` like so:

```
[openai]
api_key = "sk-..."
```

2. Place your Excel database in `app_data/Database.xlsx`
3. Run the app:

```
streamlit run app.py
```
