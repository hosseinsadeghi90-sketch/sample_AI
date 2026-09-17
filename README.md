# AI RAG Web App

وب‌اپ پرسش از فایل TXT با Streamlit، Semantic Search و OpenRouter.

## اجرای محلی

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

API Key را در `.streamlit/secrets.toml` قرار دهید:

```toml
OPENROUTER_API_KEY = "YOUR_KEY"
```

فایل secrets را هرگز به GitHub ارسال نکنید.
