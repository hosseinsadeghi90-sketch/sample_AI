import hashlib
import pickle
import re
import os
from pathlib import Path
import numpy as np
import requests
import streamlit as st


# =========================================================
# تنظیمات
# =========================================================

FILE_URL = (
    "https://drive.google.com/uc?export=download&id="
    "1lhf-kvfykyWMyLdBTGUgZM-hJJqu0sBP"
)

BASE_URL = "https://openrouter.ai/api/v1"

# مدل رایگان پاسخ‌دهنده
CHAT_MODEL = "openrouter/free"

# مدل رایگان Embedding
EMBEDDING_MODEL = "liquid/lfm-2.5-embedding-350m:free"

# مدل Embedding حداکثر 512 توکن برای هر ورودی دارد.
# برای متن فارسی، مقدار محافظه‌کارانه انتخاب شده است.
CHUNK_SIZE = 100
CHUNK_OVERLAP = 15

# چند chunk مرتبط به مدل Chat فرستاده شود
TOP_K = 5

CACHE_FILE = Path("embedding_cache.pkl")


# =========================================================
# تنظیمات صفحه
# =========================================================

st.set_page_config(
    page_title="پرسش از فایل با هوش مصنوعی",
    page_icon="🤖",
    layout="wide",
)

# =========================================================
# ظاهر و استایل فارسی
# =========================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700&display=swap');

/* -------------------------------------------------------
   تنظیمات عمومی
------------------------------------------------------- */

html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stMain"],
section.main {
    direction: rtl !important;
}

html,
body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    font-family: "Vazirmatn", Tahoma, Arial, sans-serif !important;
}

/* همه متن‌ها فارسی و RTL */
.stApp p,
.stApp span,
.stApp label,
.stApp div,
.stApp input,
.stApp textarea,
.stApp button,
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    font-family: "Vazirmatn", Tahoma, Arial, sans-serif !important;
}

/* -------------------------------------------------------
   هدر
------------------------------------------------------- */

.page-header {
    direction: rtl;
    text-align: center;
    margin-top: 0.5rem;
    margin-bottom: 2rem;
    padding: 0 1rem;
}

.page-header h1 {
    direction: rtl;
    text-align: center !important;
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1.6;
    margin: 0 0 0.8rem 0;
}

.page-header p {
    direction: rtl;
    text-align: center !important;
    font-size: 1.02rem;
    line-height: 2;
    margin: 0 auto;
    max-width: 900px;
}

/* -------------------------------------------------------
   متن و ورودی سؤال
------------------------------------------------------- */

[data-testid="stTextArea"] label,
[data-testid="stTextArea"] textarea {
    direction: rtl !important;
    text-align: right !important;
}

[data-testid="stTextArea"] textarea {
    font-size: 1rem !important;
    line-height: 2 !important;
    padding: 0.9rem !important;
}

/* placeholder */
[data-testid="stTextArea"] textarea::placeholder {
    direction: rtl !important;
    text-align: right !important;
    opacity: 0.75;
}

/* -------------------------------------------------------
   دکمه
------------------------------------------------------- */

[data-testid="stButton"] button {
    direction: rtl !important;
    text-align: center !important;
    font-family: "Vazirmatn", Tahoma, Arial, sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    min-height: 3rem;
}

/* -------------------------------------------------------
   پیام‌ها
------------------------------------------------------- */

[data-testid="stAlert"] {
    direction: rtl !important;
    text-align: right !important;
}

[data-testid="stAlert"] p {
    direction: rtl !important;
    text-align: right !important;
    line-height: 2 !important;
}

/* -------------------------------------------------------
   عنوان‌های RTL
------------------------------------------------------- */

.rtl-title {
    direction: rtl !important;
    text-align: right !important;
    font-family: "Vazirmatn", Tahoma, Arial, sans-serif !important;
    font-size: 1.35rem;
    font-weight: 700;
    line-height: 1.8;
    margin-top: 1.2rem;
    margin-bottom: 0.8rem;
}

/* -------------------------------------------------------
   کارت پاسخ
------------------------------------------------------- */

.answer-card {
    direction: rtl;
    text-align: right;
    font-family: "Vazirmatn", Tahoma, Arial, sans-serif;
    line-height: 2.1;
    font-size: 1.03rem;
    padding: 1.2rem 1.3rem;
    margin: 0.5rem 0 1rem 0;
    border: 1px solid rgba(128, 128, 128, 0.25);
    border-radius: 14px;
    background: rgba(128, 128, 128, 0.06);
}

.answer-card p,
.answer-card li,
.answer-card div {
    direction: rtl;
    text-align: right !important;
}

/* -------------------------------------------------------
   Expander
------------------------------------------------------- */

[data-testid="stExpander"] {
    direction: rtl !important;
    text-align: right !important;
}

[data-testid="stExpander"] summary {
    direction: rtl !important;
    text-align: right !important;
    font-family: "Vazirmatn", Tahoma, Arial, sans-serif !important;
}

[data-testid="stExpander"] summary span {
    direction: rtl !important;
    text-align: right !important;
}

[data-testid="stExpander"] p,
[data-testid="stExpander"] div,
[data-testid="stExpander"] span {
    direction: rtl !important;
    text-align: right !important;
}

/* -------------------------------------------------------
   متن بخش‌های بازیابی‌شده
------------------------------------------------------- */

.retrieved-title {
    direction: rtl !important;
    text-align: right !important;
    font-family: "Vazirmatn", Tahoma, Arial, sans-serif !important;
    font-weight: 600;
    line-height: 2;
}

/* جلوگیری از چپ‌چین شدن متن‌های Markdown */
.stMarkdown,
[data-testid="stMarkdownContainer"] {
    direction: rtl !important;
    text-align: right !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4,
[data-testid="stMarkdownContainer"] h5,
[data-testid="stMarkdownContainer"] h6 {
    direction: rtl !important;
    text-align: right !important;
    line-height: 2 !important;
}

/* -------------------------------------------------------
   کد و متن‌های فنی داخل Streamlit
------------------------------------------------------- */

code,
pre {
    direction: ltr;
    text-align: left;
}

/* -------------------------------------------------------
   حذف برخی فاصله‌های اضافی
------------------------------------------------------- */

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 3rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# API Key
# =========================================================

def get_api_key():
    """
    برای اجرای محلی:
        یا OPENROUTER_API_KEY را در secrets.toml بگذار
        یا موقتاً در محیط سیستم قرار بده.

    در Streamlit Cloud نیز باید همین Secret را تعریف کنی.
    در Render نیز از Environment Variable خوانده می‌شود.
    """

    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        return os.getenv("OPENROUTER_API_KEY")


API_KEY = get_api_key()


# =========================================================
# توابع کمکی
# =========================================================

def get_headers():
    if not API_KEY:
        raise RuntimeError(
            "کلید OpenRouter تنظیم نشده است. "
            "OPENROUTER_API_KEY را در Streamlit Secrets قرار بده."
        )

    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def calculate_hash(text: str) -> str:
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def split_text(text: str):
    """
    متن را بر اساس تعداد تقریبی کلمات به chunkهای کوچک تقسیم می‌کند.
    به دلیل محدودیت توکن مدل embedding، chunk کوچک انتخاب شده است.
    """
    text = re.sub(r"\n{3,}", "\n\n", text)
    words = text.split()
    if not words:
        return []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    chunks = []

    for start in range(0, len(words), step):
        chunk_words = words[start:start + CHUNK_SIZE]

        if not chunk_words:
            continue

        chunks.append(" ".join(chunk_words))

    return chunks
# =========================================================
# دریافت فایل
# =========================================================

@st.cache_data(show_spinner=False)
def download_file():
    response = requests.get(
        FILE_URL,
        timeout=60,
    )
    response.raise_for_status()

    return response.text
# =========================================================
# Embedding
# =========================================================

def create_embeddings(texts):
    embeddings = []

    batch_size = 10

    progress = st.progress(
        0,
        text="در حال ساخت embedding..."
    )

    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start:start + batch_size]

        response = requests.post(
            f"{BASE_URL}/embeddings",
            headers=get_headers(),
            json={
                "model": EMBEDDING_MODEL,
                "input": batch,
            },
            timeout=120,
        )

        if response.status_code != 200:
            raise RuntimeError(
                "Embedding Error "
                f"{response.status_code}

"
                f"{response.text}"
            )

        result = response.json()

        data = sorted(
            result["data"],
            key=lambda item: item["index"],
        )

        batch_vectors = [
            item["embedding"]
            for item in data
        ]

        embeddings.extend(batch_vectors)

        done = min(start + len(batch), total)
        progress.progress(
            done / total,
            text=f"در حال ساخت embedding... {done}/{total}"
        )

    progress.empty()

    return embeddings


# =========================================================
# Cache embedding
# =========================================================

def load_embedding_cache(file_hash):
    if not CACHE_FILE.exists():
        return None

    try:
        with CACHE_FILE.open("rb") as f:
            cache = pickle.load(f)

        if cache.get("model") != EMBEDDING_MODEL:
            return None

        if cache.get("file_hash") != file_hash:
            return None

        return cache

    except Exception:
        return None


def save_embedding_cache(file_hash, chunks, embeddings):
    cache = {
        "model": EMBEDDING_MODEL,
        "file_hash": file_hash,
        "chunks": chunks,
        "embeddings": embeddings,
    }

    with CACHE_FILE.open("wb") as f:
        pickle.dump(cache, f)


# =========================================================
# آماده‌سازی فایل
# =========================================================

@st.cache_resource(show_spinner=False)
def prepare_index(file_text, file_hash):
    cached = load_embedding_cache(file_hash)

    if cached is not None:
        return (
            cached["chunks"],
            cached["embeddings"],
            True,
        )

    chunks = split_text(file_text)

    if not chunks:
        raise RuntimeError("فایل خالی است.")

    embeddings = create_embeddings(chunks)

    save_embedding_cache(
        file_hash,
        chunks,
        embeddings,
    )

    return chunks, embeddings, False


# =========================================================
# Embedding سؤال
# =========================================================

def embed_query(question):
    response = requests.post(
        f"{BASE_URL}/embeddings",
        headers=get_headers(),
        json={
            "model": EMBEDDING_MODEL,
            "input": question,
        },
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Query Embedding Error "
            f"{response.status_code}

"
            f"{response.text}"
        )

    result = response.json()

    return np.array(
        result["data"][0]["embedding"],
        dtype=np.float32,
    )


# =========================================================
# Semantic Search
# =========================================================

def semantic_search(question, chunks, embeddings, top_k=TOP_K):
    query_vector = embed_query(question)

    matrix = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    query_norm = np.linalg.norm(query_vector)
    matrix_norm = np.linalg.norm(matrix, axis=1)

    similarities = (
        np.dot(matrix, query_vector)
        / (matrix_norm * query_norm + 1e-10)
    )

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    results = []

    for index in top_indices:
        results.append(
            {
                "chunk": chunks[index],
                "score": float(similarities[index]),
                "index": int(index),
            }
        )

    return results


# =========================================================
# Chat
# =========================================================

def ask_chat(question, results):
    context_parts = []

    for i, result in enumerate(results):
        context_parts.append(
            f"""
--- بخش {i + 1} ---
امتیاز شباهت: {result["score"]:.4f}

{result["chunk"]}
"""
        )

    context = "
".join(context_parts)

    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": """
تو یک دستیار تحقیقاتی هستی.

بر اساس بخش‌های بازیابی‌شده از فایل
به سؤال کاربر پاسخ بده.

پاسخ باید تا حد امکان بر اساس متن
ارائه‌شده باشد.

اگر پاسخ در بخش‌های بازیابی‌شده وجود ندارد،
صادقانه بگو که اطلاعات کافی در متن‌های
بازیابی‌شده وجود ندارد.

پاسخ را به زبان فارسی، دقیق و خوانا بنویس.
"""
            },
            {
                "role": "user",
                "content": f"""
بخش‌های مرتبط فایل:

{context}

================================

سؤال کاربر:

{question}
"""
            },
        ],
    }

    response = requests.post(
        f"{BASE_URL}/chat/completions",
        headers=get_headers(),
        json=payload,
        timeout=120,
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Chat Error "
            f"{response.status_code}

"
            f"{response.text}"
        )

    result = response.json()

    return result["choices"][0]["message"]["content"]


# =========================================================
# رابط کاربری
# =========================================================

# هدر وسط‌چین
st.markdown(
    """
<div class="page-header">
    <h1>🤖 پرسش از فایل با هوش مصنوعی</h1>
    <p>
        سؤال خود را درباره محتوای فایل وارد کنید.
        سیستم ابتدا جست‌وجوی معنایی انجام می‌دهد
        و سپس فقط بخش‌های مرتبط را به مدل می‌فرستد.
    </p>
</div>
""",
    unsafe_allow_html=True,
)


if not API_KEY:
    st.error(
        "API Key پیدا نشد. "
        "لطفاً OPENROUTER_API_KEY را در Streamlit Secrets تنظیم کن."
    )
    st.stop()


# ---------------------------------------------------------
# بارگذاری و آماده‌سازی فایل
# ---------------------------------------------------------

try:
    with st.spinner("در حال دریافت فایل و آماده‌سازی موتور جست‌وجوی معنایی..."):
        file_text = download_file()
        file_hash = calculate_hash(file_text)

        chunks, embeddings, from_cache = prepare_index(
            file_text,
            file_hash,
        )

    if from_cache:
        st.success(
            f"فایل آماده است — {len(chunks)} بخش از Cache بارگذاری شد."
        )
    else:
        st.success(
            f"فایل آماده شد — {len(chunks)} بخش برای Semantic Search ساخته شد."
        )

except Exception as e:
    st.error(f"خطا در آماده‌سازی فایل:

{e}")
    st.stop()


# ---------------------------------------------------------
# سؤال
# ---------------------------------------------------------

question = st.text_area(
    "سؤال شما:",
    height=140,
    placeholder="مثلاً: نویسنده درباره رابطه علم و قدرت چه توضیحی داده است؟",
)

ask_button = st.button(
    "🔎 جستجو و پرسش از هوش مصنوعی",
    type="primary",
)


# ---------------------------------------------------------
# پاسخ
# ---------------------------------------------------------

if ask_button:
    if not question.strip():
        st.warning("لطفاً ابتدا سؤال خود را وارد کنید.")
        st.stop()

    try:
        with st.spinner("در حال جستجوی معنایی و دریافت پاسخ..."):
            results = semantic_search(
                question,
                chunks,
                embeddings,
                TOP_K,
            )

            answer = ask_chat(
                question,
                results,
            )

        # عنوان پاسخ: راست‌چین
        st.markdown(
            '<div class="rtl-title">پاسخ</div>',
            unsafe_allow_html=True,
        )

        # کارت پاسخ
        st.markdown(
            '<div class="answer-card">',
            unsafe_allow_html=True,
        )

        st.markdown(
            answer,
            unsafe_allow_html=True,
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True,
        )

        # بخش‌های بازیابی‌شده
        with st.expander("مشاهده بخش‌های بازیابی‌شده"):
            for i, result in enumerate(results):
                st.markdown(
                    f'<div class="retrieved-title">بخش {i + 1} — '
                    f'Similarity: {result["score"]:.4f}</div>',
                    unsafe_allow_html=True,
                )
                st.write(result["chunk"])
                st.divider()

    except Exception as e:
        st.error(
            f"خطا در پردازش سؤال:

{e}"
        )
