import hashlib
import pickle
import re
import os
import json
import uuid
from pathlib import Path
import numpy as np
import requests
import streamlit as st
from datetime import datetime, timedelta, timezone
from supabase import create_client

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
MEMORY_MESSAGES = 10
MEMORY_RETENTION_DAYS = 3

# =========================================================
# تنظیمات صفحه
# =========================================================

st.set_page_config(
    page_title="پرسش از فایل با هوش مصنوعی",
    page_icon="🤖",
    layout="wide",
)
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;
}

.stApp {
    direction: rtl;
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;
}

p, span, div, input, textarea, button {
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;
}

input, textarea {
    direction: rtl !important;
    text-align: right !important;
}
.stTextInput > div > div > input {
    direction: rtl;
    text-align: right;
    font-family: 'Vazirmatn', Tahoma, sans-serif;
}

.stTextArea textarea {
    direction: rtl;
    text-align: right;
    font-family: 'Vazirmatn', Tahoma, sans-serif;
}

button {
    font-family: 'Vazirmatn', Tahoma, sans-serif;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# API Key
# =========================================================

def get_api_key():
    """
    برای اجرای محلی:
        یا OPENROUTER_API_KEY را در secrets.toml بگذار
        یا موقتاً در محیط سیستم قرار بده.

    در Streamlit Cloud نیز باید همین Secret را تعریف کنی.
    """

    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        return os.getenv("OPENROUTER_API_KEY")


API_KEY = get_api_key()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

if not SUPABASE_URL:
    st.error("SUPABASE_URL تنظیم نشده است.")
    st.stop()

if not SUPABASE_SECRET_KEY:
    st.error("SUPABASE_SECRET_KEY تنظیم نشده است.")
    st.stop()

@st.cache_resource
def get_supabase():
    return create_client(
        SUPABASE_URL,
        SUPABASE_SECRET_KEY
    )

supabase = get_supabase()

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
#......................................................
def get_memory(session_id, limit=MEMORY_MESSAGES):
    response = (
        supabase
        .table("messages")
        .select("role, content, created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )

    messages = response.data or []

    return list(reversed(messages))
# ........................ delete old memory....................
def delete_old_messages():
    cutoff = (
        datetime.now(timezone.utc)
        - timedelta(days=MEMORY_RETENTION_DAYS)
    )

    (
        supabase
        .table("messages")
        .delete()
        .lt("created_at", cutoff.isoformat())
        .execute()
    )

delete_old_messages()

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
#...................... tool function 1...................
def get_student_age(name):
    if name == "علی":
        return 36

    return None
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_student_age",
            "description": "سن یک دانش‌آموز را بر اساس نام او برمی‌گرداند.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "نام دانش‌آموز"
                    }
                },
                "required": ["name"]
            }
        }
    }
]
#...............................................
def save_message(role, content):
    response = (
        supabase
        .table("messages")
        .insert({
            "session_id": st.session_state.session_id,
            "role": role,
            "content": content
        })
        .execute()
    )

    return response

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
                f"{response.status_code}\n\n"
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
# پیدا کردن فایل index شده در Supabase
# =========================================================

def get_existing_document(file_hash):

    response = (
        supabase
        .table("documents")
        .select("id, chunk_count, embedding_model")
        .eq("file_hash", file_hash)
        .limit(1)
        .execute()
    )

    if response.data:
        return response.data[0]

    return None


def get_previous_document():
    response = (
        supabase
        .table("documents")
        .select("id, file_hash, created_at")
        .eq("source_url", FILE_URL)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )

    if response.data:
        return response.data[0]

    return None


def delete_document(document_id):
    (
        supabase
        .table("documents")
        .delete()
        .eq("id", document_id)
        .execute()
    )


# =========================================================
# ساخت رکورد Document
# =========================================================

def create_document(file_hash, chunk_count):

    response = (
        supabase
        .table("documents")
        .insert({
            "file_hash": file_hash,
            "source_url": FILE_URL,
            "chunk_count": chunk_count,
            "embedding_model": EMBEDDING_MODEL
        })
        .select("id")
        .execute()
    )

    if not response.data:
        raise RuntimeError(
            "رکورد Document در Supabase ساخته نشد."
        )

    return response.data[0]["id"]

# =========================================================
# ذخیره Chunkها و Embeddingها در Supabase
# =========================================================

def save_chunks(document_id, chunks, embeddings):

    rows = []

    for index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings)
    ):
        rows.append({
            "document_id": document_id,
            "chunk_index": index,
            "content": chunk,
            "embedding": embedding
        })

    batch_size = 20

    progress = st.progress(
        0,
        text="در حال ذخیره embeddingها در Supabase..."
    )

    total = len(rows)

    for start in range(0, total, batch_size):

        batch = rows[start:start + batch_size]

        (
            supabase
            .table("document_chunks")
            .insert(batch)
            .execute()
        )

        done = min(start + len(batch), total)

        progress.progress(
            done / total,
            text=f"ذخیره embeddingها... {done}/{total}"
        )

    progress.empty()


# =========================================================
# آماده‌سازی فایل
# =========================================================

def prepare_index(file_text, file_hash):

    existing = get_existing_document(
        file_hash
    )

    if existing is not None:
        return (
            existing["id"],
            existing["chunk_count"],
            True
        )

    previous = get_previous_document()

    chunks = split_text(
        file_text
    )

    if not chunks:
        raise RuntimeError("فایل خالی است.")

    embeddings = create_embeddings(
        chunks
    )

    # نسخه جدید ابتدا کامل ساخته و ذخیره می‌شود.
    document_id = create_document(
        file_hash,
        len(chunks)
    )

    save_chunks(
        document_id,
        chunks,
        embeddings
    )

    # فقط بعد از موفقیت نسخه جدید، نسخه قبلی حذف می‌شود.
    if previous is not None:
        if previous["id"] != document_id:
            delete_document(previous["id"])

    return (
        document_id,
        len(chunks),
        False
    )


# =========================================================
# Embedding سؤال
# =========================================================

def embed_query(question):

    response = requests.post(
        f"{BASE_URL}/embeddings",
        headers=get_headers(),
        json={
            "model": EMBEDDING_MODEL,
            "input": question
        },
        timeout=60
    )

    if response.status_code != 200:
        raise RuntimeError(
            "Query Embedding Error "
            f"{response.status_code}\n\n"
            f"{response.text}"
        )

    result = response.json()

    return result["data"][0]["embedding"]


# =========================================================
# Semantic Search در Supabase
# =========================================================

def semantic_search(
    document_id,
    question,
    top_k=TOP_K
):

    query_embedding = embed_query(
        question
    )

    response = (
        supabase
        .rpc(
            "match_document_chunks",
            {
                "query_embedding": query_embedding,
                "match_count": top_k,
                "p_document_id": document_id
            }
        )
        .execute()
    )

    if not response.data:
        return []

    return [
        {
            "chunk": row["content"],
            "score": float(row["similarity"]),
            "index": int(row["chunk_index"])
        }
        for row in response.data
    ]


# =========================================================
# Chat
# =========================================================

def ask_chat(question, results, history):
    context_parts = []

    for i, result in enumerate(results):
        context_parts.append(
            f"""
--- بخش {i + 1} ---
امتیاز شباهت: {result["score"]:.4f}

{result["chunk"]}
"""
        )

    context = "\n".join(context_parts)

    messages = [
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
        }
    ]

    for message in history:
        messages.append({
            "role": message["role"],
            "content": message["content"]
        })

    messages.append({
        "role": "user",
        "content": f"""
بخش‌های مرتبط فایل:

{context}

================================

سؤال کاربر:

{question}
"""
    })

    payload = {
        "model": CHAT_MODEL,
        "messages": messages,
        "tools": tools,
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
            f"{response.status_code}\n\n"
            f"{response.text}"
        )

    result = response.json()

    message = result["choices"][0]["message"]

    if "tool_calls" in message:
        tool_call = message["tool_calls"][0]

        function_name = tool_call["function"]["name"]

        arguments = json.loads(
            tool_call["function"]["arguments"]
        )

        if function_name == "get_student_age":
            tool_result = get_student_age(
                arguments["name"]
            )

            return f"نتیجه ابزار: {tool_result}"

    return message["content"]
# =========================================================
# رابط کاربری
# =========================================================

st.title("🤖 پرسش از فایل با هوش مصنوعی")

st.write(
    "سؤال خود را درباره محتوای فایل وارد کنید. "
    "سیستم ابتدا جست‌وجوی معنایی انجام می‌دهد "
    "و سپس فقط بخش‌های مرتبط را به مدل می‌فرستد."
)

if not API_KEY:
    st.error(
        "API Key پیدا نشد. "
        "لطفاً OPENROUTER_API_KEY را در Environment Variables تنظیم کن."
    )
    st.stop()


# ---------------------------------------------------------
# بارگذاری و آماده‌سازی فایل
# ---------------------------------------------------------

try:
    with st.spinner("در حال دریافت فایل و آماده‌سازی موتور جست‌وجوی معنایی..."):
        file_text = download_file()
        file_hash = calculate_hash(file_text)

        document_id, chunk_count, from_database = prepare_index(
            file_text,
            file_hash,
        )

    if from_database:
        st.success(
            f"فایل آماده است — {chunk_count} بخش از Supabase بارگذاری شد."
        )
    else:
        st.success(
            f"فایل آماده شد — {chunk_count} بخش در Supabase ذخیره شد."
        )

except Exception as e:
    st.error(f"خطا در آماده‌سازی فایل:\n\n{e}")
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
            document_id,
            question,
            TOP_K,
            )

            history = get_memory(
            st.session_state.session_id
             )

            answer = ask_chat(
                question,
                results,
                history,
            )
        save_message("user", question)
        save_message("assistant", answer)

        st.subheader("پاسخ")

        st.markdown(answer)

        with st.expander("مشاهده بخش‌های بازیابی‌شده"):
            for i, result in enumerate(results):
                st.markdown(
                    f"**بخش {i + 1} — "
                    f"Similarity: {result['score']:.4f}**"
                )
                st.write(result["chunk"])
                st.divider()

    except Exception as e:
        st.error(
            f"خطا در پردازش سؤال:\n\n{e}"
        )
