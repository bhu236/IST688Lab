# HW5.py
import os
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
import json

# Fix for working with ChromaDB and Streamlit (SQLite conflicts)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils import embedding_functions

# ----------------------------
# CONFIG / KEYS
# ----------------------------
st.set_page_config(page_title="HW5 - Short-Term Memory Doc Bot", layout="centered")

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Please set your OPENAI_API_KEY in streamlit secrets or env.")
    st.stop()

# init OpenAI client (new client style)
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------
# Chroma DB (in-memory) init
# ----------------------------
chroma_client = chromadb.Client()

# choose a collection name
COLLECTION_NAME = "hw5_collection_v1"
# create or get collection
# Use OpenAI embedding function wrapper provided by chromadb if available
emb_func = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"  # change if desired
)

# create collection with embedding function so collection.query can embed queries
collection = chroma_client.create_collection(
    name=COLLECTION_NAME,
    get_or_create=True,
    embedding_function=emb_func
)

# ----------------------------
# Helper: chunking function
# ----------------------------
def chunk_text(text: str, size: int = 500, overlap: int = 50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start += size - overlap
    return chunks

# ----------------------------
# Function: index uploaded document(s)
# ----------------------------
def index_pdf(file, prefix="doc"):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"

    chunks = chunk_text(text, size=600, overlap=100)
    ids = [f"{prefix}_{i}" for i in range(len(chunks))]

    # add to chroma collection (documents + metadata)
    collection.add(documents=chunks, ids=ids, metadatas=[{"source": getattr(file, "name", "uploaded_pdf")} for _ in chunks])
    return len(chunks)

# ----------------------------
# Function: index CSV (optional) - for courses/clubs style
# ----------------------------
def index_csv(file, text_column="description", prefix="csv"):
    import pandas as pd
    df = pd.read_csv(file)
    if text_column not in df.columns:
        # fallback: combine all columns
        df["__combined__"] = df.astype(str).apply(lambda r: " | ".join(r.values), axis=1)
        text_col = "__combined__"
    else:
        text_col = text_column

    docs = df[text_col].astype(str).tolist()
    # chunk each doc if too long
    all_chunks = []
    ids = []
    for i, doc in enumerate(docs):
        chunks = chunk_text(doc, size=600, overlap=100)
        for j, c in enumerate(chunks):
            all_chunks.append(c)
            ids.append(f"{prefix}_{i}_{j}")

    collection.add(documents=all_chunks, ids=ids, metadatas=[{"source": getattr(file, "name", "uploaded_csv")} for _ in all_chunks])
    return len(all_chunks)

# ----------------------------
# Function required by your assignment:
# takes `query` and returns `relevant_course_info` (the vector search output)
# ----------------------------
def get_relevant_info(query: str, n_results: int = 4) -> str:
    """
    Query Chroma collection and return concatenated retrieved docs as the 'relevant_course_info'.
    """
    if query.strip() == "":
        return ""

    results = collection.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas", "distances"])
    docs = results.get("documents", [[]])[0]  # list of retrieved texts
    metadatas = results.get("metadatas", [[]])[0]
    if not docs:
        return ""

    # Compose a compact context string including small metadata snippets
    parts = []
    for i, d in enumerate(docs):
        md = metadatas[i] if i < len(metadatas) else {}
        source = md.get("source", "")
        parts.append(f"Source: {source}\n{d}\n---")
    context = "\n".join(parts)
    return context

# ----------------------------
# LLM invocation function that uses the vector search context
# ----------------------------
def ask_with_context(query: str, context: str, temperature: float = 0.2, max_tokens: int = 400) -> str:
    """
    Sends a chat request to the LLM using the vector search result in the system prompt.
    This follows your instructions: don't use function-calling here — just pass the context.
    Also maintain short-term memory via session state.
    """
    # short-term memory from session (simple list of user/assistant pairs)
    history = st.session_state.get("history", [])

    # Build messages: system contains instructions + retrieved context
    system_prompt = (
        "You are a helpful assistant specialized in answering questions about the user's uploaded documents. "
        "Use only the provided Context from the document when answering. If the answer is not in the context, say you don't know and optionally suggest where to find it.\n\n"
        "Context (from vector search):\n"
        f"{context}\n\n"
        "Now answer clearly and concisely."
    )

    messages = [{"role": "system", "content": system_prompt}]

    # Append short-term memory (last few exchanges)
    for role, text in history[-6:]:  # only last 6 messages to keep prompt small
        messages.append({"role": role, "content": text})

    # Finally append the current user query
    messages.append({"role": "user", "content": query})

    # Call LLM
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )

    answer = resp.choices[0].message.content

    # Update short-term memory
    history.append(("user", query))
    history.append(("assistant", answer))
    st.session_state["history"] = history

    return answer

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("HW5 — Short-Term Memory Document Chatbot")
st.markdown("""
Upload a PDF (or CSV) of your course/club data, index it, then ask questions about those documents.
This app performs a vector search to retrieve relevant document fragments and then calls the LLM with that context.
""")

# show existing collection size (if any)
try:
    col_info = collection.count()
    st.write(f"Indexed items in collection: {col_info.get('count', 'unknown')}")
except Exception:
    st.write("Collection not created yet or cannot read count.")

# Upload
st.subheader("Index documents")
col1, col2 = st.columns(2)
with col1:
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_pdf is not None:
        n = index_pdf(uploaded_pdf, prefix="pdf")
        st.success(f"Indexed {n} PDF chunks.")

with col2:
    uploaded_csv = st.file_uploader("Upload CSV (courses/clubs)", type=["csv"])
    if uploaded_csv is not None:
        # try to auto-detect a text column name, fallbacks applied inside index_csv
        n = index_csv(uploaded_csv, text_column="description", prefix="csv")
        st.success(f"Indexed {n} CSV chunks.")

# Show a simple history viewer and clear option
st.subheader("Chat / Query")
if "history" not in st.session_state:
    st.session_state["history"] = []

if st.button("Clear short-term memory"):
    st.session_state["history"] = []
    st.success("Cleared chat memory.")

query = st.text_input("Ask a question about your indexed documents:")

if st.button("Ask") and query:
    with st.spinner("Searching vector DB and contacting LLM..."):
        context = get_relevant_info(query, n_results=4)
        if not context.strip():
            st.warning("No relevant information found in the indexed documents. Try rephrasing or index more documents.")
        else:
            # Now call the LLM with the context
            answer = ask_with_context(query, context)
            st.subheader("Answer")
            st.write(answer)

# Display conversation history
if st.session_state.get("history"):
    st.write("---")
    st.subheader("Recent conversation (short-term memory)")
    for role, text in st.session_state["history"][-10:]:
        if role == "user":
            st.markdown(f"**User:** {text}")
        else:
            st.markdown(f"**Assistant:** {text}")
