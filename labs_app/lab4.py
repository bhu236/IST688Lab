# lab4_app.py

import streamlit as st
from openai import OpenAI
import os
from PyPDF2 import PdfReader

# Fix for working with ChromaDB and Streamlit (SQLite conflicts)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from chromadb.utils import embedding_functions


def app():
    # ----------------------------
    # Initialize OpenAI client
    # ----------------------------
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("Please set your OPENAI_API_KEY environment variable.")
        return
    client = OpenAI(api_key=openai_api_key)

    # ----------------------------
    # Initialize ChromaDB
    # ----------------------------
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="lab4_collection")

    # ----------------------------
    # Streamlit App UI
    # ----------------------------
    st.title("📄 Lab 4 - PDF Q&A with ChromaDB + OpenAI")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)

        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

        st.success("PDF uploaded and text extracted successfully.")

        # Split text into chunks for embeddings
        chunks = [text[i:i + 500] for i in range(0, len(text), 500)]

        # Add chunks into ChromaDB
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                ids=[f"doc_{i}"]
            )

        st.write(f"✅ Stored {len(chunks)} chunks in ChromaDB.")

        # User query
        query = st.text_input("Ask a question about your PDF:")

        if query:
            # Search ChromaDB for relevant chunks
            results = collection.query(query_texts=[query], n_results=3)
            retrieved_docs = results["documents"][0]

            # Combine into context
            context = "\n".join(retrieved_docs)

            # Send to OpenAI
            prompt = f"""
            You are a helpful assistant. Use the following context to answer the user's question.

            Context:
            {context}

            Question: {query}
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )

            answer = response.choices[0].message.content
            st.subheader("Answer")
            st.write(answer)


# For standalone execution
if __name__ == "__main__":
    app()
