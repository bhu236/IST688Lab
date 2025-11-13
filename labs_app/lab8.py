#!/usr/bin/env python3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import json
import re
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import shutil
from datetime import datetime
import tempfile
import tiktoken  # For token counting

# PDF processing libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except ImportError:
    try:
        import PyPDF2 as pypdf
        PdfReader = pypdf.PdfReader
        PYPDF_AVAILABLE = True
    except ImportError:
        PYPDF_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="SEC 10-Q Financial Analysis Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert > div {
        word-wrap: break-word !important;
        overflow-wrap: break-word !important;
        white-space: pre-wrap !important;
        max-width: 100% !important;
    }
    
    .financial-table {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        overflow-x: auto;
    }
    
    .risk-factor-box {
        background-color: #fff5f5;
        border-left: 4px solid #dc3545;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    
    .summary-box {
        background-color: #e7f3ff;
        border: 1px solid #0066cc;
        border-radius: 0.5rem;
        padding: 1.2rem;
        margin: 1rem 0;
        color: #004080;
    }
</style>
""", unsafe_allow_html=True)

# Token counting functions
@st.cache_resource
def get_tokenizer(model: str = "gpt-3.5-turbo"):
    """Get tokenizer for the model"""
    try:
        return tiktoken.encoding_for_model(model)
    except:
        return tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text"""
    tokenizer = get_tokenizer(model)
    return len(tokenizer.encode(text))

def truncate_to_token_limit(text: str, max_tokens: int, model: str = "gpt-3.5-turbo") -> str:
    """Truncate text to stay within token limit"""
    tokenizer = get_tokenizer(model)
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens])

# Model configurations with token limits
MODEL_CONFIGS = {
    "gpt-3.5-turbo": {
        "max_tokens": 4096,
        "context_window": 4096,
        "chunk_size": 400,  # Smaller chunks for safety
        "max_chunks_per_query": 5
    },
    "gpt-3.5-turbo-16k": {
        "max_tokens": 16384,
        "context_window": 16384,
        "chunk_size": 600,
        "max_chunks_per_query": 10
    },
    "gpt-4": {
        "max_tokens": 8192,
        "context_window": 8192,
        "chunk_size": 500,
        "max_chunks_per_query": 8
    },
    "gpt-4-turbo": {
        "max_tokens": 128000,
        "context_window": 128000,
        "chunk_size": 800,
        "max_chunks_per_query": 20
    }
}

# Initialize OpenAI client
@st.cache_resource
def get_openai_client():
    """Initialize OpenAI client with API key"""
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password", key="api_key_input")
        if not api_key:
            st.sidebar.warning("Please enter your OpenAI API key to continue")
            st.stop()
    return OpenAI(api_key=api_key)

# Initialize ChromaDB
@st.cache_resource
def init_chromadb():
    """Initialize ChromaDB with OpenAI embeddings"""
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = st.session_state.get('api_key_input', '')
    
    if not api_key:
        return None
    
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )
    
    try:
        client = chromadb.PersistentClient(path="./sec_10q_financial_db")
        
        try:
            collection = client.get_collection(
                name="financial_filings",
                embedding_function=openai_ef
            )
            try:
                _ = collection.count()
            except:
                client.delete_collection(name="financial_filings")
                collection = client.create_collection(
                    name="financial_filings",
                    embedding_function=openai_ef,
                    metadata={"hnsw:space": "cosine"}
                )
        except:
            collection = client.create_collection(
                name="financial_filings",
                embedding_function=openai_ef,
                metadata={"hnsw:space": "cosine"}
            )
        
        return collection
    except Exception as e:
        st.error(f"Error initializing ChromaDB: {str(e)}")
        return None

def extract_text_with_pdfplumber(file_content: bytes, filename: str) -> Dict[str, Any]:
    """Extract text and tables from PDF using pdfplumber with page limiting"""
    if not PDFPLUMBER_AVAILABLE:
        return None
    
    extracted_data = {
        "filename": filename,
        "text": "",
        "tables": [],
        "sections": {},
        "metadata": {
            "filename": filename,
            "pages": 0,
            "extraction_method": "pdfplumber"
        }
    }
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(file_content)
            tmp_path = tmp_file.name
        
        with pdfplumber.open(tmp_path) as pdf:
            total_pages = len(pdf.pages)
            extracted_data["metadata"]["pages"] = total_pages
            
            # Process pages in batches to avoid memory issues
            MAX_PAGES = 100  # Limit pages to process
            pages_to_process = min(total_pages, MAX_PAGES)
            
            if total_pages > MAX_PAGES:
                st.warning(f"PDF has {total_pages} pages. Processing first {MAX_PAGES} pages to avoid memory issues.")
            
            full_text = ""
            for i in range(pages_to_process):
                page = pdf.pages[i]
                
                # Extract text
                try:
                    page_text = page.extract_text()
                    if page_text:
                        # Limit text per page to avoid huge chunks
                        page_text = page_text[:5000]  # Limit to 5000 chars per page
                        full_text += f"\n--- Page {i+1} ---\n{page_text}"
                        
                        # Identify sections
                        if "risk factor" in page_text.lower():
                            if "risk_factors" not in extracted_data["sections"]:
                                extracted_data["sections"]["risk_factors"] = ""
                            extracted_data["sections"]["risk_factors"] += page_text[:2000] + "\n"
                        
                        if "financial statement" in page_text.lower():
                            if "financial_statements" not in extracted_data["sections"]:
                                extracted_data["sections"]["financial_statements"] = ""
                            extracted_data["sections"]["financial_statements"] += page_text[:2000] + "\n"
                except Exception as e:
                    st.warning(f"Error extracting text from page {i+1}: {str(e)}")
                    continue
                
                # Extract tables (limit to first 20 pages for tables)
                if i < 20:
                    try:
                        tables = page.extract_tables()
                        for table in tables[:3]:  # Limit tables per page
                            if table and len(table) > 1:
                                try:
                                    df = pd.DataFrame(table[1:], columns=table[0])
                                    extracted_data["tables"].append({
                                        "page": i + 1,
                                        "data": df,
                                        "raw": table
                                    })
                                except:
                                    pass
                    except Exception as e:
                        st.warning(f"Error extracting tables from page {i+1}: {str(e)}")
            
            # Limit total text size
            if len(full_text) > 100000:  # Limit to 100k chars
                full_text = full_text[:100000] + "\n[Text truncated due to size...]"
            
            extracted_data["text"] = full_text
        
        os.unlink(tmp_path)
        
    except Exception as e:
        st.error(f"Error with PDF extraction: {str(e)}")
        return None
    
    return extracted_data

def extract_text_with_pypdf(file_content: bytes, filename: str) -> Dict[str, Any]:
    """Fallback extraction using PyPDF with size limits"""
    if not PYPDF_AVAILABLE:
        return None
    
    extracted_data = {
        "filename": filename,
        "text": "",
        "sections": {},
        "metadata": {
            "filename": filename,
            "pages": 0,
            "extraction_method": "pypdf"
        }
    }
    
    try:
        import io
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PdfReader(pdf_file)
        
        total_pages = len(pdf_reader.pages)
        extracted_data["metadata"]["pages"] = total_pages
        
        MAX_PAGES = 100
        pages_to_process = min(total_pages, MAX_PAGES)
        
        if total_pages > MAX_PAGES:
            st.warning(f"PDF has {total_pages} pages. Processing first {MAX_PAGES} pages.")
        
        full_text = ""
        for i in range(pages_to_process):
            try:
                page_text = pdf_reader.pages[i].extract_text()
                page_text = page_text[:5000]  # Limit per page
                full_text += f"\n--- Page {i+1} ---\n{page_text}"
            except:
                continue
        
        # Limit total size
        if len(full_text) > 100000:
            full_text = full_text[:100000] + "\n[Text truncated...]"
        
        extracted_data["text"] = full_text
        
    except Exception as e:
        st.error(f"Error with pypdf extraction: {str(e)}")
        return None
    
    return extracted_data

def smart_chunk_financial_document(text: str, metadata: Dict = None, model: str = "gpt-3.5-turbo") -> List[Dict]:
    """Smart chunking with token awareness"""
    
    config = MODEL_CONFIGS.get(model, MODEL_CONFIGS["gpt-3.5-turbo"])
    chunk_size = config["chunk_size"]
    
    chunks = []
    
    # Clean text
    text = re.sub(r'\s+', ' ', text)
    text = text[:50000]  # Limit total text to process
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = []
    current_tokens = 0
    
    for sentence in sentences:
        sentence_tokens = count_tokens(sentence, model)
        
        if current_tokens + sentence_tokens > chunk_size and current_chunk:
            # Save current chunk
            chunk_text = ' '.join(current_chunk)
            chunk_metadata = {"chunk_tokens": current_tokens}
            if metadata:
                chunk_metadata.update(metadata)
            
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
            
            # Start new chunk
            current_chunk = [sentence]
            current_tokens = sentence_tokens
        else:
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
    
    # Add final chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        chunk_metadata = {"chunk_tokens": current_tokens}
        if metadata:
            chunk_metadata.update(metadata)
        
        chunks.append({
            "text": chunk_text,
            "metadata": chunk_metadata
        })
    
    # Limit total chunks
    MAX_CHUNKS = 50
    if len(chunks) > MAX_CHUNKS:
        st.info(f"Document has {len(chunks)} chunks. Using first {MAX_CHUNKS} chunks.")
        chunks = chunks[:MAX_CHUNKS]
    
    return chunks

def process_uploaded_documents(uploaded_files: List, collection, model: str = "gpt-3.5-turbo") -> int:
    """Process multiple uploaded documents with token management"""
    if not collection:
        st.error("Database not initialized")
        return 0
    
    total_chunks = 0
    
    for uploaded_file in uploaded_files:
        try:
            st.info(f"Processing {uploaded_file.name}...")
            
            # Check file size
            file_size = uploaded_file.size / (1024 * 1024)  # Size in MB
            if file_size > 10:
                st.warning(f"{uploaded_file.name} is {file_size:.1f}MB. Large files may take time to process.")
            
            # Read file content
            file_content = uploaded_file.read()
            
            # Extract content based on file type
            if uploaded_file.type == "application/pdf":
                extracted = extract_text_with_pdfplumber(file_content, uploaded_file.name)
                if not extracted:
                    extracted = extract_text_with_pypdf(file_content, uploaded_file.name)
                if not extracted:
                    st.error(f"Could not extract content from {uploaded_file.name}")
                    continue
                
                text_content = extracted["text"]
                
                # Store tables
                if extracted.get("tables"):
                    if "extracted_tables" not in st.session_state:
                        st.session_state.extracted_tables = {}
                    st.session_state.extracted_tables[uploaded_file.name] = extracted["tables"][:10]  # Limit tables
                    st.success(f"Extracted {len(extracted['tables'])} tables")
                
            elif uploaded_file.type == "text/plain":
                text_content = str(file_content, "utf-8")
                text_content = text_content[:100000]  # Limit text files too
            else:
                st.warning(f"Unsupported file type: {uploaded_file.type}")
                continue
            
            # Create chunks with metadata
            base_metadata = {
                "source": uploaded_file.name,
                "upload_date": datetime.now().isoformat(),
                "file_type": uploaded_file.type
            }
            
            chunks = smart_chunk_financial_document(text_content, base_metadata, model)
            
            if not chunks:
                st.warning(f"No chunks created from {uploaded_file.name}")
                continue
            
            # Add to collection in batches
            BATCH_SIZE = 10
            for i in range(0, len(chunks), BATCH_SIZE):
                batch = chunks[i:i+BATCH_SIZE]
                texts = [chunk["text"] for chunk in batch]
                metadatas = [chunk["metadata"] for chunk in batch]
                ids = [f"{uploaded_file.name}_{j}" for j in range(i, i+len(batch))]
                
                try:
                    collection.add(
                        documents=texts,
                        ids=ids,
                        metadatas=metadatas
                    )
                except Exception as e:
                    st.warning(f"Error adding batch {i//BATCH_SIZE + 1}: {str(e)}")
            
            total_chunks += len(chunks)
            st.success(f"✅ Added {len(chunks)} chunks from {uploaded_file.name}")
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
    
    return total_chunks

def retrieve_relevant_chunks(collection, query: str, n_results: int = 10, model: str = "gpt-3.5-turbo") -> List[Dict]:
    """Retrieve relevant chunks with token limits in mind"""
    if not collection:
        return []
    
    config = MODEL_CONFIGS.get(model, MODEL_CONFIGS["gpt-3.5-turbo"])
    max_chunks = config["max_chunks_per_query"]
    
    # Adjust n_results based on model
    n_results = min(n_results, max_chunks)
    
    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, collection.count() or 1),
            include=["documents", "metadatas", "distances"]
        )
        
        chunks = []
        for i in range(len(results['documents'][0])):
            chunks.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i]
            })
        
        return chunks
    except Exception as e:
        st.error(f"Error retrieving chunks: {str(e)}")
        return []

def generate_financial_analysis(client, query: str, chunks: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate analysis with proper token management"""
    
    if not chunks:
        return "No relevant information found to answer your question."
    
    config = MODEL_CONFIGS.get(model, MODEL_CONFIGS["gpt-3.5-turbo"])
    max_context_tokens = config["context_window"] - 1500  # Reserve tokens for response
    
    # Build context within token limits
    context_parts = []
    total_tokens = 0
    
    for chunk in chunks:
        chunk_text = f"[Source: {chunk['metadata'].get('source', 'Unknown')}]\n{chunk['text']}\n"
        chunk_tokens = count_tokens(chunk_text, model)
        
        if total_tokens + chunk_tokens > max_context_tokens:
            break
        
        context_parts.append(chunk_text)
        total_tokens += chunk_tokens
    
    context = "\n".join(context_parts)
    
    # Create prompt
    prompt = f"""You are an expert financial analyst reviewing SEC 10-Q filings.

Based on the following context, answer the question accurately:

Context:
{context}

Question: {query}

Provide a detailed, structured response based on the available information."""

    # Count total tokens
    prompt_tokens = count_tokens(prompt, model)
    
    if prompt_tokens > config["context_window"] - 500:
        # Truncate if still too long
        context = truncate_to_token_limit(context, max_context_tokens - 500, model)
        prompt = f"""Based on the following context, answer the question:

Context: {context}

Question: {query}"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert financial analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=min(1500, config["max_tokens"] - prompt_tokens)
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        if "token" in str(e).lower():
            return "The document is too large. Try using a model with larger context window (gpt-3.5-turbo-16k or gpt-4-turbo) or reducing the number of chunks."
        return f"Error generating analysis: {str(e)}"

def main():
    st.title("📈 SEC 10-Q Financial Analysis Platform")
    st.markdown("*Advanced document analysis with token management*")
    
    # Initialize
    client = get_openai_client()
    collection = init_chromadb()
    
    if not collection:
        st.error("Failed to initialize database.")
        return
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 📁 Document Management")
        
        # Model selection (moved up for visibility)
        st.markdown("### 🤖 AI Model")
        model = st.selectbox(
            "Select Model",
            ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4", "gpt-4-turbo"],
            index=1,  # Default to 16k model
            help="Use gpt-3.5-turbo-16k or gpt-4-turbo for large documents"
        )
        st.session_state.model = model
        
        # Show token limits
        config = MODEL_CONFIGS.get(model, MODEL_CONFIGS["gpt-3.5-turbo"])
        st.info(f"**Token Limit:** {config['context_window']:,} tokens\n**Chunk Size:** {config['chunk_size']} tokens")
        
        # File upload
        st.markdown("### Upload 10-Q Filings")
        uploaded_files = st.file_uploader(
            "Choose documents",
            type=['pdf', 'txt'],
            accept_multiple_files=True,
            help="Upload PDF or TXT files. Large files will be automatically chunked."
        )
        
        if uploaded_files:
            if st.button("📥 Process Documents", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(uploaded_files)} documents..."):
                    total_chunks = process_uploaded_documents(uploaded_files, collection, model)
                    if total_chunks:
                        st.success(f"✅ Processed {len(uploaded_files)} documents ({total_chunks} chunks)")
                    else:
                        st.error("Failed to process documents")
        
        # Database management
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear DB", use_container_width=True):
                try:
                    client_db = chromadb.PersistentClient(path="./sec_10q_financial_db")
                    client_db.delete_collection("financial_filings")
                    st.cache_resource.clear()
                    st.success("Database cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        
        with col2:
            doc_count = collection.count() if collection else 0
            st.metric("Chunks", doc_count)
        
        # Analysis settings
        st.markdown("### ⚙️ Settings")
        
        st.session_state.num_chunks = st.slider(
            "Max chunks to retrieve",
            min_value=3,
            max_value=config["max_chunks_per_query"],
            value=min(5, config["max_chunks_per_query"]),
            help=f"Model can handle up to {config['max_chunks_per_query']} chunks"
        )
        
        # Example queries
        st.markdown("### 📝 Example Queries")
        examples = [
            "Summarize the financial performance",
            "What are the main risks?",
            "Revenue and profit trends",
            "Cash flow analysis",
            "Segment performance"
        ]
        
        for ex in examples:
            if st.button(f"→ {ex}", key=ex, use_container_width=True):
                st.session_state.query = ex
        
        # Tips
        with st.expander("💡 Tips for Large Files"):
            st.markdown("""
            **If you get token limit errors:**
            1. Use **gpt-3.5-turbo-16k** or **gpt-4-turbo**
            2. Reduce number of chunks
            3. Process fewer pages
            4. Split very large PDFs
            
            **Best practices:**
            - Files under 10MB work best
            - Use specific queries
            - Start with summaries
            """)
    
    # Main area
    st.markdown("---")
    
    if collection and collection.count() == 0:
        st.warning("📂 No documents loaded. Please upload 10-Q filings using the sidebar.")
    
    # Query interface
    query = st.text_area(
        "💭 Enter your query:",
        placeholder="Example: What are the top risks mentioned in the filing?",
        value=st.session_state.get('query', ''),
        height=100,
        key="main_query"
    )
    
    col1, col2 = st.columns([1, 9])
    with col1:
        analyze_button = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if query and analyze_button:
        with st.spinner("Analyzing..."):
            # Show token info
            query_tokens = count_tokens(query, model)
            st.caption(f"Query tokens: {query_tokens}")
            
            # Retrieve chunks
            chunks = retrieve_relevant_chunks(
                collection, 
                query, 
                st.session_state.get("num_chunks", 5),
                model
            )
            
            if chunks:
                # Show sources
                sources = list(set([chunk['metadata'].get('source', 'Unknown') for chunk in chunks]))
                st.info(f"📄 Analyzing from: {', '.join(sources[:3])}")
                
                # Generate analysis
                analysis = generate_financial_analysis(client, query, chunks, model)
                
                # Display results
                st.markdown("## 📊 Analysis")
                st.markdown(analysis)
                
                # Show chunks used
                with st.expander(f"📚 Context used ({len(chunks)} chunks)"):
                    for i, chunk in enumerate(chunks):
                        st.caption(f"**Chunk {i+1}** from {chunk['metadata'].get('source', 'Unknown')}")
                        preview = chunk['text'][:200] + "..."
                        st.text(preview)
            else:
                st.warning("No relevant information found.")

if __name__ == "__main__":
    # Check for required packages
    try:
        import tiktoken
    except ImportError:
        st.error("Please install tiktoken: pip install tiktoken")
        st.stop()
    
    main()
    