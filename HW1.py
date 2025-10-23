import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF
    
# Show title and description.
st.title("🎈FileBot - HW1 - Bhushan Jain ")
st.write(
    "Upload a document below and ask a question about it – GPT will answer! "
    "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask user for their OpenAI API key
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file (.txt or .pdf only)
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .pdf)", type=("txt", "pdf")
    )

    # Clear access if file removed
    if uploaded_file is None and "document" in st.session_state:
        del st.session_state["document"]

    # Process file
    if uploaded_file:
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension == "txt":
            st.session_state["document"] = uploaded_file.read().decode()
        elif file_extension == "pdf":
            st.session_state["document"] = read_pdf(uploaded_file)
        else:
            st.error("Unsupported file type.")

    # Ask the user for a question
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Is this course hard?",
        disabled="document" not in st.session_state,
    )

    # Models to test
    models = {
        "gpt-3.5": "gpt-3.5-turbo-0125",
        "gpt-4.1": "gpt-4.1",
        "gpt-5-chat-latest": "gpt-5-chat-latest",
        "gpt-5-nano": "gpt-5-nano",
    }

    if "document" in st.session_state and question:
        for label, mid in models.items():
            st.subheader(f"🤖 Model: {label}")
            try:
                response = client.chat.completions.create(
                    model=mid,
                    messages=[
                        {"role": "system", "content": "Answer using only the document provided."},
                        {"role": "user", "content": f"Document: {st.session_state['document']}\n\nQuestion: {question}"}
                    ],
                )
                st.write(response.choices[0].message.content)
            except Exception as e:
                st.error(f"{label} failed: {e}")


# Function to read PDF files
def read_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text
