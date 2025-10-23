import streamlit as st
import fitz  # PyMuPDF
from openai import OpenAI

# Helper: read text from uploaded PDF
def read_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

def app():
    # Title for Lab 2
    st.title("Lab 2 — PDF Summarizer")

    # Use API key from secrets (Lab 2b requirement)
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    # Sidebar controls
    st.sidebar.header("Summary Options")

    # Dropdown for language selection
    lang = st.radio(
    "Select language",
    ["English", "Spanish", "French", "German"],
    index=0
    )

    # Dropdown for summary type
    summary_type = st.radio(
        "Type of summary",
        [
            "Summarize in 100 words",
            "Summarize in 2 connecting paragraphs",
            "Summarize in 5 bullet points"
        ],
        index=0,
    )

    # Checkbox for model selection (Lab 2c requirement)
    use_advanced = st.sidebar.checkbox("Use Advanced Model (gpt-4o)")
    model_id = "gpt-4o" if use_advanced else "gpt-4o-mini"

    # File upload (.pdf only)
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded_file:
        # Extract PDF text
        doc_text = read_pdf(uploaded_file)

        if st.button("Generate Summary"):
            # Build the prompt (Lab 2c requirement)
            prompt = (
                f"Please {summary_type.lower()}.\n"
                f"Write the summary in {lang}.\n\n"
                f"Document:\n{doc_text}"
            )

            try:
                response = client.responses.create(
                    model=model_id,
                    input=prompt,
                )
                st.subheader("Summary")
                st.write(response.output_text)
            except Exception as e:
                st.error(f"Error generating summary: {e}")

    # Lab 2d: note about default model
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "ℹ️ Default model is **gpt-4o-mini**, because it provides a good balance of quality, cost, and speed. "
        "Advanced model (gpt-4o) is available for higher quality when needed."
    )
    st.button("Run")
