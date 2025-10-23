# lab6.py
import streamlit as st
from openai import OpenAI
import json, os, re

def app():
    # Load API Key
    api_key = (
        st.secrets["openai"]["api_key"]
        if "openai" in st.secrets
        else os.getenv("OPENAI_API_KEY")
    )
    client = OpenAI(api_key=api_key)

    # Streamlit UI
    st.title("AI Fact-Checker ✅ + Citation Builder🔗")
    st.write("Verify factual claims using live web search and structured reasoning.")

    if "fact_history" not in st.session_state:
        st.session_state.fact_history = []

    # Helper: Extract JSON Safely
    def extract_json(text):
        """Extracts first valid JSON block from text."""
        try:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception:
            pass
        return None

    # Core Function
    def fact_check_claim(user_claim: str):
        """Fact-checks a claim using Responses API with web_search tool."""
        resp = client.responses.create(
            model="gpt-4.1",
            input=[
                {
                    "role": "system",
                    "content": """
                    You are a factual verification assistant.
                    Search the web for each claim and return ONLY valid JSON like this:
                    {
                      "claim": "input claim",
                      "verdict": "True" | "False" | "Partly True",
                      "explanation": "brief explanation",
                      "sources": ["url1", "url2"]
                    }
                    Do not include any markdown, text, or commentary outside the JSON.
                    """
                },
                {"role": "user", "content": user_claim}
            ],
            tools=[{"type": "web_search"}],
            temperature=0.2,
        )
        return resp.output_text

    # Streamlit Layout
    user_claim = st.text_input("Enter a factual claim:")

    if st.button("Check Fact"):
        if not user_claim.strip():
            st.warning("Enter a claim first.")
        else:
            with st.spinner("Verifying..."):
                result_text = fact_check_claim(user_claim)
                result_json = extract_json(result_text)

                if not result_json:
                    st.error("Invalid JSON response — showing raw output below:")
                    st.write(result_text)
                    st.stop()

                # --- Confidence calculation based on number of sources ---
                sources = result_json.get("sources", [])
                confidence = min(1.0, len(sources) / 3)
                result_json["confidence_score"] = round(confidence * 100, 0)

                # Display Result
                st.subheader("Result")
                st.json(result_json)

                if sources:
                    st.markdown("#### Sources")
                    for s in sources:
                        st.markdown(f"- [{s}]({s})")

                st.markdown(f"**Confidence Score:** {confidence * 100:.0f}%")

                # Save history
                st.session_state.fact_history.append({
                    "claim": user_claim,
                    "verdict": result_json.get("verdict"),
                    "confidence": confidence
                })

    # Previous Checks
    if st.session_state.fact_history:
        st.markdown("---")
        st.markdown("### Previous Checks")
        for i, entry in enumerate(reversed(st.session_state.fact_history), start=1):
            st.markdown(
                f"{i}. **{entry['claim']}** — {entry['verdict']} "
                f"({entry['confidence']*100:.0f}% confidence)"
            )
