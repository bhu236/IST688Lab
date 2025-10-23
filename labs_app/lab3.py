import streamlit as st
from openai import OpenAI

# Max tokens for buffer
MAX_TOKENS = 1000  # Adjust as needed

# Simple token counting
def count_tokens(messages):
    return sum(len(m["content"].split()) for m in messages)

def app():
    st.title("Lab 3 — Interactive Kid-Friendly Streaming Chatbot")

    # Initialize OpenAI client
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception:
        st.error("OpenAI API key not found in secrets.toml")
        return

    # Sidebar: model options
    st.sidebar.header("Model Options")
    use_advanced = st.sidebar.checkbox("Use Advanced Model (gpt-4o)")
    model_id = "gpt-4o" if use_advanced else "gpt-4o-mini"

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "awaiting_more_info" not in st.session_state:
        st.session_state.awaiting_more_info = False

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if user_input := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Determine prompt based on follow-up state
        if st.session_state.awaiting_more_info:
            # User answers "Do you want more info?"
            if user_input.strip().lower() in ["yes", "y"]:
                prompt = "Provide more explanation in simple language for a 10-year-old. Then ask: 'Do you want more info?'"
            else:
                prompt = "Ask the user: 'What else can I help you with?'"
            st.session_state.awaiting_more_info = False
        else:
            # First answer to question
            prompt = f"Answer this question in simple language for a 10-year-old: {user_input}\nThen ask 'Do you want more info?'"
            st.session_state.awaiting_more_info = True

        # Build conversation buffer
        user_msgs = [m for m in st.session_state.messages if m["role"] == "user"][-2:]
        assistant_msgs = [m for m in st.session_state.messages if m["role"] == "assistant"][-2:]

        buffer = []
        for i in range(len(user_msgs)):
            buffer.append(user_msgs[i])
            if i < len(assistant_msgs):
                buffer.append(assistant_msgs[i])

        # If first message
        if len(buffer) == 0:
            buffer.append({"role": "system", "content": "You are a helpful assistant that explains things so a 10-year-old can understand."})

        # Enforce max token limit
        while count_tokens(buffer) > MAX_TOKENS and len(buffer) > 2:
            buffer.pop(0)

        # Show user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # Stream assistant response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            try:
                stream = client.chat.completions.create(
                    model=model_id,
                    messages=buffer + [{"role": "user", "content": prompt}],
                    stream=True
                )
                for event in stream:
                    if event.choices[0].delta.content is not None:
                        full_response += event.choices[0].delta.content
                        placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)

            except Exception as e:
                placeholder.error(f"Error: {e}")
                return

        # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": full_response})

        # Display token usage
        total_tokens = count_tokens(buffer) + len(full_response.split())
        st.info(f"💡 Total tokens sent to LLM: {total_tokens}")

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "💡 Default model is **gpt-4o-mini** for speed/cost. "
        "Enable **gpt-4o** for higher quality."
    )
