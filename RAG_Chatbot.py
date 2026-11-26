
import streamlit as st
from RAG_Service import rag


# 1. Page Configuration

st.set_page_config(
    page_title="ICC Cricket Knowledge Assistant",
    page_icon="🏏",
    layout="wide"
)


# 2. Custom Styling

st.markdown("""
<style>
.stApp {
    background-color: #f7f9fc;
}

.main-title {
    text-align: center;
    color: #123c69;
    font-size: 32px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 15px;
}
</style>
""", unsafe_allow_html=True)


# 3. Application Header

st.markdown(
    '<div class="main-title">🏏 ICC Cricket Knowledge Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions about ODI, T20 and Test cricket</div>',
    unsafe_allow_html=True
)

st.divider()


# 4. Sidebar

with st.sidebar:

    st.header("🏏 Cricket Assistant")

    st.write("Explore three cricket formats:")

    st.markdown("""
    - 🏆 **ODI Cricket**
    - ⚡ **T20 Cricket**
    - 🏏 **Test Cricket**
    """)

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# 5. Initialize Chat History

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# 6. Welcome Message and Suggestions

if not st.session_state.messages:

    st.info(
        "👋 Welcome! Ask me anything about ODI, T20 or Test cricket."
    )

    st.markdown("**Try asking:**")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🏆 ODI Cricket", use_container_width=True):
            st.session_state.pending_query = "Tell me about ODI cricket."
            st.rerun()

    with col2:
        if st.button("⚡ T20 Cricket", use_container_width=True):
            st.session_state.pending_query = "Tell me about T20 cricket."
            st.rerun()

    with col3:
        if st.button("🏏 Test Cricket", use_container_width=True):
            st.session_state.pending_query = "Tell me about Test cricket."
            st.rerun()


# 7. Display Chat History

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# 8. User Input

query = st.chat_input("Ask your cricket question...")

if st.session_state.pending_query:

    query = st.session_state.pending_query
    st.session_state.pending_query = None


# 9. Generate Response

if query:

    st.session_state.messages.append({
        "role": "user",
        "content": query
    })

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.spinner("Finding relevant information..."):

            try:

                response = rag(query)
                answer = response["answer"]

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                st.error(f"Something went wrong: {e}")
