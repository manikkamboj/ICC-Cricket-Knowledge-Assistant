import streamlit as st
import joblib

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_classic.retrievers import EnsembleRetriever
from langchain_cohere import CohereRerank


# ==========================================
# 1. STREAMLIT PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="ICC Cricket Knowledge Assistant",
    page_icon="🏏",
    layout="wide"
)


# ==========================================
# 2. CUSTOM STYLING
# ==========================================

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


# ==========================================
# 3. APPLICATION HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🏏 ICC Cricket Knowledge Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Ask questions about ODI, T20 and Test cricket</div>',
    unsafe_allow_html=True
)

st.divider()


# ==========================================
# 4. LOAD RAG COMPONENTS
# ==========================================

@st.cache_resource
def load_components():

    reranker = CohereRerank(
        model="rerank-v5.0-fast",
        top_n=10
    )

    bm25_retriever = joblib.load(
        "bm25_retriever.pkl"
    )

    emd_model = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    db = FAISS.load_local(
        "ICC",
        embeddings=emd_model,
        allow_dangerous_deserialization=True
    )

    db_retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 10,
            "fetch_k": 20
        }
    )

    final_retriever = EnsembleRetriever(
        retrievers=[
            db_retriever,
            bm25_retriever
        ],
        weights=[0.5, 0.5]
    )

    template = PromptTemplate(
        template="""
You are a helpful assistant.

Answer the following query from the given context only.

Query = {query}
Context = {context}

Note: Do not use your knowledge if context
is not sufficient to answer the query.
Simply say: Information not found.
""",
        input_variables=["query", "context"]
    )

    llm = ChatOpenAI(
        model="gpt-4.1-mini"
    )

    chain = template | llm

    return final_retriever, reranker, chain


# ==========================================
# 5. RAG FUNCTION
# ==========================================

def rag(query):

    final_retriever, reranker, chain = load_components()

    li_doc = final_retriever.invoke(query)

    results = reranker.compress_documents(
        documents=li_doc,
        query=query
    )

    context = ""

    for doc in results:
        current_chunk = doc.page_content
        context += current_chunk
        context += "\n"

    if not context.strip():
        return "Information not found."

    res = chain.invoke({
        "query": query,
        "context": context
    })

    return res.content


# ==========================================
# 6. SIDEBAR
# ==========================================

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


# ==========================================
# 7. INITIALIZE CHAT HISTORY
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ==========================================
# 8. WELCOME MESSAGE & SUGGESTIONS
# ==========================================

if not st.session_state.messages:

    st.info(
        "👋 Welcome! Ask me anything about ODI, T20 or Test cricket."
    )

    st.markdown("**Try asking:**")

    col1, col2, col3 = st.columns(3)

    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    with col1:

        if st.button("🏆 ODI Cricket", use_container_width=True):

            st.session_state.pending_query = (
                "Tell me about ODI cricket."
            )

            st.rerun()

    with col2:

        if st.button("⚡ T20 Cricket", use_container_width=True):

            st.session_state.pending_query = (
                "Tell me about T20 cricket."
            )

            st.rerun()

    with col3:

        if st.button("🏏 Test Cricket", use_container_width=True):

            st.session_state.pending_query = (
                "Tell me about Test cricket."
            )

            st.rerun()


# ==========================================
# 9. DISPLAY CHAT HISTORY
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# ==========================================
# 10. USER INPUT
# ==========================================

query = st.chat_input(
    "Ask your cricket question..."
)


# Handle Suggested Questions

if st.session_state.get("pending_query"):

    query = st.session_state.pending_query
    st.session_state.pending_query = None


# ==========================================
# 11. GENERATE RESPONSE
# ==========================================

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

                st.markdown(response["answer"])

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })

            except Exception as e:

                st.error(f"Something went wrong: {e}")
