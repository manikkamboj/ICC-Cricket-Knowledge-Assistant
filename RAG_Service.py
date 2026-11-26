from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import joblib
from langchain_classic.retrievers import EnsembleRetriever
from langchain_cohere import CohereRerank

reranker = CohereRerank(model="rerank-v5.0-fast", top_n=10)

bm25_retriever = joblib.load('bm25_retriever.pkl')

emd_model = OpenAIEmbeddings(
    model='text-embedding-3-small'
)

db = FAISS.load_local(
    'ICC',
    embeddings=emd_model,
    allow_dangerous_deserialization=True
)

db_retriver = db.as_retriever(
    search_type='mmr',
    search_kwargs={'k': 10, 'fetch_k': 20}
)

final_retriever = EnsembleRetriever(
    retrievers=[db_retriver, bm25_retriever],
    weights=[.5, .5]
)

template = PromptTemplate(
    template='''
You are a helpful assistant.

Answer the following query from the given context only
Query ={query}
Context={context}

Note: Do not use your knowledge if context is not sufficient to answer the query simply say, information not found

''',
    input_variables=['query', 'context']
)

llm = ChatOpenAI(
    model='gpt-4.1-mini',
    temperature=0
)

chain = template | llm

def rag(query):
    li_doc = final_retriever.invoke(query)
    results = reranker.compress_documents(documents=li_doc, query=query)

    context = ''
    for doc in results:
        current_chunk = doc.page_content
        context += current_chunk
        context += '\n'

    if not context.strip():
        return {'answer': 'Information not found.'}

    res = chain.invoke({'query': query, 'context': context})
    return {'answer': res.content}
