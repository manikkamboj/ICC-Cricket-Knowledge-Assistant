from fastapi import FastAPI,Query
from RAG_Service import rag
api=FastAPI()


@api.get("/bot")
def bot(query:str):
    return rag(query)



