from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import FAISS


def create_vectorstore(chunks):
    embeddings = MistralAIEmbeddings(model="mistral-embed")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore