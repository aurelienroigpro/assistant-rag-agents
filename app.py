# ===== IMPORTS =====
import streamlit as st
import datetime
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())


# ===== TITRE =====
st.title("Agent RAG + Calcul")


# ===== CHARGEMENT DES DOCUMENTS (UNE FOIS) =====
@st.cache_resource
def load_pipeline():

    # Load PDFs
    loader1 = PyPDFLoader("PDF_cours_informatique/IUT.pdf")
    loader2 = PyPDFLoader("PDF_cours_informatique/Architecture-ordinateur.pdf")
    loader3 = PyPDFLoader("PDF_cours_informatique/assembleur.pdf")
    loader4 = PyPDFLoader("PDF_cours_informatique/constitution-ordinateur.pdf")

    pages = [
        *loader1.load(),
        *loader2.load(),
        *loader3.load(),
        *loader4.load()
    ]

    # Nettoyage
    def clean_text(text):
        text = text.replace("\xa0", " ").replace("\n", " ")
        return " ".join(text.split())

    for doc in pages:
        doc.page_content = clean_text(doc.page_content)

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(pages)

    # Embeddings + DB
    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(chunks, embedding)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini")

    # Prompt
    prompt = ChatPromptTemplate.from_template("""
    Réponds à la question en utilisant uniquement le contexte ci-dessous.

    Contexte:
    {context}

    Question:
    {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": lambda x: x}
        | prompt
        | llm
    )

    return rag_chain, retriever, llm


rag_chain, retriever, llm = load_pipeline()


# ===== TOOLS =====
def calculator_tool(question: str):
    try:
        expression = re.findall(r"[0-9\+\-\*/\.\(\)]+", question)
        expression = "".join(expression)
        return str(eval(expression))
    except:
        return None


def date_tool():
    now = datetime.datetime.now()
    return now.strftime("Nous sommes le %d/%m/%Y, %A")


def rag_tool(question: str):
    response = rag_chain.invoke(question)
    docs = retriever.invoke(question)
    return response.content, docs


# ===== AGENT =====
def ask(question: str):

    decision_prompt = f"""
    Tu dois choisir quel outil utiliser :

    - calculator
    - date
    - rag

    Question: {question}

    Réponds uniquement par un mot.
    """

    decision = llm.invoke(decision_prompt).content.lower()

    if "calculator" in decision:
        result = calculator_tool(question)
        return f"🧮 Résultat : {result}"

    elif "date" in decision:
        return f"📅 {date_tool()}"

    else:
        answer, docs = rag_tool(question)
        sources = "\n".join(str(doc.metadata) for doc in docs)

        return f"""
📚 Réponse :
{answer}

📄 Sources :
{sources}
"""


# ===== INTERFACE =====
question = st.text_input("Pose ta question :")

if st.button("Envoyer"):
    if question:
        with st.spinner("Réflexion en cours..."):
            response = ask(question)
        st.write(response)
        