from langchain_mistralai import ChatMistralAI


def ask_question(vectorstore, question: str):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(question)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
Tu réponds uniquement à partir du contexte ci-dessous.
Si l'information n'est pas dans le contexte, dis clairement que tu ne sais pas.

Contexte :
{context}

Question :
{question}
"""

    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0
    )

    response = llm.invoke(prompt)

    return response.content, relevant_docs