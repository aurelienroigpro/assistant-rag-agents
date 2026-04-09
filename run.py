from dotenv import load_dotenv
from rag.loader import load_documents
from rag.splitter import split_documents
from rag.vectorstore import create_vectorstore
from rag.chain import ask_question

load_dotenv()


def main():
    folder_path = "data/documents"

    print("Chargement des documents...")
    documents = load_documents(folder_path)
    print(f"{len(documents)} document(s) chargé(s)")

    print("Découpage des documents...")
    chunks = split_documents(documents)
    print(f"{len(chunks)} chunk(s) créé(s)")

    print("Création de la base vectorielle...")
    vectorstore = create_vectorstore(chunks)
    print("Base vectorielle prête")

    question = input("\nPose ta question : ")

    answer, sources = ask_question(vectorstore, question)

    print("\nRéponse :")
    print(answer)

    print("\nSources utilisées :")
    for i, doc in enumerate(sources, 1):
        source = doc.metadata.get("source", "source inconnue")
        page = doc.metadata.get("page", "N/A")
        print(f"{i}. {source} - page {page}")


if __name__ == "__main__":
    main()