from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader


def load_documents(folder_path: str):
    docs = []
    folder = Path(folder_path)

    for file in folder.iterdir():
        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
            docs.extend(loader.load())
        elif file.suffix.lower() == ".docx":
            loader = Docx2txtLoader(str(file))
            docs.extend(loader.load())

    return docs