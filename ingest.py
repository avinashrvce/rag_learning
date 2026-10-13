
import hashlib
import os
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_DIR = Path(__file__).resolve().parent
PDF_PATH = PROJECT_DIR / "Documents" / "sample.pdf"
DB_PATH = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "rag_documents"


def ingest_pdf(
    pdf_path: Path = PDF_PATH,
    source_name: str | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> int:
    load_dotenv(PROJECT_DIR / ".env")

    if not os.getenv("GOOGLE_API_KEY", "").strip():
        raise ValueError("GOOGLE_API_KEY is missing from .env")

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print("Loading PDF...")
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    print(f"Pages loaded: {len(documents)}")

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = [
        chunk
        for chunk in text_splitter.split_documents(documents)
        if chunk.page_content.strip()
    ]
    print(f"Chunks created: {len(chunks)}")

    if not chunks:
        raise ValueError("No extractable text found in PDF.")

    if source_name:
        for chunk in chunks:
            chunk.metadata["source"] = source_name

    print("Initializing Gemini embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    print("Generating and storing embeddings...")
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DB_PATH),
    )

    pdf_hash = hashlib.sha256(pdf_path.read_bytes()).hexdigest()[:16]
    ids = [f"{pdf_hash}-chunk-{i}" for i in range(len(chunks))]

    batch_size = 20
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        vectorstore.add_documents(
            documents=chunks[start:end],
            ids=ids[start:end],
        )
        print(f"Indexed {min(end, len(chunks))}/{len(chunks)} chunks")
        if progress_callback:
            progress_callback(min(end, len(chunks)), len(chunks))

    print("\nIngestion completed successfully!")
    print(f"Total stored vectors: {vectorstore._collection.count()}")
    print(f"Database location: {DB_PATH}")
    return len(chunks)


def main() -> None:
    ingest_pdf()


if __name__ == "__main__":
    main()
