import os
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google.genai.errors import ServerError

from ingest import DB_PATH, ingest_pdf
from rag import answer_question, create_rag_components

PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR / ".env")


@st.cache_resource
def get_rag_components():
    return create_rag_components()


def main() -> None:
    st.set_page_config(
        page_title="Document Q&A",
        page_icon="📚",
        layout="wide",
    )
    st.title("📚 Document Q&A")
    st.caption("Upload a PDF, index it, and ask questions grounded in your documents.")

    with st.sidebar:
        st.header("Knowledge base")
        st.write(
            "Add a PDF to make its contents searchable. "
            "Your indexed documents stay in the local Chroma database."
        )
        uploaded_pdf = st.file_uploader(
            "Choose a PDF",
            type=["pdf"],
            help="The PDF is indexed locally; its contents are sent to Google Gemini for embeddings.",
        )
        if uploaded_pdf and st.button(
            "Index PDF",
            type="primary",
            use_container_width=True,
        ):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    filename = Path(uploaded_pdf.name).name
                    pdf_path = Path(temp_dir) / filename
                    pdf_path.write_bytes(uploaded_pdf.getvalue())
                    progress = st.progress(0, text="Preparing document...")

                    def update_progress(completed: int, total: int) -> None:
                        progress.progress(
                            completed / total,
                            text=f"Indexing chunks: {completed} of {total}",
                        )

                    chunk_count = ingest_pdf(
                        pdf_path,
                        source_name=filename,
                        progress_callback=update_progress,
                    )
                get_rag_components.clear()
                st.success(f"Indexed {chunk_count} chunks from {filename}.")
            except Exception as exc:
                st.error(f"Could not index this PDF: {exc}")

        st.divider()
        if os.getenv("GOOGLE_API_KEY", "").strip():
            st.success("Gemini API key configured", icon="✅")
        else:
            st.warning("Add GOOGLE_API_KEY to your .env file to get started.")

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []

    if not DB_PATH.exists():
        st.info("Upload and index a PDF from the sidebar to start a conversation.")
    elif not os.getenv("GOOGLE_API_KEY", "").strip():
        st.info("Configure GOOGLE_API_KEY in your .env file to ask questions.")
    else:
        for message in st.session_state.get("messages", []):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("View retrieved document excerpts"):
                        for index, source in enumerate(message["sources"], 1):
                            st.markdown(
                                f"**{index}. {source['name']} · Page {source['page']}**"
                            )
                            st.write(source["content"])

        question = st.chat_input("Ask a question about your documents")
        if question:
            st.session_state.setdefault("messages", []).append({
                "role": "user",
                "content": question,
            })
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("Searching your documents and preparing an answer..."):
                        retriever, answer_chain = get_rag_components()
                        answer, documents = answer_question(
                            question,
                            retriever,
                            answer_chain,
                        )
                    sources = [
                        {
                            "name": Path(
                                doc.metadata.get("source", "Unknown")
                            ).name,
                            "page": doc.metadata.get("page", 0) + 1,
                            "content": doc.page_content,
                        }
                        for doc in documents
                    ]
                    st.markdown(answer)
                    if sources:
                        with st.expander("View retrieved document excerpts"):
                            for index, source in enumerate(sources, 1):
                                st.markdown(
                                    f"**{index}. {source['name']} · Page {source['page']}**"
                                )
                                st.write(source["content"])
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as exc:
                    if isinstance(exc, ServerError) and exc.code == 503:
                        st.error(
                            "Gemini is temporarily overloaded. The request was "
                            "retried, but the service is still unavailable. "
                            "Please try again in a minute."
                        )
                    else:
                        st.error(f"Could not answer this question: {exc}")
                    st.session_state.messages.pop()


if __name__ == "__main__":
    main()
