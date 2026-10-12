# RAG Learning Project

A small retrieval-augmented generation (RAG) project that indexes a PDF into a
local Chroma vector database and uses Google Gemini to answer questions from
the indexed document.

## How it works

1. `ingest.py` loads `Documents/sample.pdf.pdf` and extracts its pages.
2. The page text is split into chunks of up to 1,000 characters, with 200
   characters of overlap.
3. Gemini embeddings are generated for the chunks and stored in the local
   `chroma_db` directory in the `rag_documents` collection.
4. `rag.py` retrieves the four most relevant chunks for each question and asks
   Gemini to answer using that context, including source page numbers where
   possible.

## Requirements

- Python 3.10 or newer
- A Google AI API key with access to the Gemini embedding and chat models used
  by this project

## Setup

Run these commands from the project directory in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create a `.env` file in the project directory and add your API key:

```dotenv
GOOGLE_API_KEY=your_google_ai_api_key
```

Keep the key private. The `.env` file is excluded from version control.

Place the PDF to index at:

```text
Documents/sample.pdf.pdf
```

## Run

From the project directory, activate the virtual environment if needed, then
ingest the PDF:

```powershell
.\.venv\Scripts\Activate.ps1
python ingest.py
```

After ingestion succeeds, start the interactive question-answering program:

```powershell
python rag.py
```

Enter a question at the prompt. Type `exit` or `quit` to stop.

## Configuration and notes

- The embedding model is configured in both scripts as
  `models/gemini-embedding-001`.
- The chat model is configured in `rag.py` as `gemini-3.5-flash`.
- `chroma_db` is local generated data and is excluded from version control. Run
  `ingest.py` before using `rag.py` to populate it.
- Ingestion uses IDs derived from the PDF's content hash. Re-ingesting the
  same PDF updates its existing chunk IDs; changing the PDF creates new IDs.
- `rag.py` currently loads `.env` and opens `chroma_db` relative to the current
  working directory, so run it from the project directory.
