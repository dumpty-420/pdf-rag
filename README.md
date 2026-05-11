# PDF RAG Assistant 📚

A Retrieval-Augmented Generation (RAG) system that lets you ask questions about PDF documents using natural language. Built with FAISS for local vector storage and Google Gemini as the LLM.

---

## What it does

- Ingests multiple PDF files and indexes them locally using FAISS
- Answers natural language questions about the PDFs using Google Gemini
- Supports multi-document search across all indexed papers
- Handles API rate limits with automatic retry logic
- Provides a clean Streamlit chat interface

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini (`GEMINI_API_KEY`) |
| Vector Store | FAISS (local, no cloud needed) |
| Embeddings | HuggingFace (local) |
| Framework | LangChain |
| UI | Streamlit |
| PDF Parsing | LlamaIndex / PyPDF |

---

## Project Structure

```
pdf-rag/
├── app.py                  # Streamlit chat UI
├── ingestion.py            # PDF loading and FAISS index creation
├── ingestion_pinecone.py   # Alternative Pinecone-based ingestion
├── query_engine.py         # FAISS index loader and QA chain builder
├── rag_engine.py           # Core RAG logic
├── prompt.py               # Prompt templates
├── verify_model.py         # Script to verify Gemini model setup
├── verify_pinecone.py      # Script to verify Pinecone connection
├── rag_files/              # Stored FAISS index files
├── requirements.txt        # Python dependencies
└── .env                    # API keys (not committed)
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/dumpty-420/pdf-rag.git
cd pdf-rag
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_google_gemini_api_key
```

### 4. Add your PDFs

Place your PDF files in the project root directory (alongside `app.py`).

### 5. Ingest PDFs

```bash
python ingestion.py
```

This creates a local FAISS index in `./storage_index/`.

### 6. Run the app

```bash
streamlit run app.py
```

---

## How it works

1. **Ingestion** — PDFs are loaded, split into chunks, embedded using HuggingFace models, and stored in a local FAISS index
2. **Query** — When you ask a question, the query is embedded and matched against the FAISS index to retrieve relevant chunks
3. **Generation** — The retrieved chunks are passed to Google Gemini as context, which generates a grounded answer
4. **Rate limiting** — Automatic retry with exponential backoff handles Gemini API quota limits

---

## Sample Documents

The repo includes 5 World Economics papers (`World_Econ_Paper_1.pdf` through `World_Econ_Paper_5.pdf`) as example PDFs to query against.

---

## Notes

- The FAISS index is stored locally — no cloud vector DB required
- An alternative Pinecone-based ingestion pipeline (`ingestion_pinecone.py`) is also included if you prefer cloud storage
- `.env` is gitignored — never commit your API keys

---

## Author

Built by [Seerat Chugh](https://github.com/dumpty-420)
