# PDF RAG Assistant 📚

A Retrieval-Augmented Generation (RAG) system that lets you ask natural language questions about PDF documents. Built with FAISS for local vector storage and Google Gemini as the LLM — no cloud vector database required.

---

## What it does

- Ingests multiple PDF files and indexes them locally using FAISS
- Answers natural language questions about the PDFs using Google Gemini
- Supports multi-document search across all indexed papers
- Handles API rate limits with automatic retry logic
- Provides a clean Streamlit chat interface with source references
- Includes an optional Pinecone-based ingestion pipeline for cloud storage

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Google Gemini (`GEMINI_API_KEY`) |
| Vector Store | FAISS (local, no cloud needed) |
| Embeddings | HuggingFace (local) |
| Framework | LangChain |
| UI | Streamlit |
| PDF Parsing | PyPDF |

---

## Project Structure & File Explanations

```
pdf-rag/
├── app.py
├── ingestion.py
├── ingestion_pinecone.py
├── prompt.py
├── query_engine.py
├── rag_engine.py
├── verify_model.py
├── verify_pinecone.py
├── requirements.txt
├── rag_files/
│   └── (FAISS index files stored here)
├── World_Econ_Paper_1.pdf
├── World_Econ_Paper_2.pdf
├── World_Econ_Paper_3.pdf
├── World_Econ_Paper_4.pdf
└── World_Econ_Paper_5.pdf
```

### `app.py`
The main Streamlit entry point — run with streamlit run app.py. Renders a chat interface titled "World Econ Paper Q&A (FAISS Edition)" with a sidebar that loads GEMINI_API_KEY from .env (or prompts for it manually) and a "Process PDFs (Ingest)" button that runs ingestion.py as a subprocess so users never need to touch the terminal. Before answering, checks that ./storage_index exists and shows a helpful error if not. Queries are passed to get_qa_chain_from_index() from query_engine.py, with up to 3 retries and progressive backoff (10s, 20s, 30s) on rate limit errors. Displays only the plain text answer — no source references are shown in the active code (that feature exists only in a commented-out earlier version). On unexpected errors, renders the full Python traceback via st.code() for easier debugging. Contains two commented-out earlier variants of the app preserved from previous experimentation.

### `ingestion.py`
Responsible for loading PDFs and building a local FAISS vector index using LlamaIndex. It reads all PDF files from the rag_files/ folder page by page using pypdf, splits them into chunks via SentenceSplitter (chunk size 128, overlap 50), generates embeddings using the BAAI/bge-small-en-v1.5 HuggingFace model (384 dimensions), and inserts documents in batches of 5 with rate-limiting delays. The final index is saved to the storage_index/ folder. Run this once before starting the app to prepare the index.

### `ingestion_pinecone.py`
An alternative ingestion pipeline that stores embeddings in Pinecone (cloud) instead of FAISS (local). Reads PDFs page by page from rag_files/ using pypdf, generates embeddings with BAAI/bge-small-en-v1.5 (384 dimensions), and upserts them into a Pinecone serverless index (pdf-rag-index by default) on AWS us-east-1 using cosine similarity. Includes smart index management — if the index already exists with the wrong dimension, it deletes and recreates it. Ingests in batches of 2 with a 1-second delay between batches, and a 60-second backoff + one automatic retry on failure. Requires PINECONE_API_KEY in your .env file.

### `prompt.py`
Contains two prompt templates used by the RAG pipeline. QA_PROMPT is a ChatPromptTemplate that instructs the LLM to answer strictly based on retrieved PDF context, with explicit rules against hallucination and repetition, plus a few-shot example for guidance. QUERY_GEN_PROMPT is a PromptTemplate for multi-query retrieval — it rewrites the user's question into 3 alternative versions to improve vector search coverage. Both prompts use langchain_core and are LLM-agnostic. The file also contains several older commented-out prompt variants left over from experimentation.

### `query_engine.py`
Loads the saved FAISS index from storage_index/ and builds the LangChain QA chain. Defines get_query_engine() which reconstructs the LlamaIndex VectorStoreIndex using BAAI/bge-small-en-v1.5 embeddings, and get_qa_chain_from_index() which wraps the LlamaIndex retriever in a custom LlamaIndexRetrieverWrapper (to make it LangChain-compatible) and connects it to a local Ollama llama3.2 model via an LCEL chain. The file also contains two large commented-out variants — one using Google Gemini and one using a local HuggingFace Qwen2.5-0.5B model — preserved from earlier experimentation.

### `rag_engine.py`
he core RAG logic for the in-memory pipeline. load_and_index_pdfs() reads all PDFs from a given directory using LlamaIndex's SimpleDirectoryReader, builds a fresh VectorStoreIndex in memory using BAAI/bge-small-en-v1.5 embeddings (no pre-built index is loaded). get_qa_chain() wraps the LlamaIndex retriever in a LlamaIndexRetrieverWrapper and connects it to a local Ollama llama3.2 model via an LCEL chain using QA_PROMPT. Includes a __main__ block for standalone testing. A commented-out variant using HuggingFacePipeline with Qwen2.5-0.5B is preserved from earlier experimentation. There is no retry logic in this file.

### `verify_model.py`
A standalone script to verify that Ollama is running locally and the llama3.2 model is accessible. Loads the model via ChatOllama, sends a hardcoded "Hello, who are you?" prompt, and prints the response — confirming the local LLM setup works end-to-end before running the main app. No API key is required. Also contains a commented-out variant that previously tested Qwen/Qwen2.5-0.5B-Instruct via HuggingFacePipeline forced to run on CPU.

### `verify_pinecone.py`
A standalone diagnostic script to verify your Pinecone index is reachable and populated. Reads PINECONE_API_KEY from a .env file, connects to Pinecone, and calls describe_index_stats() on the pdf-rag-index index (the default name, overridable via parameter) — printing the vector count and namespace stats. Does not create or modify the index in any way. Only relevant if using ingestion_pinecone.py instead of the default local FAISS pipeline.

### `requirements.txt`
Lists Python dependencies for the project. Install with pip install -r requirements.txt. Covers: Streamlit (UI), LangChain + LangChain Google GenAI (LLM chaining), LlamaIndex core + FAISS + Pinecone + HuggingFace integrations (indexing and retrieval), pypdf (PDF reading), sentence-transformers + transformers + accelerate (embeddings and local models), and python-dotenv (env vars). Note: langchain-ollama is missing from this file despite being actively used in rag_engine.py, query_engine.py, and verify_model.py — you'll need to install it separately with pip install langchain-ollama.

### `rag_files/`
Stores the source PDF files that get ingested into the vector index. This repo ships with 5 pre-loaded World Economy papers (World_Econ_Paper_1-5.pdf) in this folder. ingestion.py reads PDFs from here as its input. To use your own documents, replace or add PDFs to this folder and re-run ingestion. The built FAISS index is saved separately to storage_index/, not here.

### `World_Econ_Paper_1-5.pdf`
Five sample World Economics research papers included as example documents to query against. You can replace these with your own PDFs.

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/dumpty-420/pdf-rag.git
cd pdf-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_google_gemini_api_key
```

Get your key from [Google AI Studio](https://aistudio.google.com/).

### 5. Add your PDFs

Place your PDF files in the project root directory (alongside `app.py`). The sample World Economics papers are already included.

### 6. Run ingestion to build the FAISS index

```bash
python ingestion.py
```

This creates the FAISS index in `rag_files/`. Only needs to be run once, or whenever you add new PDFs.

### 7. Verify your setup (optional)

```bash
python verify_model.py      # test Gemini API connection
python verify_pinecone.py   # test Pinecone connection (only if using Pinecone)
```

### 8. Run the app

```bash
streamlit run app.py
```

---

## How It Works

1. **Ingestion** — PDFs are loaded, split into chunks, embedded using HuggingFace models, and stored locally in a FAISS index
2. **Query** — When you ask a question, it is embedded and matched against the FAISS index to retrieve the most relevant chunks
3. **Generation** — The retrieved chunks are passed to Google Gemini as context, which generates a grounded, cited answer
4. **Rate Limiting** — Automatic retry with exponential backoff handles Gemini API quota limits gracefully

---

## Using Pinecone Instead of FAISS

If you prefer cloud-based vector storage, use the Pinecone pipeline:

1. Add your Pinecone key to `.env`:
```env
PINECONE_API_KEY=your_pinecone_api_key
```

2. Run Pinecone ingestion instead:
```bash
python ingestion_pinecone.py
```

---

## Sample Questions

```
What are the key findings of the World Economics papers?
What factors affect global economic growth?
How does inflation impact developing economies?
What trade policies are discussed across the papers?
Summarize the main arguments in the papers.
```

---

## Notes

- The FAISS index is stored locally — no cloud vector DB required by default
- `.env` is gitignored — never commit your API keys
- Add your own PDFs to the root directory and re-run `ingestion.py` to index them

---

## Author

Built by [Seerat Chugh](https://github.com/dumpty-420)
