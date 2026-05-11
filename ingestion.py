import os
import faiss
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter

# Load environment variables
load_dotenv()


def ingest_pdfs(directory="rag_files"):
    # Configure Settings
    # Use BAAI/bge-small-en-v1.5 which is small, fast, and good for RAG.
    # It produces 384 dimensional vectors.
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.node_parser = SentenceSplitter(
        chunk_size=128,  # smaller chunks = more precise retrieval
        chunk_overlap=50,  # overlap to avoid losing context at boundaries
    )

    # Dimensions of BAAI/bge-small-en-v1.5 embeddings is 384
    d = 384
    faiss_index = faiss.IndexFlatL2(d)

    # Create Vector Store
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load documents
    # Load documents
    print(f"Loading PDFs from {directory}...")

    # Custom PDF Loader using pypdf directly
    from pypdf import PdfReader
    from llama_index.core import Document
    import glob

    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))
    if not pdf_files:
        print("No PDF files found.")
        return

    documents = []
    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")
        try:
            reader = PdfReader(pdf_file)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                # Clean text basics
                if text:
                    doc = Document(
                        text=text,
                        metadata={
                            "file_name": os.path.basename(pdf_file),
                            "page_label": str(page_num + 1),
                        },
                    )
                    documents.append(doc)
        except Exception as e:
            print(f"Failed to load {pdf_file}: {e}")

    print(f"Loaded {len(documents)} document pages.")

    # Create Index with rate limiting
    print("Indexing documents to FAISS (with rate limiting)...")

    # Initialize empty index first
    index = VectorStoreIndex.from_documents(
        [],
        storage_context=storage_context,
    )

    # Process in batches
    batch_size = 5
    import time
    from tqdm import tqdm

    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i : i + batch_size]
        try:
            index.insert_nodes(Settings.node_parser.get_nodes_from_documents(batch))
            time.sleep(5)  # Respect rate limits
        except Exception as e:
            print(f"Error inserting batch {i}: {e}")
            time.sleep(20)  # Backoff on error
            try:
                index.insert_nodes(Settings.node_parser.get_nodes_from_documents(batch))
            except Exception as retry_e:
                print(f"Retry failed for batch {i}: {retry_e}")

    # Save Index to Disk
    persist_dir = "./storage_index"
    print(f"Saving index to {persist_dir}...")
    index.storage_context.persist(persist_dir=persist_dir)
    print("Ingestion complete!")


if __name__ == "__main__":
    try:
        ingest_pdfs()
    except Exception as e:
        print(f"Error: {e}")
