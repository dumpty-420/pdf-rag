import os
import time
from dotenv import load_dotenv
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

# Load environment variables
load_dotenv()


def ingest_pdfs_pinecone(directory="rag_files", index_name="pdf-rag-index"):
    # 1. Validate Keys
    pinecone_key = os.getenv("PINECONE_API_KEY")

    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")

    # 2. Configure Settings
    print("Configuring HuggingFace Embeddings...")
    # Using BAAI/bge-small-en-v1.5 (Dimension 384)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    # 3. Initialize Pinecone
    print("Initializing Pinecone...")
    pc = Pinecone(api_key=pinecone_key)

    # 4. Create Index if it doesn't exist
    # 4. Create Index if it doesn't exist or has wrong dimension
    existing_indexes = pc.list_indexes().names()
    if index_name in existing_indexes:
        index_description = pc.describe_index(index_name)
        if index_description.dimension != 384:
            print(
                f"Index '{index_name}' exists but has dimension {index_description.dimension} (expected 384). Deleting..."
            )
            pc.delete_index(index_name)
            while index_name in pc.list_indexes().names():
                time.sleep(1)
            print("Index deleted.")
            existing_indexes = pc.list_indexes().names()  # Refresh list

    if index_name not in existing_indexes:
        print(f"Creating Pinecone index '{index_name}'...")
        # Dimensions for BAAI/bge-small-en-v1.5 is 384
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Waiting for index to be ready...")
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)
    else:
        print(f"Pinecone index '{index_name}' already exists with correct dimension.")

    pinecone_index = pc.Index(index_name)

    # 5. Create Vector Store
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 6. Load documents
    print(f"Loading PDFs from {directory}...")

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

    # 7. Ingest in Batches (Rate Limited)
    print("Indexing documents to Pinecone (with rate limiting)...")

    # Initialize index linked to Pinecone
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

    batch_size = 2
    for i in tqdm(range(0, len(documents), batch_size)):
        batch = documents[i : i + batch_size]
        try:
            nodes = Settings.node_parser.get_nodes_from_documents(batch)
            index.insert_nodes(nodes)
            print(f"Inserted batch {i // batch_size + 1}")
            time.sleep(
                1
            )  # Reduced delay as we aren't hitting Gemini API for embeddings
        except Exception as e:
            print(
                f"Error inserting batch starts at {i}: {e}. Waiting 60s before retry."
            )
            time.sleep(60)  # Increased backoff
            try:
                # Retry once
                nodes = Settings.node_parser.get_nodes_from_documents(batch)
                index.insert_nodes(nodes)
                print(f"Retry success for batch {i // batch_size + 1}")
            except Exception as retry_e:
                print(f"Retry failed for batch {i}: {retry_e}")

    print("Ingestion to Pinecone complete!")


if __name__ == "__main__":
    try:
        ingest_pdfs_pinecone()
    except Exception as e:
        print(f"Error: {e}")
