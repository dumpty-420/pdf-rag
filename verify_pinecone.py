import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

def verify_pinecone_stats(index_name="pdf-rag-index"):
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        print("PINECONE_API_KEY not found.")
        return

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    
    try:
        stats = index.describe_index_stats()
        print(f"Stats for index '{index_name}':")
        print(stats)
    except Exception as e:
        print(f"Error fetching stats: {e}")

if __name__ == "__main__":
    verify_pinecone_stats()
