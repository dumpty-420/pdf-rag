# import os
# import faiss
# from llama_index.core import (
#     VectorStoreIndex,
#     StorageContext,
#     load_index_from_storage,
#     Settings,
# )
# from llama_index.vector_stores.faiss import FaissVectorStore
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain.chains import create_retrieval_chain
# # from langchain.chains.combine_documents import create_stuff_documents_chain

# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser

# from langchain_core.retrievers import BaseRetriever
# from langchain_core.documents import Document as LCDocument
# from typing import List, Any

# from prompt import QA_PROMPT


# def get_query_engine(api_key):
#     Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

#     persist_dir = "./storage_index"
#     if not os.path.exists(persist_dir):
#         raise FileNotFoundError(
#             f"Index not found at {persist_dir}. Please run ingestion first."
#         )

#     storage_context = StorageContext.from_defaults(
#         persist_dir=persist_dir,
#         vector_store=FaissVectorStore.from_persist_dir(persist_dir),
#     )
#     index = load_index_from_storage(storage_context)
#     return index


# class LlamaIndexRetrieverWrapper(BaseRetriever):
#     vector_retriever: Any

#     def _get_relevant_documents(
#         self, query: str, *, run_manager=None
#     ) -> List[LCDocument]:
#         nodes = self.vector_retriever.retrieve(query)
#         return [
#             LCDocument(page_content=node.get_content(), metadata=node.metadata)
#             for node in nodes
#         ]


# def get_qa_chain_from_index(index, api_key):
#     vector_retriever = index.as_retriever(similarity_top_k=5)
#     retriever = LlamaIndexRetrieverWrapper(vector_retriever=vector_retriever)

#     # Use Gemini instead of local Qwen (avoids segfault from memory overload)
#     from langchain_google_genai import ChatGoogleGenerativeAI
#     from langchain_core.runnables import RunnablePassthrough
#     from langchain_core.output_parsers import StrOutputParser

#     llm = ChatGoogleGenerativeAI(
#         model="gemini-1.5-flash-latest",
#         google_api_key=api_key,
#         temperature=0.3,
#     )

#     def format_docs(docs):
#         return "\n\n".join(doc.page_content for doc in docs)

#     qa_chain = (
#         {"context": retriever | format_docs, "input": RunnablePassthrough()}
#         | QA_PROMPT
#         | llm
#         | StrOutputParser()
#     )

#     return qa_chain


import os
import faiss
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as LCDocument
from typing import List, Any

from prompt import QA_PROMPT


def get_query_engine(api_key):
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    persist_dir = "./storage_index"
    if not os.path.exists(persist_dir):
        raise FileNotFoundError(
            f"Index not found at {persist_dir}. Please run ingestion first."
        )

    storage_context = StorageContext.from_defaults(
        persist_dir=persist_dir,
        vector_store=FaissVectorStore.from_persist_dir(persist_dir),
    )
    index = load_index_from_storage(storage_context)
    return index


class LlamaIndexRetrieverWrapper(BaseRetriever):
    vector_retriever: Any

    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[LCDocument]:
        # Handle both string and dict input
        if isinstance(query, dict):
            query = query.get("question", query.get("input", str(query)))
        
        nodes = self.vector_retriever.retrieve(query)
        return [
            LCDocument(page_content=node.get_content(), metadata=node.metadata)
            for node in nodes
        ]


def get_qa_chain_from_index(index, api_key):
    vector_retriever = index.as_retriever(similarity_top_k=5)
    retriever = LlamaIndexRetrieverWrapper(vector_retriever=vector_retriever)

    llm = ChatOllama(
    model="llama3.2",
    temperature=0.3,
    top_p=0.9,
    top_k=40,
    num_predict=512,   # max output tokens
    num_ctx=4096,      # context window (input token limit)
    repeat_penalty=1.1,
)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    qa_chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
            "question": RunnableLambda(lambda x: x["question"]),
        }
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )

    return qa_chain

# import os
# import faiss
# from llama_index.core import (
#     VectorStoreIndex,
#     StorageContext,
#     load_index_from_storage,
#     Settings,
# )
# from llama_index.vector_stores.faiss import FaissVectorStore
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains import RetrievalQA

# from prompt import QA_PROMPT


# def get_query_engine(api_key):

#     # Set up LlamaIndex Settings
#     Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

#     # Load Index from Storage
#     persist_dir = "./storage_index"
#     if not os.path.exists(persist_dir):
#         raise FileNotFoundError(
#             f"Index not found at {persist_dir}. Please run ingestion first."
#         )

#     # Reconstruct Vector Store
#     storage_context = StorageContext.from_defaults(
#         persist_dir=persist_dir,
#         vector_store=FaissVectorStore.from_persist_dir(persist_dir),
#     )
#     index = load_index_from_storage(storage_context)

#     return index


# from langchain_core.retrievers import BaseRetriever
# from typing import List, Any
# from langchain_core.documents import Document as LCDocument


# class LlamaIndexRetrieverWrapper(BaseRetriever):
#     vector_retriever: Any

#     def _get_relevant_documents(
#         self, query: str, *, run_manager=None
#     ) -> List[LCDocument]:
#         nodes = self.vector_retriever.retrieve(query)
#         docs = []
#         for node in nodes:
#             docs.append(
#                 LCDocument(page_content=node.get_content(), metadata=node.metadata)
#             )
#         return docs


# def get_qa_chain_from_index(index, api_key):
#     # Create LangChain Retriever
#     vector_retriever = index.as_retriever(similarity_top_k=5)
#     retriever = LlamaIndexRetrieverWrapper(vector_retriever=vector_retriever)

#     # Create LangChain LLM
#     from langchain_huggingface import HuggingFacePipeline

#     # Use a small, efficient open-source model: Qwen/Qwen2.5-0.5B-Instruct
#     # Switching to 0.5B to avoid memory-related segmentation faults
#     print("Loading local Qwen2.5-0.5B-Instruct model...")
#     llm = HuggingFacePipeline.from_model_id(
#         model_id="Qwen/Qwen2.5-0.5B-Instruct",
#         task="text-generation",
#         pipeline_kwargs={
#             "max_new_tokens": 512,
#             "do_sample": True,
#             "temperature": 0.3,
#             "top_k": 50,
#             "top_p": 0.95,
#             "return_full_text": False,
#         },
#         model_kwargs={"torch_dtype": "auto", "low_cpu_mem_usage": True},
#     )

#     # Create QA Chain
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         chain_type_kwargs={"prompt": QA_PROMPT},
#         return_source_documents=True,
#     )
#     return qa_chain
